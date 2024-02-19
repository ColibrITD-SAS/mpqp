from typing import Optional

import numpy as np
from typeguard import typechecked
from statistics import mean

from mpqp.core.instruction.measurement import ComputationalBasis
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.execution.devices import ATOSDevice
from ..connection.qlm_connection import get_QLMaaSConnection
from ..job import Job, JobType, JobStatus
from ..result import Result, Sample, StateVector
from mpqp.qasm import qasm2_to_myqlm_Circuit
from mpqp import QCircuit, Language
from ...tools.errors import QLMRemoteExecutionError

from qat.qlmaas.result import AsyncResult
from qat.core.contexts import QPUContext
from qat.core.qpu.qpu import QPUHandler
from qat.pylinalg import PyLinalg
from qat.clinalg.qpu import CLinalg
from qat.core.wrappers.observable import Observable as QLM_Observable
from qat.plugins.observable_splitter import ObservableSplitter
from qat.core.wrappers.result import Result as QLM_Result
from qat.core.wrappers.circuit import Circuit
from qat.core.wrappers.job import Job as JobQLM
from qat.comm.qlmaas.ttypes import JobStatus as QLM_JobStatus, QLMServiceException


@typechecked
def get_local_qpu(device: ATOSDevice) -> QPUHandler:
    """
    Returns the myQLM local QPU associated with the ATOSDevice given in parameter.

    Args:
        device: ATOSDevice referring to the myQLM local QPU.

    Raises:
        ValueError
    """
    if device.is_remote():
        raise ValueError("Excepted a local device, not the remote QLM")
    if device == ATOSDevice.MYQLM_PYLINALG:
        return PyLinalg()
    return CLinalg()


@typechecked
def generate_state_vector_job(
    myqlm_circuit: Circuit, device: ATOSDevice = ATOSDevice.MYQLM_PYLINALG
) -> tuple[JobQLM, QPUContext]:
    """Generates a myQLM job from the myQLM circuit and selects the right myQLM
    QPU to run on it.

    Args:
        myqlm_circuit: MyQLM circuit of the job.
        device: ATOSDevice on which the user wants to run the job (to know if
            the run in local of remote).

    Returns:
        A myQLM Job and the right myQLM QPUHandler on which it will be submitted.
    """

    if device.is_remote():
        get_QLMaaSConnection()
        from qlmaas.qpus import LinAlg  # type: ignore

        qpu = LinAlg()
    else:
        qpu = get_local_qpu(device)

    return myqlm_circuit.to_job(job_type="SAMPLE"), qpu


@typechecked
def generate_sample_job(myqlm_circuit: Circuit, job: Job) -> tuple[JobQLM, QPUContext]:
    """Generates a myQLM job from the myQLM circuit and job sample info (target,
    shots, ...), and selects the right myQLM QPU to run on it.

    Args:
        myqlm_circuit: MyQLM circuit of the job.
        job: Original mpqp job used to generate the myQLM job.

    Returns:
        A myQLM Job and the right myQLM QPUHandler on which it will be submitted.
    """

    assert job.measure is not None

    myqlm_job = myqlm_circuit.to_job(
        job_type="SAMPLE",
        qubits=job.measure.targets,
        nbshots=job.measure.shots,
    )

    if job.device.is_remote():
        get_QLMaaSConnection()
        from qlmaas.qpus import LinAlg  # type: ignore

        qpu = LinAlg()
    else:
        assert isinstance(job.device, ATOSDevice)
        qpu = get_local_qpu(job.device)

    return myqlm_job, qpu


@typechecked
def generate_observable_job(
    myqlm_circuit: Circuit, job: Job
) -> tuple[JobQLM, QPUContext]:
    """Generates a myQLM job from the myQLM circuit and observable, and selects
    the right myQLM QPU to run on it.

    Args:
        myqlm_circuit: MyQLM circuit of the job.
        job: Original mpqp job used to generate the myQLM job.

    Returns:
        A myQLM Job and the right myQLM QPUHandler on which it will be submitted.
    """

    assert job.measure is not None and isinstance(job.measure, ExpectationMeasure)

    myqlm_job = myqlm_circuit.to_job(
        job_type="OBS",
        observable=QLM_Observable(
            job.measure.nb_qubits, matrix=job.measure.observable.matrix
        ),
        nbshots=job.measure.shots,
    )
    if job.device.is_remote():
        get_QLMaaSConnection()
        from qlmaas.qpus import LinAlg  # type: ignore

        qpu = LinAlg()
    else:
        assert isinstance(job.device, ATOSDevice)
        qpu = ObservableSplitter() | get_local_qpu(job.device)

    return myqlm_job, qpu


@typechecked
def extract_state_vector_result(
    myqlm_result: QLM_Result,
    job: Optional[Job] = None,
    device: ATOSDevice = ATOSDevice.MYQLM_PYLINALG,
) -> Result:
    """Constructs a Result from the result given by the myQLM/QLM run in state
    vector mode.

    Args:
        myqlm_result: Result returned by myQLM/QLM after running of the job.
        job: Original mpqp job used to generate the run. Used to retrieve more
            easily info to instantiate the result.
        device: ATOSDevice on which the job was submitted. Used to know if the
            run was remote or local.

    Returns:
        A Result containing the result info extracted from the myQLM/QLM
        statevector result.
    """
    if job is None:
        nb_qubits = (
            myqlm_result.qregs[0].length
            if device.is_remote()
            else sum(len(qreg.qbits) for qreg in myqlm_result.data.qregs)
        )
        job = Job(JobType.STATE_VECTOR, QCircuit(nb_qubits), device, None)
    else:
        nb_qubits = job.circuit.nb_qubits

    nb_states = 2**nb_qubits
    amplitudes = np.zeros(nb_states, np.complex64)
    probas = np.zeros(nb_states, np.float32)
    for sample in myqlm_result:
        amplitudes[sample._state] = sample.amplitude
        probas[sample._state] = sample.probability

    return Result(job, StateVector(amplitudes, nb_qubits, probas), 0, 0)


@typechecked
def extract_sample_result(
    myqlm_result: QLM_Result,
    job: Optional[Job] = None,
    device: ATOSDevice = ATOSDevice.MYQLM_PYLINALG,
) -> Result:
    """Constructs a Result from the result given by the myQLM/QLM run in sample
    mode.

    Args:
        myqlm_result: Result returned by myQLM/QLM after running of the job.
        job: Original mpqp job used to generate the run. Used to retrieve more
            easily info to instantiate the result.
        device: ATOSDevice on which the job was submitted. Used to know if the
            run was remote or local.

    Returns:
        A Result containing the result info extracted from the myQLM/QLM sample
        result.
    """
    if job is None:
        assert isinstance(myqlm_result.qregs[0].length, int)
        nb_qubits = (
            myqlm_result.qregs[0].length
            if device.is_remote()
            else sum(len(qreg.qbits) for qreg in myqlm_result.raw_data.qregs)
        )
        nb_shots = int(myqlm_result.meta_data["nbshots"])
        job = Job(
            JobType.SAMPLE,
            QCircuit(nb_qubits),
            device,
            BasisMeasure(targets=list(range(nb_qubits)), shots=nb_shots),
        )
    else:
        nb_qubits = job.circuit.nb_qubits
        if job.measure is None:
            raise NotImplementedError("We cannot handle job without measure for now")
        nb_shots = job.measure.shots

    # we here take the average of errors over all samples
    error = mean([sample.err for sample in myqlm_result])
    samples = [
        Sample(
            nb_qubits,
            index=sample._state,
            probability=sample.probability,
            bin_str=str(sample.state)[1:-1],
        )
        for sample in myqlm_result
    ]

    return Result(job, samples, error, nb_shots)


@typechecked
def extract_observable_result(
    myqlm_result: QLM_Result,
    job: Optional[Job] = None,
    device: ATOSDevice = ATOSDevice.MYQLM_PYLINALG,
) -> Result:
    """Constructs a Result from the result given by the myQLM/QLM run in
    observable mode.

    Args:
        myqlm_result: Result returned by myQLM/QLM after running of the job.
        job: Original mpqp job used to generate the run. Used to retrieve more
            easily info to instantiate the result.
        device: ATOSDevice on which the job was submitted. Used to know if the
            run was remote or local.

    Returns:
        A Result containing the result info extracted from the myQLM/QLM
        observable result.
    """
    if job is None:
        if device.is_remote():
            nb_qubits = myqlm_result.data.qregs[0].length
        else:
            nb_qubits = sum(len(qreg.qbits) for qreg in myqlm_result.data.qregs)
        nb_shots = int(myqlm_result.meta_data["nbshots"])
        job = Job(
            JobType.OBSERVABLE,
            QCircuit(nb_qubits),
            device,
            ExpectationMeasure(
                targets=list(range(nb_qubits)),
                observable=Observable(
                    np.zeros((2**nb_qubits, 2**nb_qubits), dtype=np.complex64)
                ),
                shots=nb_shots,
            ),
        )
    else:
        if job.measure is None:
            raise NotImplementedError("We cannot handle job without measure for now")
        nb_shots = job.measure.shots

    return Result(job, myqlm_result.value, myqlm_result.error, nb_shots)


@typechecked
def extract_result(
    myqlm_result: QLM_Result,
    job: Optional[Job] = None,
    device: ATOSDevice = ATOSDevice.MYQLM_PYLINALG,
) -> Result:
    """
    Constructs a Result from the result given by the myQLM/QLM run.

    Args:
        myqlm_result: Result returned by myQLM/QLM after running of the job.
        job: Original mpqp job used to generate the run. Used to retrieve more
            easily info to instantiate the result.
        device: ATOSDevice on which the job was submitted. Used to know if the
            run was remote or local.

    Returns:
        A Result containing the result info extracted from the myQLM/QLM result.
    """

    if (job is None) or job.device.is_remote():
        if myqlm_result.value is None:
            if list(myqlm_result)[0].amplitude is None:
                job_type = JobType.SAMPLE
            else:
                job_type = JobType.STATE_VECTOR
        else:
            job_type = JobType.OBSERVABLE
    else:
        job_type = job.job_type

    if job_type == JobType.STATE_VECTOR:
        return extract_state_vector_result(myqlm_result, job, device)
    elif job_type == JobType.SAMPLE:
        return extract_sample_result(myqlm_result, job, device)
    else:
        return extract_observable_result(myqlm_result, job, device)


@typechecked
def job_pre_processing(job: Job) -> Circuit:
    """Extracts the myQLM circuit and check if ``job.type`` and ``job.measure``
    are coherent.

    Args:
        job: Mpqp job used to instantiate the myQLM circuit.

    Returns:
          The myQLM Circuit translated from the circuit of the job in parameter.
    """

    if (
        job.job_type == JobType.STATE_VECTOR
        and job.measure is not None
        and not isinstance(job.measure, BasisMeasure)
    ):
        raise ValueError("`STATE_VECTOR` jobs require basis measure to be run")
    if job.job_type == JobType.OBSERVABLE and not isinstance(
        job.measure, ExpectationMeasure
    ):
        raise ValueError("`OBSERVABLE` jobs require `ExpectationMeasure` to be run")

    myqlm_circuit = job.circuit.to_other_language(Language.MY_QLM)

    return myqlm_circuit


@typechecked
def run_atos(job: Job) -> Result:
    """Executes the job on the right ATOS device precised in the job in parameter.
    This function is not meant to be used directly, please use
    ``runner.run(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    """
    return run_myQLM(job) if not job.device.is_remote() else run_QLM(job)


@typechecked
def run_myQLM(job: Job) -> Result:
    """Executes the job on the local myQLM simulator. This function is not meant
    to be used directly, please use ``runner.run(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    """

    result = None
    myqlm_job = None
    myqlm_result = None
    qpu = None

    myqlm_circuit = job_pre_processing(job)

    if job.job_type == JobType.STATE_VECTOR:
        myqlm_job, qpu = generate_state_vector_job(myqlm_circuit)

    elif job.job_type == JobType.SAMPLE:
        assert isinstance(job.measure, BasisMeasure)
        if isinstance(job.measure.basis, ComputationalBasis):
            myqlm_job, qpu = generate_sample_job(myqlm_circuit, job)
        else:
            raise NotImplementedError(
                "Does not handle other basis than the ComputationalBasis for the moment"
            )

    elif job.job_type == JobType.OBSERVABLE:
        assert isinstance(job.measure, ExpectationMeasure)
        myqlm_job, qpu = generate_observable_job(myqlm_circuit, job)

    else:
        raise ValueError(f"Job type {job.job_type} not handled")

    job.status = JobStatus.RUNNING
    myqlm_result = qpu.submit(myqlm_job)

    # retrieving the results
    result = extract_result(myqlm_result, job, job.device)  # type: ignore

    job.status = JobStatus.DONE
    return result


@typechecked
def submit_QLM(job: Job) -> tuple[str, AsyncResult]:
    """Submits the job on the remote QLM machine. This function is not meant to
    be used directly, please use ``runner.submit(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        The job_id and the AsyncResult of the submitted job.
    """

    myqlm_job = None
    qpu = None

    myqlm_circuit = job_pre_processing(job)

    if job.device == ATOSDevice.QLM_LINALG:
        if job.job_type == JobType.STATE_VECTOR:
            assert isinstance(job.device, ATOSDevice)
            myqlm_job, qpu = generate_state_vector_job(myqlm_circuit, job.device)

        elif job.job_type == JobType.SAMPLE:
            assert isinstance(job.measure, BasisMeasure)
            if isinstance(job.measure.basis, ComputationalBasis):
                myqlm_job, qpu = generate_sample_job(myqlm_circuit, job)
            else:
                raise NotImplementedError(
                    "Does not handle other basis than the ComputationalBasis for the moment"
                )

        elif job.job_type == JobType.OBSERVABLE:
            assert isinstance(job.measure, ExpectationMeasure)
            myqlm_job, qpu = generate_observable_job(myqlm_circuit, job)

        else:
            raise ValueError(f"Job type {job.job_type} not handled")

        job.status = JobStatus.RUNNING
        async_result = qpu.submit(myqlm_job)
        job_id = async_result.get_info().id
        job.id = job_id
    else:
        raise NotImplementedError(f"Device {job.device} not handled")

    return job_id, async_result


@typechecked
def run_QLM(job: Job) -> Result:
    """Submits the job on the remote QLM machine and waits for it to be done. This
    function is not meant to be used directly, please use ``runner.run(...)``
    instead.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    """

    if not isinstance(job.device, ATOSDevice) or not job.device.is_remote():
        raise ValueError(
            "The job given is not a QLM one, so it cannot be handled by this "
            "function."
        )

    _, async_result = submit_QLM(job)
    qlm_result = async_result.join()

    return extract_result(qlm_result, job, job.device)


@typechecked
def get_result_from_qlm_job_id(job_id: str) -> Result:
    """Retrieve the result, described by the job_id in parameter, from the
    remote QLM and convert it into an mpqp result. If the job is still running,
    we wait (blocking) until it is DONE.

    Args:
        job_id: Id of the remote QLM job.

    Raises:
        QLMRemoteExecutionError

    """
    connection = get_QLMaaSConnection()

    try:
        qlm_job = connection.get_job(job_id)
    except QLMServiceException as e:
        raise QLMRemoteExecutionError(
            f"Job with id {job_id} not found.\nTrace: " + str(e)
        )

    status = qlm_job.get_status()
    if status in [
        QLM_JobStatus.CANCELLED,
        QLM_JobStatus.UNKNOWN_JOB,
        QLM_JobStatus.DELETED,
        QLM_JobStatus.FAILED,
        QLM_JobStatus.STOPPED,
    ]:
        raise QLMRemoteExecutionError(
            f"Trying to retrieve a QLM result for a job in status {status.name}"
        )
    elif status in [QLM_JobStatus.WAITING, QLM_JobStatus.RUNNING]:
        qlm_job.join()

    qlm_result: QLM_Result = qlm_job.get_result()

    return extract_result(qlm_result, None, ATOSDevice.QLM_LINALG)
