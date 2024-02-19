from __future__ import annotations

import math
from typing import Optional

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit.providers import BackendV1, BackendV2, JobStatus as IBM_JobStatus
from qiskit.compiler import transpile
from qiskit.primitives import (
    Estimator as Qiskit_Estimator,
    BackendEstimator,
    EstimatorResult,
    SamplerResult,
)
from qiskit.quantum_info import Operator
from qiskit.result import Result as QiskitResult
from qiskit_ibm_provider.job import IBMJob
from qiskit_ibm_runtime import (
    RuntimeJob,
    Sampler as Runtime_Sampler,
    Estimator as Runtime_Estimator,
    Session,
)
from typeguard import typechecked

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.execution.connection.ibm_connection import (
    get_IBMProvider,
    get_QiskitRuntimeService,
)
from mpqp.execution.devices import IBMDevice
from mpqp.execution.job import Job, JobType, JobStatus
from mpqp.execution.result import StateVector, Sample, Result
from mpqp.core.languages import Language
from mpqp.tools.errors import DeviceJobIncompatibleError, IBMRemoteExecutionError


@typechecked
def run_ibm(job: Job) -> Result:
    """
    Executes the job on the right IBM Q device precised in the job in parameter.
    This function is not meant to be used directly, please use ``runner.run(...)`` instead.

    Args:
        job: Job to be executed

    Returns:
        a Result after submission and execution of the job
    """
    return run_aer(job) if not job.device.is_remote() else run_ibmq(job)


@typechecked
def compute_expectation_value(
    ibm_circuit: QuantumCircuit, ibm_backend: Optional[BackendV1 | BackendV2], job: Job
) -> Result:
    """
    Configures observable job and run it locally, and returns the corresponding Result.
    This function is not meant to be used directly, please use ``runner.run(...)`` instead.

    Args:
        ibm_circuit: QuantumCircuit (already reversed bits)
        ibm_backend: The IBM backend (local of remote) on which we execute the job.
        job: Mpqp job describing the observable job to run.

    """
    if not isinstance(job.measure, ExpectationMeasure):
        raise ValueError(
            "Cannot compute expectation value if measure used in job is not of "
            "type ExpectationMeasure"
        )
    nb_shots = job.measure.shots
    qiskit_observable = Operator(job.measure.observable.matrix)

    if nb_shots != 0:
        assert ibm_backend is not None
        estimator = BackendEstimator(backend=ibm_backend)
    else:
        estimator = Qiskit_Estimator()

    # 6M-TODO : think of the possibility to compute several expectation values at the same time when the circuit is
    #  the same apparently the estimator.run() can take several circuits and observables at the same time,
    #  to verify if putting them all together increases the performance

    job.status = JobStatus.RUNNING
    job_expectation = estimator.run(
        [ibm_circuit], [qiskit_observable], shots=nb_shots if nb_shots != 0 else None
    )
    estimator_result = job_expectation.result()
    assert isinstance(job.device, IBMDevice)
    return extract_result(estimator_result, job, job.device)


@typechecked
def check_job_compatibility(job: Job):
    """
    Checks whether the job in parameter has coherent and compatible attributes.

    Args:
        job: Job for which we want to check compatibility.

    Raises:
        DeviceJobIncompatibleError
    """
    if not type(job.measure) in job.job_type.value:
        raise DeviceJobIncompatibleError(
            f"An {job.job_type.name} job is valid only if the corresponding circuit has an measure in "
            f"{list(map(lambda cls: cls.__name__, job.job_type.value))}. "
            f"{type(job.measure).__name__} was given instead."
        )
    if (
        job.job_type == JobType.STATE_VECTOR
        and job.device != IBMDevice.AER_SIMULATOR_STATEVECTOR
    ):
        raise DeviceJobIncompatibleError(
            "Cannot reconstruct state vector with this device. Please use "
            f"{IBMDevice.AER_SIMULATOR_STATEVECTOR} instead"
        )
    if job.device == IBMDevice.AER_SIMULATOR_STATEVECTOR:
        if job.job_type == JobType.SAMPLE:
            raise DeviceJobIncompatibleError(
                "Cannot use sample mode with the statevector simulator."
            )
        if job.job_type == JobType.OBSERVABLE:
            assert job.measure is not None
            if job.measure.shots > 0:
                raise DeviceJobIncompatibleError(
                    "Cannot compute expectation values with non-zero shots"
                    f" with {IBMDevice.AER_SIMULATOR_STATEVECTOR}.\nSet the"
                    " shots to zero to get the exact value, or select "
                    "another device instead"
                )


@typechecked
def run_aer(job: Job):
    """
    Executes the job on the right AER local simulator precised in the job in parameter.
    This function is not meant to be used directly, please use ``runner.run(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    """
    qiskit_circuit = (
        job.circuit.without_measurements().to_other_language(Language.QISKIT)
        if (job.job_type == JobType.STATE_VECTOR)
        else job.circuit.to_other_language(Language.QISKIT)
    )

    qiskit_circuit = qiskit_circuit.reverse_bits()
    check_job_compatibility(job)

    # define backend simulator
    if job.device == IBMDevice.AER_SIMULATOR:
        backend_sim = AerSimulator()
        if job.job_type == JobType.SAMPLE:
            assert job.measure is not None
            run_input = transpile(qiskit_circuit, backend_sim)
            job.status = JobStatus.RUNNING
            job_sim = backend_sim.run(run_input, shots=job.measure.shots)
            result_sim = job_sim.result()
            result = extract_result(result_sim, job, IBMDevice.AER_SIMULATOR)
        elif job.job_type == JobType.OBSERVABLE:
            result = compute_expectation_value(qiskit_circuit, backend_sim, job)
        else:
            raise ValueError(f"Job type {job.job_type} not handled on {job.device}")

    elif job.device == IBMDevice.AER_SIMULATOR_STATEVECTOR:
        if job.job_type == JobType.STATE_VECTOR:
            backend_sim = Aer.get_backend(job.device.value)
            # the save_statevector method is patched on qiskit_aer load, meaning
            # the type checker can't find it. I hate it but it is what it is.
            # this explains the `type: ignore`. This method is needed to get a
            # statevector our of the statevector simulator...
            qiskit_circuit.save_statevector()  # type: ignore
            job.status = JobStatus.RUNNING
            job_sim = backend_sim.run(qiskit_circuit, shots=0)
            result_sim = job_sim.result()
            result = extract_result(
                result_sim, job, IBMDevice.AER_SIMULATOR_STATEVECTOR
            )
        elif job.job_type == JobType.OBSERVABLE:
            result = compute_expectation_value(qiskit_circuit, None, job)
        else:
            raise ValueError(f"job type {job.job_type} not handled on {job.device}")

    else:
        raise ValueError(f"job device {job.device} not handled yet")

    job.status = JobStatus.DONE
    return result


@typechecked
def submit_ibmq(job: Job) -> tuple[str, RuntimeJob | IBMJob]:
    """
    Submits the job on the remote IBM device (quantum computer or simulator).
    This function is not meant to be used directly, please use ``runner.submit(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        The job_id of the submitted job and the RuntimeJob/IBMCircuitJob.
    """
    if job.job_type == JobType.STATE_VECTOR:
        raise DeviceJobIncompatibleError(
            "State vector cannot be computed using IBM remote simulators and"
            " devices. Please use a local simulator instead."
        )

    if job.job_type == JobType.OBSERVABLE:
        if not isinstance(job.measure, ExpectationMeasure):
            raise ValueError(
                "An observable job must is valid only if the corresponding "
                "circuit has an expectation measure."
            )
        if job.measure.shots == 0:
            raise DeviceJobIncompatibleError(
                "Expectation values cannot be computed exactly using IBM remote"
                " simulators and devices. Please use a local simulator instead."
            )
    check_job_compatibility(job)

    qiskit_circuit = job.circuit.to_other_language(Language.QISKIT).reverse_bits()

    service = get_QiskitRuntimeService()
    backend_str = job.device.value
    session = Session(service=service, backend=backend_str)
    if job.job_type == JobType.OBSERVABLE:
        assert isinstance(job.measure, ExpectationMeasure)
        estimator = Runtime_Estimator(session=session)
        qiskit_observable = Operator(job.measure.observable.matrix)

        ibm_job = estimator.run(
            qiskit_circuit, qiskit_observable, shots=job.measure.shots
        )
    elif job.job_type == JobType.SAMPLE:
        assert job.measure is not None
        sampler = Runtime_Sampler(session=session)
        ibm_job = sampler.run(qiskit_circuit, shots=job.measure.shots)
    else:
        raise NotImplementedError(f"{job.job_type} not handled.")

    job.id = ibm_job.job_id()

    return job.id, ibm_job


@typechecked
def run_ibmq(job: Job) -> Result:
    """
    Submits the job on the right IBMQ remote device, precised in the job in parameter, and waits until the job is
    completed.
    This function is not meant to be used directly, please use ``runner.run(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    """
    _, remote_job = submit_ibmq(job)
    ibm_result = remote_job.result()

    assert isinstance(job.device, IBMDevice)
    return extract_result(ibm_result, job, job.device, remote_job)


@typechecked
def extract_result(
    result: QiskitResult | EstimatorResult | SamplerResult,
    job: Optional[Job] = None,
    device: IBMDevice = IBMDevice.AER_SIMULATOR,
    ibm_job: Optional[IBMJob | RuntimeJob] = None,
) -> Result:
    """
    Parses a result from IBM execution (remote or local) into an mpqp Result.
    Depending on which service you run the job (local/remote backend, Estimator, Sampler),
    you retrieve a different result.

    Args:
        result: Result returned by IBM after running of the job.
        job: Original mpqp job used to generate the run. Used to retrieve more easily info to instantiate the result.
        device: IBMDevice on which the job was submitted. Used to know if the run was remote or local
        ibm_job: IBM or Runtime job used to retrieve info about the circuit and the submitted job (in the remote case).

    Returns:
        A Result containing the result info extracted from the IBM result.
    """

    if job is not None and (
        isinstance(result, EstimatorResult) != (job.job_type == JobType.OBSERVABLE)
    ):
        raise ValueError(
            "Mismatch between job type and result type: either the result is an"
            " `EstimatorResult` and the job is od type of both those assertions"
            " are false."
        )

    if isinstance(result, EstimatorResult):
        if job is None:
            job = Job(JobType.OBSERVABLE, QCircuit(0), device)
        shots = 0 if len(result.metadata[0]) == 0 else result.metadata[0]["shots"]
        variance = (
            None if len(result.metadata[0]) == 0 else result.metadata[0]["variance"]
        )
        return Result(job, result.values[0], variance, shots)

    elif isinstance(result, SamplerResult):
        shots = result.metadata[0]["shots"]
        probas = result.quasi_dists[0]
        if job is None:
            if ibm_job is None:
                #  If we don't have access to the remote Sampler RuntimeJob, we determine the number of qubits by taking
                #  the max index in the counts and take the upper power of two. Of course this is not a clean way and
                #  can lead to a lower number of qubit than the real one. We asked IBM support, apparently there is no
                #  way to retrieve the right nb_qubits with SamplerResult only. That is why we encourage to input the
                #  ibm_job to this function
                max_index = max(list(probas.keys()))
                nb_qubits = math.ceil(math.log2(max_index + 1))
            else:
                if isinstance(ibm_job, RuntimeJob):
                    nb_qubits = len(ibm_job.inputs["circuits"][0].qubits)
                else:
                    raise ValueError(
                        f"Expected a RuntimeJob as optional parameter but got an {type(ibm_job)} instead"
                    )
            job = Job(
                JobType.SAMPLE,
                QCircuit(nb_qubits),
                device,
                BasisMeasure(list(range(nb_qubits)), shots=shots),
            )

        data = [
            Sample(
                index=item, probability=probas[item], nb_qubits=job.circuit.nb_qubits
            )
            for item in probas
        ]
        return Result(job, data, None, shots)

    elif isinstance(
        result, QiskitResult
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        if job is None:
            job_data = result.data()
            if "statevector" in job_data:
                job_type = JobType.STATE_VECTOR
                nb_qubits = int(math.log(len(result.get_statevector()), 2))
                job = Job(job_type, QCircuit(nb_qubits), device)
            elif "counts" in job_data:
                job_type = JobType.SAMPLE
                nb_qubits = len(list(result.get_counts())[0])
                shots = result.results[0].shots
                job = Job(
                    job_type,
                    QCircuit(nb_qubits),
                    device,
                    BasisMeasure(list(range(nb_qubits)), shots=shots),
                )
            else:
                if len(result.data()) == 0:
                    raise ValueError(
                        "Result data is empty, cannot extract anything. Check "
                        "if the associated job was successfully completed."
                    )
                else:
                    raise ValueError(
                        f"Data with keys {result.data().keys()} in result not handled."
                    )

        if job.job_type == JobType.STATE_VECTOR:
            vector = result.get_statevector()
            state_vector = StateVector(
                vector.data,  # pyright: ignore[reportGeneralTypeIssues]
                job.circuit.nb_qubits,
            )
            return Result(job, state_vector, 0, 0)
        elif job.job_type == JobType.SAMPLE:
            assert job.measure is not None
            counts = result.get_counts()
            data = [
                Sample(
                    bin_str=item, count=counts[item], nb_qubits=job.circuit.nb_qubits
                )
                for item in counts
            ]
            return Result(job, data, None, job.measure.shots)
        else:
            raise NotImplementedError(f"{job.job_type} not handled.")

    else:
        raise NotImplementedError(f"Result type {type(result)} not handled")


@typechecked
def get_result_from_ibm_job_id(job_id: str) -> Result:
    """
    Retrieves from IBM remote platform and parse the result of the job_id given in parameter.
    If the job is still running, we wait (blocking) until it is DONE.

    Args:
        job_id: Id of the remote IBM job.

    Returns:
        A Result containing the formatted result of the executed ibm job.
    """
    # search for job id in the connector given in parameter first
    # if not found, try with IBMProvider, then QiskitRuntimeService
    # if not found, raise an error
    connector = get_IBMProvider()
    ibm_job = (
        connector.retrieve_job(job_id)
        if job_id in [job.job_id() for job in connector.jobs()]
        else None
    )
    if ibm_job is None:
        connector = get_QiskitRuntimeService()
        ibm_job = (
            connector.job(job_id)
            if job_id in [job.job_id() for job in connector.jobs()]
            else None
        )
    if ibm_job is None:
        raise IBMRemoteExecutionError(
            f"Job with id {job_id} was not found on this account."
        )

    status = ibm_job.status()
    if status in [IBM_JobStatus.CANCELLED, IBM_JobStatus.ERROR]:
        raise IBMRemoteExecutionError(
            f"Trying to retrieve an IBM result for a job in status {status.name}"
        )

    # If the job is finished, it will get the result, if still running it is block until it finishes
    result = ibm_job.result()
    backend = ibm_job.backend()
    assert isinstance(backend, (BackendV1, BackendV2))
    ibm_device = IBMDevice(backend.name)

    return extract_result(result, None, ibm_device, ibm_job)
