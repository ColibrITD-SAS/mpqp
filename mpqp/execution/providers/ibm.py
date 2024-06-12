from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.primitives import BackendEstimator, PrimitiveResult, EstimatorResult, Estimator as Qiskit_Estimator
from qiskit.providers import BackendV1, BackendV2
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.result import Result as QiskitResult
from qiskit_aer import Aer, AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Runtime_Estimator, SamplerV2 as Runtime_Sampler, RuntimeJobV2, Session
from typeguard import typechecked

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.core.languages import Language
from mpqp.execution.connection.ibm_connection import get_QiskitRuntimeService, get_backend
from mpqp.execution.devices import IBMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.tools.errors import DeviceJobIncompatibleError, IBMRemoteExecutionError


@typechecked
def run_ibm(job: Job) -> Result:
    """Executes the job on the right IBM Q device precised in the job in
    parameter.

    Args:
        job: Job to be executed.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    return run_aer(job) if not job.device.is_remote() else run_remote_ibm(job)


@typechecked
def compute_expectation_value(
    ibm_circuit: QuantumCircuit, ibm_backend: Optional[BackendV1 | BackendV2], job: Job
) -> Result:
    """Configures observable job and run it locally, and returns the
    corresponding Result.

    Args:
        ibm_circuit: QuantumCircuit (already reversed bits)
        ibm_backend: The IBM backend (local of remote) on which we execute the job.
        job: Mpqp job describing the observable job to run.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    if not isinstance(job.measure, ExpectationMeasure):
        raise ValueError(
            "Cannot compute expectation value if measure used in job is not of "
            "type ExpectationMeasure"
        )
    nb_shots = job.measure.shots
    qiskit_observable = job.measure.observable.to_other_language(Language.QISKIT)
    assert isinstance(qiskit_observable, SparsePauliOp)

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
    """Checks whether the job in parameter has coherent and compatible
    attributes.

    Args:
        job: Job for which we want to check compatibility.

    Raises:
        DeviceJobIncompatibleError: If there is a mismatch between information
            contained in the job (measure and job_type, device and job_type,
            etc...).
    """
    if not type(job.measure) in job.job_type.value:
        raise DeviceJobIncompatibleError(
            f"An {job.job_type.name} job is valid only if the corresponding circuit has an measure in "
            f"{list(map(lambda cls: cls.__name__, job.job_type.value))}. "
            f"{type(job.measure).__name__} was given instead."
        )
    if (
        job.job_type == JobType.STATE_VECTOR
        and job.device not in {IBMDevice.AER_SIMULATOR_STATEVECTOR,
                               IBMDevice.AER_SIMULATOR,
                               IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
                               IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER,}
    ):
        raise DeviceJobIncompatibleError(
            "Cannot reconstruct state vector with this device. Please use "
            f"{IBMDevice.AER_SIMULATOR_STATEVECTOR} instead (or change the job "
            "type, by for example giving a number of shots to a BasisMeasure)."
        )
    if job.device == IBMDevice.AER_SIMULATOR_STATEVECTOR:
        if job.job_type == JobType.OBSERVABLE: # TODO: to check if this is still true
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
    """Executes the job on the right AER local simulator precised in the job in
    parameter.

    Args:
        job: Job to be executed.

    Returns:
        the result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    check_job_compatibility(job)


    qiskit_circuit = (
        job.circuit.without_measurements().to_other_language(Language.QISKIT)
        if (job.job_type == JobType.STATE_VECTOR)
        else job.circuit.to_other_language(Language.QISKIT)
    )
    if TYPE_CHECKING:
        assert isinstance(qiskit_circuit, QuantumCircuit)

    qiskit_circuit = qiskit_circuit.reverse_bits()
    backend_sim = AerSimulator(method=job.device.value)
    run_input = transpile(qiskit_circuit, backend_sim)

    if job.job_type == JobType.STATE_VECTOR:
        # the save_statevector method is patched on qiskit_aer load, meaning
        # the type checker can't find it. I hate it but it is what it is.
        # this explains the `type: ignore`. This method is needed to get a
        # statevector our of the statevector simulator...
        qiskit_circuit.save_statevector()  # type: ignore
        job.status = JobStatus.RUNNING
        job_sim = backend_sim.run(qiskit_circuit, shots=0)
        result_sim = job_sim.result()
        assert isinstance(job.device, IBMDevice)
        result = extract_result(result_sim, job, job.device)

    elif job.job_type == JobType.SAMPLE:
        assert job.measure is not None

        job.status = JobStatus.RUNNING
        job_sim = backend_sim.run(run_input, shots=job.measure.shots)
        result_sim = job_sim.result()
        assert isinstance(job.device, IBMDevice)
        result = extract_result(result_sim, job, job.device)

    elif job.job_type == JobType.OBSERVABLE:
        result = compute_expectation_value(qiskit_circuit, None, job)

    else:
        raise ValueError(f"Job type {job.job_type} not handled.")

    job.status = JobStatus.DONE
    return result


@typechecked
def submit_remote_ibm(job: Job) -> tuple[str, RuntimeJobV2]:
    """Submits the job on the remote IBM device (quantum computer or simulator).

    Args:
        job: Job to be executed.

    Returns:
        IBM's job id and the ``qiskit`` job itself.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
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

    qiskit_circuit = job.circuit.to_other_language(Language.QISKIT)
    if TYPE_CHECKING:
        assert isinstance(qiskit_circuit, QuantumCircuit)
    qiskit_circuit = qiskit_circuit.reverse_bits()
    service = get_QiskitRuntimeService()
    backend = get_backend(job.device.value)
    session = Session(service=service, backend=backend)
    qiskit_circuit = transpile(qiskit_circuit, backend)

    if job.job_type == JobType.OBSERVABLE:
        assert isinstance(job.measure, ExpectationMeasure)
        estimator = Runtime_Estimator(session=session)
        qiskit_observable = job.measure.observable.to_other_language(Language.QISKIT)
        assert isinstance(qiskit_observable, SparsePauliOp)

        # Fills the Pauli strings with identities to make the observable size match the circuit size
        qiskit_observable = SparsePauliOp(
            [pauli._pauli_list[0].tensor(Pauli("I"*(qiskit_circuit.num_qubits - job.measure.observable.nb_qubits)))
             for pauli in qiskit_observable],
            coeffs=qiskit_observable.coeffs
        )

        precision = 1/np.sqrt(job.measure.shots)
        # FIXME: when we precise the target precision like this, it does not give the right number of shots at the end.
        #  Tried once with shots=1234, but got shots=1280 with the real experiment
        ibm_job = estimator.run([(qiskit_circuit, qiskit_observable)], precision=precision)
    elif job.job_type == JobType.SAMPLE:
        assert isinstance(job.measure, BasisMeasure)
        sampler = Runtime_Sampler(session=session)
        ibm_job = sampler.run([qiskit_circuit], shots=job.measure.shots)
    else:
        raise NotImplementedError(f"{job.job_type} not handled.")

    job.id = ibm_job.job_id()

    return job.id, ibm_job


@typechecked
def run_remote_ibm(job: Job) -> Result:
    """Submits the job on the right IBM remote device, precised in the job in
    parameter, and waits until the job is completed.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    _, remote_job = submit_remote_ibm(job)
    ibm_result = remote_job.result()

    assert isinstance(job.device, IBMDevice)
    return extract_result(ibm_result, job, job.device, remote_job)


@typechecked
def extract_result(
    result: QiskitResult | EstimatorResult | PrimitiveResult,
    job: Optional[Job] = None,
    device: Optional[IBMDevice] = IBMDevice.AER_SIMULATOR,
) -> Result:
    """Parses a result from ``IBM`` execution (remote or local) in a ``MPQP``
    :class:`Result<mpqp.execution.result.Result>`.

    Args:
        result: Result returned by IBM after running of the job.
        job: ``MPQP`` job used to generate the run. Enables a more complete
            result.
        device: IBMDevice on which the job was submitted. Used to know if the
            run was remote or local
        ibm_job: Runtime job (V2) used to retrieve info about the circuit and
            the submitted job (in the remote case).

    Returns:
        The ``qiskit`` result converted to our format.
    """

    # If this is a PubResult from primitives V2
    if isinstance(result, PrimitiveResult):
        res_data = result[0].data
        # If we are in observable mode
        if hasattr(res_data, "evs"):
            if job is None:
                job = Job(JobType.OBSERVABLE, QCircuit(0), device)
            expectation = float(res_data.evs)
            error = float(res_data.stds)
            shots = result[0].metadata['shots']
            return Result(job, expectation, error, shots)
        # If we are in sample mode
        else:
            if job is None:
                shots = res_data.meas.num_shots
                nb_qubits = res_data.meas.num_bits
                job = Job(
                    JobType.SAMPLE,
                    QCircuit(nb_qubits),
                    device,
                    BasisMeasure(list(range(nb_qubits)), shots=shots),
                )
            counts = res_data.meas.get_counts()
            data=[
                Sample(
                    bin_str=item, count=counts[item], nb_qubits=job.circuit.nb_qubits
                )
                for item in counts
            ]
            return Result(job, data, None, job.measure.shots)

    else:

        if job is not None and (
            isinstance(result, EstimatorResult) != (job.job_type == JobType.OBSERVABLE)
        ):
            raise ValueError(
                "Mismatch between job type and result type: if the result is an"
                " `EstimatorResult` the job must be of type `OBSERVABLE` but here was not."
            )

        if isinstance(result, EstimatorResult):
            if job is None:
                job = Job(JobType.OBSERVABLE, QCircuit(0), device)
            shots = 0 if len(result.metadata[0]) == 0 else result.metadata[0]["shots"]
            variance = (
                None if len(result.metadata[0]) == 0 else result.metadata[0]["variance"]
            )
            return Result(job, result.values[0], variance, shots)

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
                vector = np.array(result.get_statevector())
                state_vector = StateVector(
                    vector,  # pyright: ignore[reportArgumentType]
                    job.circuit.nb_qubits,
                )
                return Result(job, state_vector, 0, 0)
            elif job.job_type == JobType.SAMPLE:
                assert job.measure is not None
                counts = result.get_counts(0)
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
    """Retrieves from IBM remote platform and parse the result of the job_id
    given in parameter. If the job is still running, we wait (blocking) until it
    is ``DONE``.

    Args:
        job_id: Id of the remote IBM job.

    Returns:
        The result converted to our format.
    """

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
    if status in ["CANCELLED", "ERROR"]:
        raise IBMRemoteExecutionError(
            f"Trying to retrieve an IBM result for a job in status {status.name}"
        )

    # If the job is finished, it will get the result, if still running it is block until it finishes
    result = ibm_job.result()
    backend = ibm_job.backend()
    assert isinstance(backend, (BackendV1, BackendV2))
    ibm_device = IBMDevice(backend.name)

    return extract_result(result, None, ibm_device)
