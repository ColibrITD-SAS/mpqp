from __future__ import annotations

import math
from collections import Counter
from numbers import Complex
from typing import TYPE_CHECKING

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.execution.connection.quantinuum_connection import get_quantinuum_config
from mpqp.execution.devices import QUANTINUUMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from pytket.backends.backendresult import BackendResult
    from qiskit import QuantumCircuit
    from qnexus.models.references import (
        CircuitRef,
        ExecuteJobRef,
        ExecutionResultRef,
    )


def run_quantinuum(job: Job) -> Result:
    """Execute a job on a Quantinuum Nexus device.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    try:
        _, execute_job_ref = submit_job_quantinuum(job)

        import qnexus as qnx

        execution_status = qnx.jobs.wait_for(execute_job_ref)
        result_refs = qnx.jobs.results(execute_job_ref)
        if not result_refs:
            status = execution_status.status.value
            raise RuntimeError(
                f"Quantinuum Nexus execution job '{execute_job_ref.id}' finished "
                f"with status '{status}', but no result was returned."
            )

        result_ref = result_refs[0]
        if TYPE_CHECKING:
            assert isinstance(result_ref, ExecutionResultRef)

        backend_result = result_ref.download_result()
        if TYPE_CHECKING:
            assert isinstance(backend_result, BackendResult)

        return extract_result(backend_result, job)
    except Exception as error:
        job.status = JobStatus.ERROR
        job.status_message = str(error)
        raise


def submit_job_quantinuum(job: Job) -> tuple[str, "ExecuteJobRef"]:
    """Submit a job to a supported Quantinuum Nexus backend."""
    if not isinstance(job.device, QUANTINUUMDevice):
        raise ValueError(
            "`job` must correspond to a `QUANTINUUMDevice`, but corresponds to "
            f"a {job.device} instead."
        )

    if job.job_type == JobType.SAMPLE:
        if not job.device.supports_samples():
            raise ValueError(f"{job.device} does not support `SAMPLE` jobs.")
        if job.measure is None:
            raise ValueError("Sample jobs must have a measurement.")
    elif job.job_type == JobType.STATE_VECTOR:
        if not job.device.supports_state_vector():
            raise ValueError(f"{job.device} does not support `STATE_VECTOR` jobs.")
    else:
        raise ValueError(f"Job type {job.job_type} not handled on {job.device}.")

    import qnexus as qnx
    from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

    if job.circuit.transpiled_circuit is None:
        tket_circuit = job.circuit.to_other_device(job.device)
    else:
        qiskit_circuit = job.circuit.transpiled_circuit
        if TYPE_CHECKING:
            assert isinstance(qiskit_circuit, QuantumCircuit)
        tket_circuit = qiskit_to_tk(qiskit_circuit)

    backend_config = get_quantinuum_config(job.device)

    name = f"mpqp-{job.job_type.name.lower()}-{job.device.value}"

    uploaded_circuit_ref = qnx.circuits.upload(
        circuit=tket_circuit,
        name=f"{name}-circuit",
    )

    compile_job_ref = qnx.start_compile_job(
        programs=[uploaded_circuit_ref],
        backend_config=backend_config,
        optimisation_level=0,
        name=f"{name}-compilation-job",
    )

    compilation_status = qnx.jobs.wait_for(compile_job_ref)

    compiled_circuit_refs = qnx.jobs.results(compile_job_ref)
    if not compiled_circuit_refs:
        status = compilation_status.status.value
        raise RuntimeError(
            f"Quantinuum Nexus compilation job '{compile_job_ref.id}' finished "
            f"with status '{status}', but no compiled circuit was returned."
        )

    compilation_result_ref = compiled_circuit_refs[0]
    compiled_circuit_ref = compilation_result_ref.get_output()
    if TYPE_CHECKING:
        assert isinstance(compiled_circuit_ref, CircuitRef)

    job.status = JobStatus.RUNNING

    n_shots = job.measure.shots if job.measure is not None else None
    execute_job_ref = qnx.start_execute_job(
        programs=[compiled_circuit_ref],
        backend_config=backend_config,
        n_shots=[n_shots],
        name=f"{name}-execution-job",
    )

    job.id = str(execute_job_ref.id)

    return job.id, execute_job_ref


def extract_state_vector_result(
    amplitudes: "list[Complex] | npt.NDArray[np.complex128]",
    job: Job,
) -> Result:
    """Construct an MPQP result from Quantinuum state-vector amplitudes.

    Args:
        amplitudes: State-vector amplitudes returned by Quantinuum Nexus.
        job: Original MPQP job used for the execution.

    Returns:
        A result containing the state vector and its probabilities.
    """
    state_vector = StateVector(amplitudes, nb_qubits=job.circuit.nb_qubits)
    job.status = JobStatus.DONE
    return Result(job, state_vector, 0, 0)


def extract_sample_result(
    raw_counts: "Counter[tuple[int, ...]]",
    job: Job,
) -> Result:
    """Construct an MPQP result from Quantinuum sample counts.

    Args:
        raw_counts: Number of occurrences of each measured state.
        job: Original MPQP job used for the execution.

    Returns:
        A result containing the samples and their probabilities.
    """
    if job.measure is None:
        raise ValueError("Cannot extract samples without a measurement.")

    samples = [
        Sample(
            bin_str="".join(str(bit) for bit in outcome),
            nb_qubits=job.circuit.nb_qubits,
            count=int(count),
        )
        for outcome, count in raw_counts.items()
    ]

    job.status = JobStatus.DONE
    return Result(job, samples, None, job.measure.shots)


def extract_result(backend_result: "BackendResult", job: Job) -> Result:
    """Convert a Quantinuum Nexus backend result into an MPQP result.

    Args:
        backend_result: Result returned by Quantinuum Nexus.
        job: Original MPQP job used for the execution.

    Returns:
        The MPQP result corresponding to the job type.
    """
    if job.job_type == JobType.STATE_VECTOR:
        return extract_state_vector_result(backend_result.get_state(), job)
    elif job.job_type == JobType.SAMPLE:
        return extract_sample_result(backend_result.get_counts(), job)
    else:
        raise ValueError(f"Job type {job.job_type} not handled on {job.device}.")


def get_result_from_quantinuum_job_id(job_id: str) -> Result:
    """Retrieve a Quantinuum Nexus job result and convert it to an MPQP result."""
    import qnexus as qnx

    job_ref = qnx.jobs.get(id=job_id)
    execution_status = qnx.jobs.wait_for(job_ref)
    result_refs = qnx.jobs.results(job_ref)
    if not result_refs:
        status = execution_status.status.value
        raise RuntimeError(
            f"Quantinuum Nexus execution job '{job_id}' finished with status "
            f"'{status}', but no result was returned."
        )

    backend_config = job_ref.backend_config_store
    if backend_config is None:
        raise ValueError(
            f"Quantinuum Nexus job '{job_id}' does not contain backend "
            "configuration information."
        )

    if isinstance(backend_config, qnx.AerStateConfig):
        backend_result = result_refs[0].download_result()
        amplitudes = backend_result.get_state()
        nb_qubits = int(math.log2(len(amplitudes)))
        job = Job(
            JobType.STATE_VECTOR,
            QCircuit(nb_qubits),
            QUANTINUUMDevice.NEXUS_AER_STATE_SIMULATOR,
        )
        job.id = job_id
        return extract_state_vector_result(amplitudes, job)

    if not isinstance(backend_config, qnx.QuantinuumConfig):
        raise ValueError(
            f"Quantinuum Nexus job '{job_id}' used unsupported backend "
            f"configuration '{type(backend_config).__name__}'."
        )

    device_name = backend_config.device_name
    try:
        device = QUANTINUUMDevice(device_name)
    except ValueError as error:
        raise ValueError(
            f"Quantinuum Nexus job '{job_id}' targeted unsupported device "
            f"'{device_name}'."
        ) from error

    backend_result = result_refs[0].download_result()
    raw_counts = backend_result.get_counts()
    if not raw_counts:
        raise ValueError(f"Quantinuum Nexus job '{job_id}' returned no sample counts.")

    nb_qubits = len(list(raw_counts)[0])
    shots = sum(raw_counts.values())
    circuit = QCircuit(
        [BasisMeasure(list(range(nb_qubits)), shots=shots)],
        nb_qubits=nb_qubits,
    )
    job = Job(JobType.SAMPLE, circuit, device)
    job.id = job_id

    return extract_sample_result(raw_counts, job)
