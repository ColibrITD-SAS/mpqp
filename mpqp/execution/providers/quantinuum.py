from __future__ import annotations

from typing import TYPE_CHECKING

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.execution.connection.quantinuum_connection import get_quantinuum_config
from mpqp.execution.devices import IBMDevice, QUANTINUUMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample

if TYPE_CHECKING:
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
            status = getattr(execution_status.status, "value", execution_status.status)
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
    """Submit a sample job to a Quantinuum Nexus device."""
    if not isinstance(job.device, QUANTINUUMDevice):
        raise ValueError(
            "`job` must correspond to a `QUANTINUUMDevice`, but corresponds to a "
            f"{job.device} instead."
        )

    if job.job_type != JobType.SAMPLE:
        raise ValueError(f"Job type {job.job_type} not handled on Quantinuum devices.")

    if job.measure is None:
        raise ValueError("Sample jobs must have a measurement.")

    import qnexus as qnx
    from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

    if job.circuit.transpiled_circuit is None:
        qiskit_circuit = job.circuit.to_other_device(IBMDevice.AER_SIMULATOR)
    else:
        qiskit_circuit = job.circuit.transpiled_circuit

    if TYPE_CHECKING:
        assert isinstance(qiskit_circuit, QuantumCircuit)

    tket_circuit = qiskit_to_tk(qiskit_circuit)

    backend_config = get_quantinuum_config(job.device.value)

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
        status = getattr(compilation_status.status, "value", compilation_status.status)
        raise RuntimeError(
            f"Quantinuum Nexus compilation job '{compile_job_ref.id}' finished "
            f"with status '{status}', but no compiled circuit was returned."
        )

    compilation_result_ref = compiled_circuit_refs[0]
    compiled_circuit_ref = compilation_result_ref.get_output()
    if TYPE_CHECKING:
        assert isinstance(compiled_circuit_ref, CircuitRef)

    job.status = JobStatus.RUNNING

    execute_job_ref = qnx.start_execute_job(
        programs=[compiled_circuit_ref],
        backend_config=backend_config,
        n_shots=[job.measure.shots],
        name=f"{name}-execution-job",
    )

    job.id = str(execute_job_ref.id)

    return job.id, execute_job_ref


def extract_result(backend_result: "BackendResult", job: Job) -> Result:
    """Convert Quantinuum Nexus counts into an MPQP result."""
    if job.job_type != JobType.SAMPLE:
        raise ValueError(f"Job type {job.job_type} not handled on Quantinuum devices.")
    if job.measure is None:
        raise ValueError("Cannot extract samples without a measurement.")

    raw_counts = backend_result.get_counts()
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


def get_result_from_quantinuum_job_id(job_id: str) -> Result:
    """Retrieve a Quantinuum Nexus job result and convert it to an MPQP result."""
    import qnexus as qnx

    job_ref = qnx.jobs.get(id=job_id)
    execution_status = qnx.jobs.wait_for(job_ref)
    result_refs = qnx.jobs.results(job_ref)
    if not result_refs:
        status = getattr(execution_status.status, "value", execution_status.status)
        raise RuntimeError(
            f"Quantinuum Nexus execution job '{job_id}' finished with status "
            f"'{status}', but no result was returned."
        )

    backend_config = job_ref.backend_config_store
    device_name = (
        None if backend_config is None else getattr(backend_config, "device_name", None)
    )
    if device_name is None:
        raise ValueError(
            f"Quantinuum Nexus job '{job_id}' does not contain a device name."
        )

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

    nb_qubits = len(next(iter(raw_counts)))
    shots = sum(int(count) for count in raw_counts.values())
    circuit = QCircuit(
        [BasisMeasure(list(range(nb_qubits)), shots=shots)],
        nb_qubits=nb_qubits,
    )
    job = Job(JobType.SAMPLE, circuit, device)
    job.id = job_id

    return extract_result(backend_result, job)
