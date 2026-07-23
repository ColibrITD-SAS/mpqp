from __future__ import annotations

from typing import TYPE_CHECKING

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

# TODO: to enhance docs


def run_quantinuum(job: Job) -> Result:
    """Execute a job on a Quantinuum Nexus device.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    _, execute_job_ref = submit_job_quantinuum(job)

    import qnexus as qnx

    qnx.jobs.wait_for(execute_job_ref)

    result_ref = qnx.jobs.results(execute_job_ref)[0]
    if TYPE_CHECKING:
        assert isinstance(result_ref, ExecutionResultRef)

    backend_result = result_ref.download_result()
    if TYPE_CHECKING:
        assert isinstance(backend_result, BackendResult)

    return extract_result(backend_result, job)


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
        raise ValueError("Sample jobs must have a measure.")

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

    qnx.jobs.wait_for(compile_job_ref)

    compiled_circuit_ref = qnx.jobs.results(compile_job_ref)[0]
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
    """Convert Quantinuum Nexus counts into MPQP Result."""

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
