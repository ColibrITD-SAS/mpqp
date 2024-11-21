from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from qiskit.result import Result as QiskitResult
    from azure.quantum.target.microsoft.result import MicrosoftEstimatorResult

from typeguard import typechecked

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.languages import Language
from mpqp.execution.connection.azure_connection import (
    get_azure_provider,
    get_jobs_by_id,
)
from mpqp.execution.devices import AZUREDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample


@typechecked
def run_azure(job: Job) -> Result:
    """Executes the job on the right AZURE device precised in the job in
    parameter.

    Args:
        job:  The job to be executed, containing the circuit and device information.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    from qiskit import QuantumCircuit

    qiskit_circuit = (
        job.circuit.without_measurements().to_other_language(Language.QISKIT)
        if (job.job_type == JobType.STATE_VECTOR)
        else job.circuit.to_other_language(Language.QISKIT)
    )
    if TYPE_CHECKING:
        assert isinstance(qiskit_circuit, QuantumCircuit)

    qiskit_circuit = qiskit_circuit.reverse_bits()

    backend_sim = get_azure_provider().get_backend(job.device.value)

    if TYPE_CHECKING:
        assert isinstance(job.device, AZUREDevice)

    if job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert job.measure is not None
        job.status = JobStatus.RUNNING
        job_sim = backend_sim.run(qiskit_circuit, shots=job.measure.shots)
        result_sim = job_sim.result()
        result = extract_result(result_sim, job, job.device)
    else:
        raise ValueError(f"Job type {job.job_type} not handled on Azure devices.")

    job.status = JobStatus.DONE
    return result


@typechecked
def extract_result(
    result: "MicrosoftEstimatorResult | QiskitResult",
    job: Optional[Job],
    device: AZUREDevice,
) -> Result:
    """Extract the result from Azure or Qiskit result objects and convert it into our format.

    Args:
        result: The result object to extract data from.
        job: The associated job object (optional for Microsoft results).
        device: The device where the job was executed.

    Returns:
        The formatted result object containing the job results.

    Raises:
        ValueError: If the result type is unsupported.
    """
    from qiskit.result import Result as QiskitResult
    from azure.quantum.target.microsoft.result import MicrosoftEstimatorResult

    if isinstance(result, QiskitResult):
        from mpqp.execution.providers.ibm import extract_result as extract_result_ibm

        return extract_result_ibm(result, job, device)
    elif isinstance(
        result, MicrosoftEstimatorResult
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        if job is None:
            job = Job(JobType.OBSERVABLE, QCircuit(1), device)
        return Result(job, 0, result.data())
    else:
        raise ValueError(f"result type not supported: {type(result)}")


@typechecked
def get_result_from_azure_job_id(job_id: str) -> Result:
    """Retrieves from Azure remote platform and parse the result of the job_id
    given in parameter. If the job is still running, we wait (blocking) until it
    is ``DONE``.

    Args:
        job_id: Id of the remote Azure job.

    Returns:
        The result converted to our format.
    """
    job = get_jobs_by_id(job_id)
    result = job.get_results()
    nb_qubits = 0
    if job.details.metadata is not None:
        nb_qubits = int(job.details.metadata["num_qubits"])

    if "c" in result:
        result_list = result["c"]
        result_dict = {item: result_list.count(item) for item in set(result_list)}

        data = [
            Sample(
                bin_str=state,
                count=count,
                nb_qubits=nb_qubits,
            )
            for (state, count) in result_dict.items()
        ]
    elif "histogram" in result:
        result_dict = result["histogram"]
        isinstance(result_dict, dict)
        data = [
            Sample(
                index=int(state),
                count=int(count),
                probability=count,
                nb_qubits=nb_qubits,
            )
            for (state, count) in result_dict.items()
        ]
    else:
        raise ValueError(f"Result dictionary not compatible: {result}")

    shots = 0
    if job.details.input_params is not None:
        shots = job.details.input_params["shots"]
    device = AZUREDevice(job.details.target)

    job_ = Job(
        JobType.SAMPLE,
        QCircuit(nb_qubits),
        device,
        BasisMeasure(list(range(nb_qubits)), shots=shots),
    )
    return Result(job_, data, None, shots)


def extract_samples(job: Job, result: QiskitResult) -> list[Sample]:
    """Extracts the sample data from a Qiskit result object.

    Args:
        job: The job associated with the result.
        result: The Qiskit result object containing the counts data.

    Returns:
        A list of sample objects extracted from the Qiskit result.
    """
    job_data = result.data()
    return [
        Sample(
            bin_str="".join(map(str, state)),
            nb_qubits=job.circuit.nb_qubits,
            count=int(count),
        )
        for (state, count) in job_data.get("counts").items()
    ]
