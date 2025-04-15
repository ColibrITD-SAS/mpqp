from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from qiskit.result import Result as QiskitResult
    from azure.quantum.qiskit.job import AzureQuantumJob

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
def run_azure(job: Job, warnings: bool = True) -> Result:
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
    _, job_sim = submit_job_azure(job, warnings)
    result_sim = job_sim.result()
    if TYPE_CHECKING:
        assert isinstance(job.device, AZUREDevice)
    return extract_result(result_sim, job, job.device)


@typechecked
def submit_job_azure(
    job: Job, translation_warning: bool = True
) -> tuple[str, "AzureQuantumJob"]:
    """Submits the job on the remote Azure device (quantum computer or simulator).

    Args:
        job: Job to be executed.

    Returns:
        Azure's job id and the job itself.
        translation_warning: If `True`, a warning will be raised.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    from qiskit import QuantumCircuit

    if job.circuit.transpiled_circuit is None:
        qiskit_circuit = (
            (
                # 3M-TODO: careful, if we ever support several measurements, the
                # line bellow will have to changer
                job.circuit.without_measurements()
                + job.circuit.pre_measure()
            ).to_other_language(
                Language.QISKIT, translation_warning=translation_warning
            )
            if (job.job_type == JobType.STATE_VECTOR)
            else job.circuit.to_other_language(
                Language.QISKIT, translation_warning=translation_warning
            )
        )
    else:
        qiskit_circuit = job.circuit.transpiled_circuit
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
    else:
        raise ValueError(f"Job type {job.job_type} not handled on Azure devices.")

    return job_sim.id(), job_sim


@typechecked
def extract_result(
    result: "QiskitResult",
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

    from mpqp.execution.providers.ibm import extract_result as extract_result_ibm

    return extract_result_ibm(result, job, device)


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
    elif isinstance(result, dict):
        data = [
            Sample(
                index=int("".join(map(str, eval(state))), 2),
                count=int(count),
                probability=count,
                nb_qubits=nb_qubits,
            )
            for (state, count) in result.items()
        ]
    else:
        raise ValueError(f"Result dictionary not compatible: {type(result)}\n{result}")

    shots = 0
    if job.details.input_params is not None:
        shots = job.details.input_params["shots"]
    device = AZUREDevice(job.details.target)

    job_ = Job(
        JobType.SAMPLE,
        QCircuit(
            [BasisMeasure(list(range(nb_qubits)), shots=shots)], nb_qubits=nb_qubits
        ),
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
