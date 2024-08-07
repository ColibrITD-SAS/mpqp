from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from qiskit.result import Result as QiskitResult
    from azure.quantum.target.microsoft.result import MicrosoftEstimatorResult


from typeguard import typechecked

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.languages import Language
from mpqp.execution.connection.azure_connection import get_azure_provider
from mpqp.execution.devices import AZUREDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector


@typechecked
def run_azure(job: Job) -> Result:
    """Executes the job on the right AZURE Q device precised in the job in
    parameter.

    Args:
        job: Job to be executed.

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

    assert isinstance(job.device, AZUREDevice)

    # if job.job_type == JobType.STATE_VECTOR:
    #     qiskit_circuit.save_statevector() pyright: ignore[reportAttributeAccessIssue]
    #     job.status = JobStatus.RUNNING
    #     job_sim = backend_sim.run(qiskit_circuit, shots=0)
    #     result_sim = job_sim.result()
    #     result = extract_result(result_sim, job, job.device)
    if job.job_type == JobType.SAMPLE:
        assert job.measure is not None
        job.status = JobStatus.RUNNING
        job_sim = backend_sim.run(qiskit_circuit, shots=job.measure.shots)
        result_sim = job_sim.result()
        result = extract_result(result_sim, job, job.device)
    else:
        raise ValueError(f"Job type {job.job_type} not handled.")

    job.status = JobStatus.DONE
    return result


@typechecked
def extract_result(
    result: "MicrosoftEstimatorResult | QiskitResult",
    job: Optional[Job],
    device: AZUREDevice,
) -> Result:
    from qiskit.result import Result as QiskitResult

    if isinstance(result, QiskitResult):
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
            job_data = result.data()
            data = [
                Sample(
                    bin_str="".join(map(str, state)),
                    probability=(
                        job_data.get("probabilities").get(state)
                        if "probabilities" in job_data
                        else None
                    ),
                    nb_qubits=job.circuit.nb_qubits,
                    count=int(count),
                )
                for (state, count) in job_data.get("counts").items()
            ]
            return Result(job, data, None, job.measure.shots)
        else:
            raise NotImplementedError(f"{job.job_type} not handled.")
    else:
        raise NotImplementedError(f"{job} not handled.")
