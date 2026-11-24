from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from mpqp import BasisMeasure, GOOGLEDevice, QCircuit
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.providers.google import run_google_remote

if TYPE_CHECKING:
    from cirq_ionq import QPUResult, SimulatorResult


@pytest.fixture
def ionq_service(monkeypatch: pytest.MonkeyPatch) -> Mock:
    import cirq_ionq

    service = Mock(spec=cirq_ionq.Service)
    monkeypatch.setattr(cirq_ionq, "Service", Mock(return_value=service))
    monkeypatch.setattr(
        "mpqp.environment.env_manager.load_env_variables", Mock(return_value=True)
    )
    service.create_job.return_value.job_id.return_value = "ionq-job-id"
    return service


@pytest.fixture
def ionq_sample_job() -> Job:
    import cirq

    measure = BasisMeasure([0], shots=4)
    circuit = QCircuit([measure])
    circuit.transpiled_circuit = cirq.Circuit(cirq.I(cirq.NamedQubit("q_0")))
    return Job(JobType.SAMPLE, circuit, GOOGLEDevice.IONQ_SIMULATOR, measure)


def _ionq_result(qpu: bool) -> "QPUResult | SimulatorResult":
    import cirq_ionq

    if qpu:
        return cirq_ionq.QPUResult(
            counts={0: 3, 1: 1}, num_qubits=1, measurement_dict={"m": [0]}
        )
    return cirq_ionq.SimulatorResult(
        probabilities={0: 1.0},
        num_qubits=1,
        measurement_dict={"m": [0]},
        repetitions=4,
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("qpu", [False, True], ids=["simulator", "qpu"])
@pytest.mark.parametrize("as_list", [False, True], ids=["scalar", "singleton-list"])
def test_ionq_sample_accepts_result_shapes(
    ionq_service: Mock, ionq_sample_job: Job, qpu: bool, as_list: bool
):
    provider_result = _ionq_result(qpu)
    ionq_service.create_job.return_value.results.return_value = (
        [provider_result] if as_list else provider_result
    )
    if qpu:
        ionq_sample_job.device = GOOGLEDevice.IONQ_QPU

    result = run_google_remote(ionq_sample_job)

    assert result.counts == ([3, 1] if qpu else [4, 0])
    assert ionq_sample_job.id == "ionq-job-id"
    assert ionq_sample_job.status == JobStatus.DONE
    ionq_service.create_job.return_value.results.assert_called_once_with()


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("result_count", [0, 2], ids=["empty", "multiple"])
def test_ionq_sample_rejects_non_singleton_results(
    ionq_service: Mock, ionq_sample_job: Job, result_count: int
):
    ionq_service.create_job.return_value.results.return_value = [
        _ionq_result(False) for _ in range(result_count)
    ]

    with pytest.raises(NotImplementedError, match="multiple results"):
        run_google_remote(ionq_sample_job)

    assert ionq_sample_job.id == "ionq-job-id"
    assert ionq_sample_job.status == JobStatus.ERROR
