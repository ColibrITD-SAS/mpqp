import pytest

from mpqp import CNOT, AWSDevice, BasisMeasure, H, QCircuit, run
from mpqp.execution.providers.aws import estimate_cost_single_job
from mpqp.execution.runner import generate_job
from mpqp.tools.errors import DeviceJobIncompatibleError


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circuit",
    [
        QCircuit([H(2)]),
        QCircuit([H(0), H(2)]),
        QCircuit([H(0), CNOT(1, 3), H(2)]),
        QCircuit([H(2)], nb_qubits=5),
    ],
)
def test_braket_non_contiguous_qubits(circuit: QCircuit):
    run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.parametrize(
    ("device", "expected_cost"),
    [
        (AWSDevice.AQT_IBEX_Q1, 2.65),
        (AWSDevice.IONQ_FORTE_1, 8.3),
        (AWSDevice.IONQ_FORTE_ENTERPRISE_1, 8.3),
        (AWSDevice.IQM_GARNET, 0.445),
        (AWSDevice.IQM_EMERALD, 0.46),
        (AWSDevice.RIGETTI_ANKAA_3, 0.39),
        (AWSDevice.RIGETTI_CEPHEUS_1_108Q, 0.3425),
    ],
)
def test_estimate_cost_for_supported_qpus(device: AWSDevice, expected_cost: float):
    circuit = QCircuit([H(0), BasisMeasure(shots=100)])
    job = generate_job(circuit, device)

    assert estimate_cost_single_job(job) == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("device", "error_message"),
    [
        (AWSDevice.IONQ_ARIA_1, "has been retired by Amazon Braket"),
        (AWSDevice.QUERA_AQUILA, "does not support MPQP QCircuit jobs"),
    ],
)
def test_estimate_cost_rejects_unsupported_devices(
    device: AWSDevice, error_message: str
):
    circuit = QCircuit([H(0), BasisMeasure(shots=100)])
    job = generate_job(circuit, device)

    with pytest.raises(DeviceJobIncompatibleError, match=error_message):
        estimate_cost_single_job(job)
