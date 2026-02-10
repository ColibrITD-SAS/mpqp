# 3M-TODO
import pytest

from mpqp import CNOT, AWSDevice, H, QCircuit, run


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
