# 3M-TODO
import pytest

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.execution import run, AWSDevice


@pytest.mark.parametrize(
    "circuit",
    [
        QCircuit([H(2)]),
        QCircuit([H(0), H(2)]),
        QCircuit([H(0), CNOT(1, 3), H(2)]),
        QCircuit([H(2)], nb_qubits=5),
    ]
)
def test_braket_non_contiguous_qubits(circuit: QCircuit):
    run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

