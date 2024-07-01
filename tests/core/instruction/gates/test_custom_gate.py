import pytest

import numpy as np

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.tools.maths import matrix_eq
from mpqp.execution import run, ATOSDevice


@pytest.mark.parametrize(
    "circuit, state_vector",
    [
        (
            QCircuit([CustomGate(UnitaryMatrix())]),
            np.array()
        )
    ],
)
def test_custom_gate_state_vector(circuit: QCircuit, state_vector: np.ndarray):
    assert matrix_eq(run(circuit, ATOSDevice.MYQLM_PYLINALG).state_vector.amplitudes, state_vector)

def