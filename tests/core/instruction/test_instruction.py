from mpqp.core.instruction.gates.native_gates import *
from mpqp.tools.circuit import random_gate
from mpqp.tools.maths import matrix_eq
import numpy as np
import pytest


@pytest.mark.parametrize("gate", NATIVE_GATES)
def test_inverse_gate(gate: type):
    gate_build = random_gate([gate])
    gate_build_matrix = gate_build.to_matrix()
    gate_build_dagger = gate_build.inverse()
    gate_build_dagger_dagger = gate_build_dagger.inverse()
    # G†† = G
    assert matrix_eq(gate_build_dagger_dagger.to_matrix(), gate_build_matrix)
    # G†G = I
    assert matrix_eq(
        np.matmul(gate_build_dagger.to_matrix(), gate_build_matrix),
        np.eye(gate_build_matrix.shape[0], dtype=np.complex64),
    )
