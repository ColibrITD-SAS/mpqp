import random

import pytest
from typing import TYPE_CHECKING
import numpy as np

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.tools.circuit import random_circuit
from mpqp.tools.maths import matrix_eq, rand_orthogonal_matrix, is_unitary
from mpqp.execution import run, ATOSDevice, IBMDevice, AWSDevice, Result


def test_custom_gate_is_unitary():
    definition = UnitaryMatrix(np.array([[1, 0], [0, 1j]]))
    assert is_unitary(CustomGate(definition, [0]).to_matrix())


@pytest.mark.parametrize(
    "n_circ",
    [(random.randint(1, 5)) for _ in range(10)],
)
def test_random_orthogonal_matrix(n_circ: int):
    # TODO: test CIRQ when Qasm2 parsing working
    """
    Args:
        n_circ: size of the whole circuit
    """
    # size of the custom gate
    n_gate = random.randint(1, n_circ)
    # target index from which the custom start to be applied
    t = random.randint(0, n_circ - n_gate)
    # orthogonal matrix (which is unitary)
    m = rand_orthogonal_matrix(2**n_gate)
    c = QCircuit(
        [CustomGate(UnitaryMatrix(m), [(t + i) for i in range(n_gate)])],
        nb_qubits=n_circ,
    )
    # building the expected state vector
    exp_state_vector = m[:, 0]
    for _ in range(0, t):
        exp_state_vector = np.kron(np.array([1, 0]), exp_state_vector)
    for _ in range(t + n_gate, n_circ):
        exp_state_vector = np.kron(exp_state_vector, np.array([1, 0]))

    execution_ibm = run(c, IBMDevice.AER_SIMULATOR)
    execution_aws = run(c, AWSDevice.BRAKET_LOCAL_SIMULATOR)
    execution_qlm = run(c, ATOSDevice.MYQLM_PYLINALG)

    if TYPE_CHECKING:
        assert isinstance(execution_ibm, Result)
        assert isinstance(execution_aws, Result)
        assert isinstance(execution_qlm, Result)

    assert matrix_eq(execution_ibm.amplitudes, exp_state_vector, 1e-05, 1e-05)
    assert matrix_eq(execution_aws.amplitudes, exp_state_vector, 1e-05, 1e-05)
    assert matrix_eq(execution_qlm.amplitudes, exp_state_vector, 1e-05, 1e-05)
    assert matrix_eq(execution_qlm.amplitudes, execution_ibm.amplitudes, 1e-05, 1e-05)
    assert matrix_eq(execution_aws.amplitudes, execution_ibm.amplitudes, 1e-05, 1e-05)


def test_custom_gate_with_native_gates():
    # TODO: test CIRQ when Qasm2 parsing working
    x = UnitaryMatrix(np.array([[0, 1], [1, 0]]))
    h = UnitaryMatrix(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    cnot = UnitaryMatrix(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    )
    z = UnitaryMatrix(np.array([[1, 0], [0, -1]]))

    c1 = QCircuit(
        [
            CustomGate(x, [0]),
            CustomGate(h, [1]),
            CustomGate(cnot, [1, 2]),
            CustomGate(z, [0]),
        ],
        nb_qubits=4,
    )
    c2 = QCircuit([X(0), H(1), CNOT(1, 2), Z(0)], nb_qubits=4)

    execution_ibm = run(c1, IBMDevice.AER_SIMULATOR)
    execution_aws = run(c1, AWSDevice.BRAKET_LOCAL_SIMULATOR)
    execution_qlm = run(c1, ATOSDevice.MYQLM_PYLINALG)

    expected_ibm = run(c2, IBMDevice.AER_SIMULATOR)
    expected_aws = run(c2, AWSDevice.BRAKET_LOCAL_SIMULATOR)
    expected_qlm = run(c2, ATOSDevice.MYQLM_PYLINALG)

    if TYPE_CHECKING:
        assert isinstance(execution_ibm, Result)
        assert isinstance(execution_aws, Result)
        assert isinstance(execution_qlm, Result)
        assert isinstance(expected_ibm, Result)
        assert isinstance(expected_aws, Result)
        assert isinstance(expected_qlm, Result)

    assert matrix_eq(
        execution_ibm.amplitudes,
        expected_qlm.amplitudes,
        1e-05,
        1e-05,
    )
    assert matrix_eq(
        execution_aws.amplitudes,
        expected_ibm.amplitudes,
        1e-05,
        1e-05,
    )
    assert matrix_eq(
        execution_qlm.amplitudes,
        expected_aws.amplitudes,
        1e-05,
        1e-05,
    )


@pytest.mark.parametrize(
    "qubits",
    [(random.randint(1, 5)) for _ in range(10)],
)
def test_custom_gate_with_random_circuit(qubits: int):
    # TODO: test CIRQ when Qasm2 parsing working
    random_circ = random_circuit(nb_qubits=qubits)
    matrix = random_circ.to_matrix()
    custom_gate_circ = QCircuit(
        [CustomGate(UnitaryMatrix(matrix), list(range(qubits)))]
    )

    execution_ibm = run(custom_gate_circ, IBMDevice.AER_SIMULATOR)
    execution_aws = run(custom_gate_circ, AWSDevice.BRAKET_LOCAL_SIMULATOR)
    execution_qlm = run(custom_gate_circ, ATOSDevice.MYQLM_PYLINALG)

    expected_ibm = run(random_circ, IBMDevice.AER_SIMULATOR)
    expected_aws = run(random_circ, AWSDevice.BRAKET_LOCAL_SIMULATOR)
    expected_qlm = run(random_circ, ATOSDevice.MYQLM_PYLINALG)

    if TYPE_CHECKING:
        assert isinstance(execution_ibm, Result)
        assert isinstance(execution_aws, Result)
        assert isinstance(execution_qlm, Result)
        assert isinstance(expected_ibm, Result)
        assert isinstance(expected_aws, Result)
        assert isinstance(expected_qlm, Result)

    assert matrix_eq(
        execution_ibm.amplitudes,
        expected_qlm.amplitudes,
        1e-05,
        1e-05,
    )
    assert matrix_eq(
        execution_aws.amplitudes,
        expected_ibm.amplitudes,
        1e-05,
        1e-05,
    )
    assert matrix_eq(
        execution_qlm.amplitudes,
        expected_aws.amplitudes,
        1e-05,
        1e-05,
    )
