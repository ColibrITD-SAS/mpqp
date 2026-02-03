from itertools import product
import random
from functools import reduce

import numpy as np
import pytest
from numpy import array  # pyright: ignore[reportUnusedImport]
from sympy import symbols

from mpqp import ATOSDevice, AWSDevice, GOOGLEDevice, IBMDevice, QCircuit, Result, run
from mpqp.core.instruction.gates.gate import SingleQubitGate
from mpqp.execution import AvailableDevice
from mpqp.gates import *
from mpqp.tools.circuit import random_circuit
from mpqp.tools.maths import is_unitary, matrix_eq, rand_orthogonal_matrix


def test_custom_gate_is_unitary():
    definition = np.array([[1, 0], [0, 1j]])
    assert is_unitary(CustomGate(definition, [0]).to_matrix())


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("circ_size", range(1, 6))
def test_random_orthogonal_matrix_qiskit(circ_size: int):
    exec_random_orthogonal_matrix(circ_size, IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("circ_size", range(1, 6))
def test_random_orthogonal_matrix_braket(circ_size: int):
    exec_random_orthogonal_matrix(circ_size, AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("circ_size", range(1, 6))
def test_random_orthogonal_matrix_cirq(circ_size: int):
    exec_random_orthogonal_matrix(circ_size, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("circ_size", range(1, 6))
def test_random_orthogonal_matrix_myqlm(circ_size: int):
    exec_random_orthogonal_matrix(circ_size, ATOSDevice.MYQLM_PYLINALG)


def exec_random_orthogonal_matrix(circ_size: int, device: AvailableDevice):
    gate_size = random.randint(1, circ_size)
    targets_start = random.randint(0, circ_size - gate_size)
    m = rand_orthogonal_matrix(2**gate_size)
    c = QCircuit(
        [CustomGate(m, list(range(targets_start, targets_start + gate_size)))],
        nb_qubits=circ_size,
    )
    # building the expected state vector
    exp_state_vector = m[:, 0]
    for _ in range(0, targets_start):
        exp_state_vector = np.kron(np.array([1, 0]), exp_state_vector)
    for _ in range(targets_start + gate_size, circ_size):
        exp_state_vector = np.kron(exp_state_vector, np.array([1, 0]))

    result = run(c, device)

    # we reduce the precision because of approximation errors coming from CustomGate usage
    assert isinstance(result, Result)
    assert matrix_eq(result.amplitudes, exp_state_vector, 1e-5, 1e-5)


@pytest.mark.provider("qiskit")
def test_custom_gate_with_native_gates_qiskit():
    exec_custom_gate_with_native_gates(IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("braket")
def test_custom_gate_with_native_gates_braket():
    exec_custom_gate_with_native_gates(AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("cirq")
def test_custom_gate_with_native_gates_cirq():
    exec_custom_gate_with_native_gates(GOOGLEDevice.CIRQ_LOCAL_SIMULATOR)


@pytest.mark.provider("myqlm")
def test_custom_gate_with_native_gates_myqlm():
    exec_custom_gate_with_native_gates(ATOSDevice.MYQLM_PYLINALG)


def exec_custom_gate_with_native_gates(device: AvailableDevice):
    x = np.array([[0, 1], [1, 0]])
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    z = np.array([[1, 0], [0, -1]])

    c1 = QCircuit(
        [
            CustomGate(x, [0]),
            CustomGate(h, [1]),
            CustomGate(cnot, [1, 2]),
            CustomGate(z, [0]),
        ]
    )
    c2 = QCircuit([X(0), H(1), CNOT(1, 2), Z(0)])

    result1 = run(c1, device)

    result2 = run(c2, device)

    # we reduce the precision because of approximation errors coming from CustomGate usage
    assert isinstance(result1, Result)
    assert isinstance(result2, Result)
    assert matrix_eq(result1.amplitudes, result2.amplitudes, 1e-4, 1e-4)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("circ_size", range(1, 6))
def test_custom_gate_with_random_circuit_qiskit(circ_size: int):
    exec_custom_gate_with_random_circuit(circ_size, IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("circ_size", range(1, 6))
def test_custom_gate_with_random_circuit_braket(circ_size: int):
    exec_custom_gate_with_random_circuit(circ_size, AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("circ_size", range(1, 6))
def test_custom_gate_with_random_circuit_cirq(circ_size: int):
    exec_custom_gate_with_random_circuit(circ_size, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("circ_size", range(1, 6))
def test_custom_gate_with_random_circuit_myqlm(circ_size: int):
    exec_custom_gate_with_random_circuit(circ_size, ATOSDevice.MYQLM_PYLINALG)


def exec_custom_gate_with_random_circuit(circ_size: int, device: AvailableDevice):
    random_circ = random_circuit(nb_qubits=circ_size)
    matrix = random_circ.to_matrix()
    custom_gate_circ = QCircuit([CustomGate(matrix, list(range(circ_size)))])

    result1 = run(random_circ, device)
    result2 = run(custom_gate_circ, device)

    assert isinstance(result1, Result)
    assert isinstance(result2, Result)
    # precision reduced from approximation errors (CustomGate usage)
    assert matrix_eq(result1.amplitudes, result2.amplitudes, 1e-4, 1e-4)


def _test_matrix_equality(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    circuit = QCircuit([gate(position) for gate, position in gates_n_positions])
    matrix = reduce(
        np.kron,
        [gate(0).to_matrix().astype(np.complex64) for gate, _ in gates_n_positions],
    )
    targets = [position for _, position in gates_n_positions]
    assert matrix_eq(
        QCircuit([CustomGate(matrix, targets)]).to_matrix(),
        circuit.to_matrix(),
    )


def non_contiguous_targets():
    return [[(H, 0), (X, 1), (Z, 3)]]


def non_ordered_targets():
    return [
        [(X, 1), (Y, 0)],
        [(H, 2), (X, 1), (Y, 0)],
        [(H, 0), (X, 2), (Y, 1)],
        [(H, 2), (X, 0), (Y, 1)],
    ]


def non_contiguous_ordered_targets():
    return [[(H, 0), (Y, 2)]]


@pytest.mark.parametrize("gates_n_positions", non_contiguous_targets())
def test_non_continuous_targets(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    _test_matrix_equality(gates_n_positions)


@pytest.mark.parametrize("gates_n_positions", non_ordered_targets())
def test_non_ordered_targets(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    _test_matrix_equality(gates_n_positions)


@pytest.mark.parametrize("gates_n_positions", non_contiguous_ordered_targets())
def test_non_contiguous_ordered_targets(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    _test_matrix_equality(gates_n_positions)


def _test_execution_equivalence(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]], device: AvailableDevice
):
    circuit = QCircuit([gate(position) for gate, position in gates_n_positions])
    matrix = reduce(
        np.kron,
        [gate(0).to_matrix().astype(np.complex64) for gate, _ in gates_n_positions],
    )
    targets = [position for _, position in gates_n_positions]

    result_custom_gate = run(QCircuit([CustomGate(matrix, targets)]), device)
    result_circuit = run(circuit, device)
    assert matrix_eq(
        result_custom_gate.amplitudes, result_circuit.amplitudes, 1e-4, 1e-4
    )


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("gates_n_positions", non_ordered_targets())
def test_non_ordered_targets_execution_qiskit(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_ordered_targets_execution(gates_n_positions, IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("gates_n_positions", non_ordered_targets())
def test_non_ordered_targets_execution_braket(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_ordered_targets_execution(
        gates_n_positions, AWSDevice.BRAKET_LOCAL_SIMULATOR
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("gates_n_positions", non_ordered_targets())
def test_non_ordered_targets_execution_cirq(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_ordered_targets_execution(
        gates_n_positions, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("gates_n_positions", non_ordered_targets())
def test_non_ordered_targets_execution_myqlm(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_ordered_targets_execution(gates_n_positions, ATOSDevice.MYQLM_PYLINALG)


def exec_non_ordered_targets_execution(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]], device: AvailableDevice
):
    _test_execution_equivalence(gates_n_positions, device)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("gates_n_positions", non_contiguous_targets())
def test_non_contiguous_targets_execution_qiskit(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_contiguous_targets_execution(gates_n_positions, IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("gates_n_positions", non_contiguous_targets())
def test_non_contiguous_targets_execution_braket(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_contiguous_targets_execution(
        gates_n_positions, AWSDevice.BRAKET_LOCAL_SIMULATOR
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("gates_n_positions", non_contiguous_targets())
def test_non_contiguous_targets_execution_cirq(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_contiguous_targets_execution(
        gates_n_positions, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("gates_n_positions", non_contiguous_targets())
def test_non_contiguous_targets_execution_myqlm(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_contiguous_targets_execution(gates_n_positions, ATOSDevice.MYQLM_PYLINALG)


def exec_non_contiguous_targets_execution(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]], device: AvailableDevice
):
    _test_execution_equivalence(gates_n_positions, device)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("gates_n_positions", non_contiguous_ordered_targets())
def test_non_contiguous_ordered_targets_execution_qiskit(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_contiguous_ordered_targets_execution(
        gates_n_positions, IBMDevice.AER_SIMULATOR
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize("gates_n_positions", non_contiguous_ordered_targets())
def test_non_contiguous_ordered_targets_execution_braket(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_contiguous_ordered_targets_execution(
        gates_n_positions, AWSDevice.BRAKET_LOCAL_SIMULATOR
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("gates_n_positions", non_contiguous_ordered_targets())
def test_non_contiguous_ordered_targets_execution_cirq(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_contiguous_ordered_targets_execution(
        gates_n_positions, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("gates_n_positions", non_contiguous_ordered_targets())
def test_non_contiguous_ordered_targets_execution_myqlm(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    exec_non_contiguous_ordered_targets_execution(
        gates_n_positions, ATOSDevice.MYQLM_PYLINALG
    )


def exec_non_contiguous_ordered_targets_execution(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]], device: AvailableDevice
):
    _test_execution_equivalence(gates_n_positions, device)


def test_subs():
    theta = symbols("theta")
    g = CustomGate(np.array([[theta, 0], [0, 1]]), [0])
    assert g.matrix[0, 0] == theta
    g2 = g.subs({"theta": 1})  # pyright: ignore
    assert g2.matrix[0, 0] == 1


def test_symbolic_str():
    theta = symbols("theta")

    correct_result = """\
   ┌───────────────────┐
q: ┤ CustomGate(theta) ├
   └───────────────────┘"""

    assert str(CustomGate(np.array([[theta, 0], [0, 1]]), [0])) == correct_result
