import contextlib
import random
from functools import reduce
from itertools import product

import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.gates.gate import SingleQubitGate
from mpqp.execution import (
    ATOSDevice,
    AvailableDevice,
    AWSDevice,
    GOOGLEDevice,
    IBMDevice,
    Result,
)
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.gates import *
from mpqp.tools.circuit import random_circuit
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning
from mpqp.tools.maths import is_unitary, matrix_eq, rand_orthogonal_matrix


def test_custom_gate_is_unitary():
    definition = UnitaryMatrix(np.array([[1, 0], [0, 1j]]))
    assert is_unitary(CustomGate(definition, [0]).to_matrix())


@pytest.mark.parametrize(
    "circ_size, device",
    product(
        range(1, 6),
        [
            IBMDevice.AER_SIMULATOR,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            ATOSDevice.MYQLM_PYLINALG,
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        ],
    ),
)
def test_random_orthogonal_matrix(circ_size: int, device: AvailableDevice):
    gate_size = random.randint(1, circ_size)
    targets_start = random.randint(0, circ_size - gate_size)
    m = UnitaryMatrix(rand_orthogonal_matrix(2**gate_size))
    c = QCircuit(
        [CustomGate(m, list(range(targets_start, targets_start + gate_size)))],
        nb_qubits=circ_size,
    )
    # building the expected state vector
    exp_state_vector = m.matrix[:, 0]
    for _ in range(0, targets_start):
        exp_state_vector = np.kron(np.array([1, 0]), exp_state_vector)
    for _ in range(targets_start + gate_size, circ_size):
        exp_state_vector = np.kron(exp_state_vector, np.array([1, 0]))

    with (
        pytest.warns(UnsupportedBraketFeaturesWarning)
        if isinstance(device, AWSDevice)
        else contextlib.suppress()
    ):
        result = _run_single(c, device, {})

    # we reduce the precision because of approximation errors coming from CustomGate usage
    assert isinstance(result, Result)
    assert matrix_eq(result.amplitudes, exp_state_vector, 1e-5, 1e-5)


@pytest.mark.parametrize(
    "device",
    [
        IBMDevice.AER_SIMULATOR,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ATOSDevice.MYQLM_PYLINALG,
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
    ],
)
def test_custom_gate_with_native_gates(device: AvailableDevice):
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
        ]
    )
    c2 = QCircuit([X(0), H(1), CNOT(1, 2), Z(0)])

    with (
        pytest.warns(UnsupportedBraketFeaturesWarning)
        if isinstance(device, AWSDevice)
        else contextlib.suppress()
    ):
        result1 = _run_single(c1, device, {})

    with (
        pytest.warns(UnsupportedBraketFeaturesWarning)
        if isinstance(device, AWSDevice)
        else contextlib.suppress()
    ):
        result2 = _run_single(c2, device, {})

    # we reduce the precision because of approximation errors coming from CustomGate usage
    assert isinstance(result1, Result)
    assert isinstance(result2, Result)
    assert matrix_eq(result1.amplitudes, result2.amplitudes, 1e-4, 1e-4)


@pytest.mark.parametrize(
    "circ_size, device",
    product(
        range(1, 6),
        [
            IBMDevice.AER_SIMULATOR,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            ATOSDevice.MYQLM_PYLINALG,
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        ],
    ),
)
def test_custom_gate_with_random_circuit(circ_size: int, device: AvailableDevice):
    random_circ = random_circuit(nb_qubits=circ_size)
    matrix = random_circ.to_matrix()
    custom_gate_circ = QCircuit(
        [CustomGate(UnitaryMatrix(matrix), list(range(circ_size)))]
    )

    with (
        pytest.warns(UnsupportedBraketFeaturesWarning)
        if isinstance(device, AWSDevice)
        else contextlib.suppress()
    ):
        result1 = _run_single(random_circ, device, {})
        result2 = _run_single(custom_gate_circ, device, {})

    assert isinstance(result1, Result)
    assert isinstance(result2, Result)
    # precision reduced from approximation errors (CustomGate usage)
    assert matrix_eq(result1.amplitudes, result2.amplitudes, 1e-4, 1e-4)


@pytest.mark.parametrize(
    "gates_n_positions",
    [
        ([(H, 0), (X, 1), (Z, 3)]),
    ],
)
def test_non_continuous_targets(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    circuit = QCircuit([gate(position) for gate, position in gates_n_positions])
    matrix = reduce(
        np.kron,
        [gate(0).to_matrix().astype(np.complex64) for gate, _ in gates_n_positions],
    )
    targets = [position for _, position in gates_n_positions]
    assert matrix_eq(
        QCircuit([CustomGate(UnitaryMatrix(matrix), targets)]).to_matrix(),
        circuit.to_matrix(),
    )


@pytest.mark.parametrize(
    "gates_n_positions",
    [
        ([(X, 1), (Y, 0)]),
        ([(H, 2), (X, 1), (Y, 0)]),
        ([(H, 0), (X, 2), (Y, 1)]),
        ([(H, 2), (X, 0), (Y, 1)]),
    ],
)
def test_non_ordered_targets(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]],
):
    circuit = QCircuit([gate(position) for gate, position in gates_n_positions])
    matrix = reduce(
        np.kron,
        [gate(0).to_matrix().astype(np.complex64) for gate, _ in gates_n_positions],
    )
    targets = [position for _, position in gates_n_positions]
    assert matrix_eq(
        QCircuit([CustomGate(UnitaryMatrix(matrix), targets)]).to_matrix(),
        circuit.to_matrix(),
    )


@pytest.mark.parametrize(
    "gates_n_positions, device",
    product(
        [
            ([(X, 1), (Y, 0)]),
            ([(H, 2), (X, 1), (Y, 0)]),
            ([(H, 0), (X, 2), (Y, 1)]),
            ([(H, 2), (X, 0), (Y, 1)]),
        ],
        [
            IBMDevice.AER_SIMULATOR,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            ATOSDevice.MYQLM_PYLINALG,
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        ],
    ),
)
def test_non_ordered_targets_execution(
    gates_n_positions: list[tuple[type[SingleQubitGate], int]], device: AvailableDevice
):
    circuit = QCircuit([gate(position) for gate, position in gates_n_positions])
    matrix = reduce(
        np.kron,
        [gate(0).to_matrix().astype(np.complex64) for gate, _ in gates_n_positions],
    )
    targets = [position for _, position in gates_n_positions]

    result_custom_gate = _run_single(
        QCircuit([CustomGate(UnitaryMatrix(matrix), targets)]), device, {}
    )
    result_circuit = _run_single(circuit, device, {})
    assert matrix_eq(
        result_custom_gate.amplitudes, result_circuit.amplitudes, 1e-4, 1e-4
    )
