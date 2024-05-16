import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.gates import *
from mpqp.noise import Depolarizing
from mpqp.execution import *


def test_all_native_gates_global_noise():
    circuit = QCircuit(3, label="Test noisy native gates")
    circuit.add([H(0), X(1), Y(2), Z(0), S(1), T(0)])
    circuit.add([Rx(1.2324, 2), Ry(-2.43, 0), Rz(1.04, 1), Rk(-1, 0), P(-323, 2)])
    circuit.add(U(1.2, 2.3, 3.4, 2))
    circuit.add(SWAP(2, 1))
    circuit.add([CNOT(0, 1), CRk(4, 2, 0), CZ(1, 2)])
    circuit.add(BasisMeasure([0, 1, 2], shots=1023))
    circuit.add(Depolarizing(0.23, [0, 1,]))
    #circuit.add(Depolarizing(0.23, [0, 1, 2], dimension=2, gates=[SWAP, CNOT, CZ, CRk]))
    run(
        circuit,
        [
            # ATOSDevice.QLM_MPO,
            # ATOSDevice.QLM_NOISYQPROC,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ],
    )
    assert True


def test_all_native_gates_local_noise():
    circuit = QCircuit(3, label="Test noisy native gates")
    circuit.add([H(0), X(1), Y(2), Z(0), S(1), T(0)])
    circuit.add([Rx(1.2324, 2), Ry(-2.43, 0), Rz(1.04, 1), Rk(-1, 0), P(-323, 2)])
    circuit.add(U(1.2, 2.3, 3.4, 2))
    circuit.add(SWAP(2, 1))
    circuit.add([CNOT(0, 1), CRk(4, 2, 0), CZ(1, 2)])
    circuit.add(BasisMeasure([0, 1, 2], shots=1023))
    circuit.add(Depolarizing(0.23, [0, 2], gates=[H, X, Y, Z, S, T, Rx, Ry, Rz, Rk, P, U]))
    circuit.add(Depolarizing(0.23, [0, 1], dimension=2, gates=[SWAP, CNOT, CZ, CRk]))
    run(
        circuit,
        [
            # ATOSDevice.QLM_MPO,
            # ATOSDevice.QLM_NOISYQPROC,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ],
    )
    assert True