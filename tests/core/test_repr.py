from copy import deepcopy
import itertools
import sys
from numpy import array, complex64  # pyright: ignore[reportUnusedImport]
import pytest
from sympy import Expr, cos  # pyright: ignore[reportUnusedImport]

from mpqp import QCircuit
from mpqp.measures import I as pauli_I, X as pauli_X, Y as pauli_Y, Z as pauli_Z
from mpqp.tools.circuit import random_gate, random_noise
from mpqp.all import *


def generate_qcircuits():
    qcircuits = []
    for args in itertools.product(
        [None, 5],  # nb_qubits
        [None, 5],  # nb_cbits
        [None, "test"],  # label
        [[], [random_gate()], [random_gate(), random_gate()]],  # gates
        [[], [random_noise()], [random_noise(), random_noise()]],  # noises
        [[], [BasisMeasure()]],
    ):
        nb_qubits, nb_cbits, label, gates, noises, measures = args
        kwargs = {}
        if nb_qubits is not None:
            kwargs["nb_qubits"] = nb_qubits
        if nb_cbits is not None:
            kwargs["nb_cbits"] = nb_cbits
        if label is not None:
            kwargs["label"] = label

        qcircuits.append(
            QCircuit(deepcopy(gates) + deepcopy(measures) + deepcopy(noises), **kwargs)
        )
    return qcircuits


@pytest.mark.parametrize("qcircuit", generate_qcircuits())
def repr_qcircuits(qcircuit: QCircuit):
    assert eval(repr(qcircuit)) == qcircuit


def test_repr_qcircuits_random():
    for _ in range(20):
        qcircuit = random_circuit()
        assert eval(repr(qcircuit)) == qcircuit


def generate_basis_measures():
    measures = []
    for args in itertools.product(
        [None, [0, 1]],  # target
        [None, [0, 1]],  # c_targets
        [None, 1, 1024],  # shots
        [
            None,
            HadamardBasis(),
            HadamardBasis(2),
            ComputationalBasis(),
            ComputationalBasis(2),
            VariableSizeBasis([np.array([1, 0]), np.array([0, -1])]),
            VariableSizeBasis(
                [np.array([1, 0]), np.array([0, -1])], symbols=("0", "1")
            ),
            VariableSizeBasis(
                [np.array([1, 0]), np.array([0, -1])], symbols=("↑", "↓")
            ),
            Basis(
                [
                    np.array([1, 0, 0, 0]),
                    np.array([0, -1, 0, 0]),
                    np.array([0, 0, 1, 0]),
                    np.array([0, 0, 0, -1]),
                ]
            ),
            Basis(
                [
                    np.array([1, 0, 0, 0]),
                    np.array([0, -1, 0, 0]),
                    np.array([0, 0, 1, 0]),
                    np.array([0, 0, 0, -1]),
                ],
                symbols=("0", "1"),
            ),
            Basis(
                [
                    np.array([1, 0, 0, 0]),
                    np.array([0, -1, 0, 0]),
                    np.array([0, 0, 1, 0]),
                    np.array([0, 0, 0, -1]),
                ],
                symbols=("↑", "↓"),
            ),
            Basis(
                [
                    np.array([1, 0, 0, 0]),
                    np.array([0, -1, 0, 0]),
                    np.array([0, 0, 1, 0]),
                    np.array([0, 0, 0, -1]),
                ],
                basis_vectors_labels=["↑", "↓"],
            ),
        ],  # basis
        [None, "test"],  # label
    ):
        targets, c_targets, shots, basis, label = args
        kwargs = {}
        if targets is not None:
            kwargs["targets"] = targets
            if c_targets is not None:
                kwargs["c_targets"] = c_targets
        if shots is not None:
            kwargs["shots"] = shots
        if label is not None:
            kwargs["label"] = label
        if basis is not None:
            kwargs["basis"] = deepcopy(basis)

        measures.append(BasisMeasure(**kwargs))
    return measures


@pytest.mark.parametrize("measure", generate_basis_measures())
def repr_basis_measure(measure: Measure):
    assert eval(repr(measure)) == measure


def generate_expectation_measures():
    measures = []
    x = symbols("x")
    number = x.subs(x, 4)
    for args in itertools.product(
        [None, [0, 1]],  # target
        [None, 1, 0],  # shots
        [
            Observable(np.diag([0.7, -1, 1, 1])),
            Observable(pauli_I @ pauli_X),
            Observable(pauli_I @ pauli_X + pauli_Y @ pauli_Z),
            Observable(number * pauli_I @ pauli_X),
            Observable(-5.5 * pauli_I @ pauli_X + -6 * pauli_Y @ pauli_Z),
        ],  # observables
        [None, "test"],  # label
    ):
        targets, shots, observables, label = args
        kwargs = {}
        if targets is not None:
            kwargs["targets"] = targets
        if shots is not None:
            kwargs["shots"] = shots
        if label is not None:
            kwargs["label"] = label

        measures.append(ExpectationMeasure(deepcopy(observables), **kwargs))
    return measures


@pytest.mark.parametrize("measure", generate_expectation_measures())
def repr_expectation_measures(measure: Measure):
    measure_repr = (
        repr(measure)
        .replace("I", "pauli_I")
        .replace("X", "pauli_X")
        .replace("Y", "pauli_Y")
        .replace("Z", "pauli_Z")
    )
    print(measure_repr)
    assert eval(measure_repr) == measure


def repr_barriers():
    for size in [0, 1, 2]:
        barrier = Barrier(size)
        assert eval(repr(barrier)) == barrier
    barrier = Barrier()
    assert eval(repr(barrier)) == barrier


def repr_breakpoints():
    for args in itertools.product(
        [None, False, True],  # draw_circuit
        [None, False, True],  # enabled
        [None, "test"],  # label
    ):
        draw_circuit, enabled, label = args
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        if draw_circuit is not None:
            kwargs["draw_circuit"] = draw_circuit
        if enabled is not None:
            kwargs["enabled"] = enabled

        breakpoint = Breakpoint(**kwargs)
        print(repr(breakpoint))
        print(kwargs)
        if eval(repr(breakpoint)) != breakpoint:
            print(repr(breakpoint))
            print(kwargs)
        assert eval(repr(breakpoint)) == breakpoint


def repr_Depolarizing_noise():
    for args in itertools.product(
        [0, 0.5],  # prob
        [None, [0, 1]],  # targets
        [None, 1],  # dimension
        [None, [H]],  # gates
    ):
        prob, targets, dimension, gates = args
        kwargs = {}
        kwargs["prob"] = prob
        if targets is not None:
            kwargs["targets"] = targets
        if dimension is not None:
            kwargs["dimension"] = dimension
        if gates is not None:
            kwargs["gates"] = gates

        noise = Depolarizing(**kwargs)
        assert eval(repr(noise)) == noise


def repr_BitFlip_noise():
    for args in itertools.product(
        [0, 0.5],  # prob
        [None, [0, 1]],  # targets
        [None, [H]],  # gates
    ):
        prob, targets, gates = args
        kwargs = {}
        kwargs["prob"] = prob
        if targets is not None:
            kwargs["targets"] = targets
        if gates is not None:
            kwargs["gates"] = gates

        noise = BitFlip(**kwargs)
        assert eval(repr(noise)) == noise


def repr_AmplitudeDamping_noise():
    for args in itertools.product(
        [None, 0.5],  # prob
        [None, [0, 1]],  # targets
        [1, 0.2],  # gamma
        [None, [H]],  # gates
    ):
        prob, targets, gamma, gates = args
        kwargs = {}
        kwargs["gamma"] = gamma
        if prob is not None:
            kwargs["prob"] = prob
        if targets is not None:
            kwargs["targets"] = targets
        if gates is not None:
            kwargs["gates"] = gates

        noise = AmplitudeDamping(**kwargs)
        assert eval(repr(noise)) == noise


def repr_PhaseDamping_noise():
    for args in itertools.product(
        [None, [0, 1]],  # targets
        [1, 0.2],  # gamma
        [None, [H]],  # gates
    ):
        targets, gamma, gates = args
        kwargs = {}
        kwargs["gamma"] = gamma
        if targets is not None:
            kwargs["targets"] = targets
        if gates is not None:
            kwargs["gates"] = gates

        noise = PhaseDamping(**kwargs)
        assert eval(repr(noise)) == noise


if "--long-local" in sys.argv or "--long" in sys.argv:
    test_repr_qcircuits = repr_qcircuits
    test_repr_basis_measure = repr_basis_measure
    test_repr_expectation_measures = repr_expectation_measures
    test_repr_barriers = repr_barriers
    test_repr_breakpoints = repr_breakpoints
    test_repr_Depolarizing_noise = repr_Depolarizing_noise
    test_repr_BitFlip_noise = repr_BitFlip_noise
    test_repr_AmplitudeDamping_noise = repr_AmplitudeDamping_noise
    test_repr_PhaseDamping_noise = repr_PhaseDamping_noise
