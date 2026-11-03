from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from cirq.ops.linear_combinations import PauliSum as CirqPauliSum
    from cirq.ops.pauli_string import PauliString as CirqPauliString

import numpy as np
import pytest

from mpqp import ExpectationMeasure, Language, Observable, QCircuit, X, pI, pX, pY, pZ
from mpqp.core.instruction.measurement.pauli_string import (
    CommutingTypes,
    GroupingMethods,
)
from mpqp.execution.devices import AWSDevice, IBMDevice
from mpqp.tools.errors import NumberQubitsError


@pytest.mark.parametrize("targets", [None, [0], [2]])
def test_expectation_measure_targets_stay_fixed(targets: Optional[list[int]]):
    measurement = ExpectationMeasure(Observable(pZ), targets)
    expected_targets = targets if targets is not None else [0]
    circuit = QCircuit([measurement])

    assert measurement.targets == expected_targets
    circuit.add(X(3))
    assert measurement.targets == expected_targets
    circuit.remove(X(3))
    assert measurement.targets == expected_targets
    assert circuit.nb_qubits == max(expected_targets) + 1


def test_expectation_measure_fits_larger_circuit():
    measurement = ExpectationMeasure(Observable(pZ))
    circuit = QCircuit([X(2), measurement])

    assert circuit.nb_qubits == 3
    assert measurement.targets == [0]


@pytest.mark.provider("qiskit")
def test_pre_transpile_observables_caches_translation_per_device():
    measurement = ExpectationMeasure(Observable(pZ))
    observable = measurement.observables[0]
    device = IBMDevice.AER_SIMULATOR

    assert observable.pre_transpiled == {}
    assert measurement.pre_transpile_observables(device) is None
    cached = observable.pre_transpiled[device]

    measurement.pre_transpile_observables(device)
    assert observable.pre_transpiled[device] is cached


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("representation", ["matrix", "pauli", "diagonal"])
def test_observable_mutation_clears_pre_transpiled_cache(representation: str):
    measurement = ExpectationMeasure(Observable(pZ))
    observable = measurement.observables[0]
    device = IBMDevice.AER_SIMULATOR
    measurement.pre_transpile_observables(device)

    if representation == "matrix":
        observable.matrix = np.array([[0, 1], [1, 0]])
    elif representation == "pauli":
        observable.pauli_string = pX
    else:
        observable.diagonal_elements = [1, 1]

    assert observable.pre_transpiled == {}


def test_observable_mutation_invalidates_cached_pauli_grouping():
    measurement = ExpectationMeasure(
        Observable(pX), grouping_method=GroupingMethods.GREEDY
    )
    first_grouping = measurement.get_pauli_grouping()

    measurement.observables[0].pauli_string = pZ
    second_grouping = measurement.get_pauli_grouping()

    assert second_grouping is not first_grouping
    assert second_grouping[0][0].short_name == "Z"


def test_observables_collection_cannot_be_mutated_through_property():
    measurement = ExpectationMeasure(Observable(pZ))
    exposed_observables = measurement.observables

    exposed_observables.append(Observable(pX))

    assert measurement.nb_observables == 1


def test_replacing_observables_clears_measurement_caches(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(QCircuit, "to_other_language", lambda *_args, **_kwargs: object())
    measurement = ExpectationMeasure(
        Observable(pZ), grouping_method=GroupingMethods.GREEDY
    )
    device = AWSDevice.BRAKET_LOCAL_SIMULATOR
    measurement.get_pauli_grouping()
    measurement.pre_transpile_observables(device)

    measurement.observables = [Observable(pX)]

    assert measurement.current_grouping is None
    assert measurement.translated_pre_measures == {}
    assert measurement.observables[0].pauli_string == pX


@pytest.mark.parametrize(
    "mutation", ["shots", "targets", "observable", "commuting_type"]
)
def test_pre_transpile_observables_invalidates_stale_aws_cache(
    monkeypatch: pytest.MonkeyPatch, mutation: str
):
    monkeypatch.setattr(QCircuit, "to_other_language", lambda *_args, **_kwargs: object())
    measurement = ExpectationMeasure(
        Observable(pX @ pZ),
        targets=[1, 3],
        shots=100,
        grouping_method=GroupingMethods.GREEDY,
    )
    device = AWSDevice.BRAKET_LOCAL_SIMULATOR

    assert measurement.pre_transpile_observables(device) is None
    cached = measurement.translated_pre_measures[device]
    measurement.pre_transpile_observables(device)
    assert measurement.translated_pre_measures[device] is cached

    if mutation == "shots":
        measurement.shots = 0
    elif mutation == "targets":
        measurement.targets = [0, 2]
    elif mutation == "observable":
        measurement.observables[0].pauli_string = pY @ pZ
    else:
        measurement.commuting_type = CommutingTypes.FULL

    measurement.pre_transpile_observables(device)
    assert measurement.translated_pre_measures[device] is not cached


@pytest.mark.provider("braket")
def test_pre_transpile_observables_caches_aws_measurement_preparation():
    measurement = ExpectationMeasure(Observable(pX @ pZ), targets=[1, 3], shots=100)
    device = AWSDevice.BRAKET_LOCAL_SIMULATOR

    assert measurement.pre_transpile_observables(device) is None
    cached = measurement.translated_pre_measures[device]

    measurement.pre_transpile_observables(device)
    assert measurement.translated_pre_measures[device] is cached


@pytest.mark.parametrize("targets", [[], [0, 1]])
def test_expectation_measure_rejects_wrong_target_count(targets: list[int]):
    with pytest.raises(
        NumberQubitsError,
        match=f"Target size {len(targets)} doesn't match observable size 1",
    ):
        ExpectationMeasure(Observable(pZ), targets)


@pytest.mark.parametrize(
    "targets",
    [[0, 1, 2], [1, 2, 3], list(range(4, 10))],
)
def test_expectation_measure_right_targets(targets: list[int]):
    obs = Observable(np.diag([1] * 2 ** len(targets)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExpectationMeasure(obs, targets)


# TODO: complete this
@pytest.fixture
def list_to_cirq_pauli() -> (
    list[tuple[Observable, Union[CirqPauliSum, CirqPauliString]]]
):
    from cirq.devices.line_qubit import LineQubit
    from cirq.ops.identity import I as Cirq_I
    from cirq.ops.pauli_gates import X as Cirq_X

    a, b = LineQubit.range(2)

    return [
        (
            Observable(pI @ pI + pI @ pX),
            sum(1.0 * Cirq_I(a) * Cirq_I(b) + Cirq_X(b)),
        ),
    ]


@pytest.mark.provider("cirq")
def test_to_other_language_cirq(
    list_to_cirq_pauli: list[tuple[Observable, Union[CirqPauliSum, CirqPauliString]]],
):
    for obs, translation in list_to_cirq_pauli:
        assert obs.to_other_language(Language.CIRQ) == translation


@pytest.mark.provider("quantinuum")
def test_to_other_language_tket():
    from pytket.circuit import Qubit
    from pytket.pauli import Pauli, QubitPauliString
    from pytket.utils.operators import QubitPauliOperator

    observable = Observable(2 * (pX @ pZ) - 0.5 * (pY @ pI))
    expected = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1)],
                [Pauli.X, Pauli.Z],
            ): 2,
            QubitPauliString(
                [Qubit(0), Qubit(1)],
                [Pauli.Y, Pauli.I],
            ): -0.5,
        }
    )

    assert observable.to_other_language(Language.TKET) == expected


@pytest.mark.provider("quantinuum")
def test_to_other_language_tket_explicit_targets():
    from pytket.circuit import Qubit
    from pytket.pauli import Pauli, QubitPauliString
    from pytket.utils.operators import QubitPauliOperator

    observable = Observable(pX @ pI @ pZ)
    expected = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(2), Qubit(0)],
                [Pauli.X, Pauli.Z],
            ): 1,
        }
    )

    assert observable.to_other_language(Language.TKET, targets=[2, 1, 0]) == expected
