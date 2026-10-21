"""Content lookup must retain submitted circuits and every individual run."""

from __future__ import annotations

from copy import deepcopy
from numbers import Complex
from pathlib import Path
from sqlite3 import connect
from typing import Any, Optional, cast
from unittest.mock import patch

import numpy as np
import pytest
from sympy import Expr, Symbol

from mpqp import (
    CNOT,
    BasisMeasure,
    Depolarizing,
    ExpectationMeasure,
    H,
    IBMDevice,
    Observable,
    QCircuit,
    Rx,
    X,
    pZ,
    run,
)
from mpqp.core._keys import decode_key, encode_key, object_key
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.measurement.basis import HadamardBasis
from mpqp.execution import BatchResult, Job, JobType, Result, StateVector
from mpqp.execution.devices import AvailableDevice


def make_result(circuit: QCircuit) -> Result:
    """Make a local result without running or authenticating a provider."""
    return Result(
        Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR),
        StateVector(np.eye(1, 2**circuit.nb_qubits, dtype=complex).ravel()),
        g_phase_handling=False,
    )


def test_lookup_and_sequence_compatibility():
    circuit = QCircuit([H(0), CNOT(0, 1)])
    result = make_result(circuit)
    other = make_result(QCircuit([X(0)]))
    batch = BatchResult([result, other])

    assert batch[circuit] is result
    assert batch[deepcopy(circuit)] is result
    assert batch[QCircuit([H(0), CNOT(0, 1)], label="another label")] is result
    assert batch[0] is result
    assert batch[-1] is other
    assert batch[:] == [result, other]
    assert list(batch) == [result, other]
    assert len(batch) == 2
    assert "BatchResult: 2 results" in str(batch)
    assert repr(batch).startswith("BatchResult([Result(")
    with pytest.raises(KeyError):
        batch[QCircuit([H(0)])]
    with pytest.raises(IndexError):
        batch[2]


def test_duplicates_and_detached_lists():
    circuit = QCircuit([H(0)])
    first, other, last = (
        make_result(circuit),
        make_result(QCircuit([X(0)])),
        make_result(deepcopy(circuit)),
    )
    inputs = [first, other, last]
    batch = BatchResult(inputs)
    inputs.clear()
    batch.results.clear()
    matches = batch.get_all(circuit)
    assert matches == [first, last]
    matches.clear()
    assert batch.get_all(circuit) == [first, last]
    assert list(batch) == [first, other, last]
    assert len(batch) == 3
    assert batch.get_all(QCircuit(0)) == []
    with pytest.raises(ValueError, match="get_all"):
        batch[circuit]


def test_snapshot_survives_nested_mutation():
    circuit = QCircuit([Rx(0.5, 0), BasisMeasure([0], shots=100)])
    original = deepcopy(circuit)
    result = make_result(circuit)
    batch = BatchResult([result])
    circuit.instructions[0].targets[0] = 1
    circuit.measurements[0].shots = 200
    circuit.add(X(0))
    assert batch[original] is result
    with pytest.raises(KeyError):
        batch[circuit]


@pytest.mark.parametrize(
    "value",
    [
        QCircuit(0),
        H(0),
        CNOT(1, 0),
        Rx(Symbol("theta", real=True), 0),
        CustomGate(np.array([[0, 1j], [1j, 0]]), [0]),
        BasisMeasure([0], shots=0, basis=HadamardBasis(1)),
        ExpectationMeasure(Observable(pZ), [0], shots=100),
        Observable(np.array([[1, 1j], [-1j, -1]])),
        Depolarizing(0.1, [0]),
    ],
)
def test_component_keys_are_independent_and_serializable(value: Any):
    key = object_key(value)
    assert key == object_key(deepcopy(value))
    assert {key: "found"}[object_key(deepcopy(value))] == "found"
    assert decode_key(encode_key(key)) == key


def test_composable_circuit_measurement_keys():
    circuit, measure = QCircuit([H(0)]), BasisMeasure([0], shots=100)
    key = object_key((circuit, measure))
    entries = {key: "found"}
    assert entries[object_key((deepcopy(circuit), deepcopy(measure)))] == "found"
    measure.shots += 1
    assert object_key((circuit, measure)) not in entries


def test_numeric_values_use_exact_keys():
    first = object_key(CustomGate(np.eye(2), [0]))
    # These matrices compare approximately equal under existing public equality.
    close = object_key(CustomGate(np.diag([1, np.exp(1e-9j)]), [0]))
    assert first != close
    assert object_key(np.eye(2)) == object_key(np.eye(2, dtype=complex))
    assert object_key(1) == object_key(1.0) == object_key(1 + 0j)
    symbol = Symbol("theta")
    assert object_key(symbol) != object_key(Symbol("theta", real=True))


@pytest.mark.parametrize(
    "observable",
    [Observable(pZ), Observable([1, -1]), Observable(np.diag([1, -1]))],
)
def test_observable_key_ignores_lazy_caches(
    observable: Observable, monkeypatch: pytest.MonkeyPatch
):
    before = object_key(observable)
    _ = observable.matrix
    _ = observable.pauli_string
    _ = observable.diagonal_elements
    monkeypatch.setattr(observable, "pre_transpiled", object())
    assert object_key(observable) == before
    observable.matrix = np.diag([2, -2])
    assert object_key(observable) != before


def test_semantic_fields_and_execution_caches(monkeypatch: pytest.MonkeyPatch):
    circuit = QCircuit([H(0), BasisMeasure([0], shots=100)])
    key = object_key(circuit)
    circuit.label = "renamed"
    monkeypatch.setattr(circuit, "transpiled_circuit", object())
    monkeypatch.setattr(circuit, "transpiled_noise_model", object())
    monkeypatch.setattr(circuit, "_generated_g_phase", 0.4)
    assert object_key(circuit) == key
    circuit.input_g_phase = 0.1
    assert object_key(circuit) != key
    circuit.input_g_phase = 0
    circuit.measurements[0].shots = 200
    assert object_key(circuit) != key
    assert object_key(QCircuit([H(0)])) == object_key(QCircuit([H(0)], nb_qubits=1))
    assert object_key(Observable(pZ, label="a")) != object_key(
        Observable(pZ, label="b")
    )


def test_unknown_extensions_fail_explicitly():
    class UnhandledGate(H):
        pass

    with pytest.raises(TypeError, match="explicit structural key"):
        object_key(UnhandledGate(0))
    with pytest.raises(ValueError, match="Non-finite"):
        object_key(float("nan"))


def test_runner_captures_all_inputs_before_execution():
    theta = Symbol("theta")
    circuit = QCircuit([Rx(theta, 0)])
    original = deepcopy(circuit)

    def execute(
        circ: QCircuit,
        device: AvailableDevice,
        values: Optional[dict[Expr | str, Complex]],
        *args: object,
    ) -> Result:
        assert values is not None
        adapted = circ.subs(values)
        circ.add(X(0))  # Simulate a provider mutating the shared input.
        return make_result(adapted)

    with patch("mpqp.execution.runner._run_single", side_effect=execute):
        values: dict[Expr | str, Complex] = {theta: cast(Complex, 0.5)}
        batch = run([circuit, circuit], [IBMDevice.AER_SIMULATOR], values)
    assert isinstance(batch, BatchResult)
    assert batch.get_all(original) == list(batch)
    assert len(batch) == 2
    with pytest.raises(KeyError):
        batch[circuit]


@pytest.mark.provider("qiskit")
def test_real_local_batch_and_multiple_devices():
    circuit = QCircuit([H(0), CNOT(0, 1)])
    original = deepcopy(circuit)
    batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    np.testing.assert_allclose(
        batch[circuit].amplitudes, [2**-0.5, 0, 0, 2**-0.5], atol=1e-14
    )
    assert batch[original] is batch[0]
    multiple = run(circuit, [IBMDevice.AER_SIMULATOR, IBMDevice.AER_SIMULATOR])
    assert isinstance(multiple, BatchResult)
    assert len(multiple.get_all(circuit)) == 2


@pytest.mark.provider("qiskit")
def test_sampling_lookup():
    circuit = QCircuit([X(0), BasisMeasure([0], shots=16)])
    original = deepcopy(circuit)
    batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    assert batch[original].counts == [0, 16]
    assert batch[circuit] is batch[0]


@pytest.mark.provider("qiskit")
def test_observable_lookup_uses_submitted_measurement():
    circuit = QCircuit(
        [X(1), ExpectationMeasure(Observable(pZ), [1], shots=0)], nb_qubits=2
    )
    original = deepcopy(circuit)
    batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    assert batch[original].expectation_values == pytest.approx(-1)
    assert batch[circuit] is batch[0]


@pytest.mark.provider("qiskit")
def test_symbolic_lookup_uses_submitted_parameters():
    theta = Symbol("theta")
    circuit = QCircuit([Rx(theta, 0)])
    values: dict[Expr | str, Complex] = {theta: cast(Complex, np.pi)}
    batch = run([circuit], IBMDevice.AER_SIMULATOR, values)
    assert isinstance(batch, BatchResult)
    np.testing.assert_allclose(np.abs(batch[circuit].amplitudes), [0, 1], atol=1e-14)


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mpqp.environment import env_manager
    from mpqp.local_storage.setup import setup_local_storage

    monkeypatch.setattr(env_manager, "MPQP_ENV", tmp_path / "mpqp.env")
    monkeypatch.setenv("DB_PATH", "")
    path = tmp_path / "results.db"
    setup_local_storage(str(path))
    return path


def test_storage_preserves_submitted_keys_and_cleans_up(isolated_storage: Path):
    original = QCircuit([Rx(Symbol("theta"), 0)])
    with patch(
        "mpqp.execution.runner._run_single",
        side_effect=lambda circ, dev, values, *args: make_result(circ.subs(values)),
    ):
        values: dict[Expr | str, Complex] = {"theta": cast(Complex, 0.5)}
        batch = run([original], IBMDevice.AER_SIMULATOR, values)
    assert isinstance(batch, BatchResult)
    ids = batch.save()
    restored = BatchResult.load_by_local_ids(ids)
    assert len(restored) == 1
    assert restored[original] is restored[0]
    assert restored.get_all(QCircuit([Rx(0.5, 0)])) == []
    with connect(isolated_storage) as connection:
        assert connection.execute("SELECT COUNT(*) FROM result_keys").fetchone() == (1,)
        connection.execute("DELETE FROM results")
        assert connection.execute("SELECT COUNT(*) FROM result_keys").fetchone() == (0,)


def test_legacy_storage_falls_back_to_job_circuit(isolated_storage: Path):
    from mpqp.local_storage.save import insert_results

    circuit = QCircuit([X(0)])
    ids = insert_results(make_result(circuit))
    restored = BatchResult.load_by_local_ids(ids)
    assert restored[circuit] is restored[0]
    assert len(restored) == 1
