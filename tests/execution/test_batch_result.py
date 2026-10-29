"""Content lookup must retain submitted circuits and every individual run."""

from __future__ import annotations

import json
import pickle
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
    AWSDevice,
    Barrier,
    BasisMeasure,
    Breakpoint,
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
from mpqp.core._keys import (
    Snapshot,
    decode_key,
    decode_result_key,
    encode_key,
    encode_result_key,
    object_key,
    result_key,
)
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.native_gates import CP, PRX, SWAP, TOF, CRk, Rxx, U
from mpqp.core.instruction.measurement.basis import HadamardBasis
from mpqp.execution import BatchResult, Job, JobType, Result, Sample, StateVector
from mpqp.execution.devices import AvailableDevice


def make_result(
    circuit: QCircuit, device: AvailableDevice = IBMDevice.AER_SIMULATOR
) -> Result:
    """Make a local result without running or authenticating a provider."""
    return Result(
        Job(JobType.STATE_VECTOR, circuit, device),
        StateVector(np.eye(1, 2**circuit.nb_qubits, dtype=complex).ravel()),
        g_phase_handling=False,
    )


def test_lookup_and_sequence_compatibility():
    circuit = QCircuit([H(0), CNOT(0, 1)])
    result = make_result(circuit)
    other = make_result(QCircuit([X(0)]))
    batch = BatchResult([result, other])

    assert batch.get_matching(circuit=circuit).results == [result]
    assert batch.get_matching(circuit=deepcopy(circuit)).results == [result]
    assert (
        batch.get_matching(
            circuit=QCircuit([H(0), CNOT(0, 1)], label="another label")
        ).results
        == []
    )
    assert batch[0] is result
    assert batch[-1] is other
    assert batch[:] == [result, other]
    assert list(batch) == [result, other]
    assert len(batch) == 2
    assert "BatchResult: 2 results" in str(batch)
    assert repr(batch).startswith("BatchResult([Result(")
    assert batch.get_matching(circuit=QCircuit([H(0)])).results == []
    with pytest.raises(TypeError):
        batch[cast(Any, circuit)]
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
    matches = batch.get_matching(circuit=circuit)
    assert isinstance(matches, BatchResult)
    assert matches.results == [first, last]
    matches.results.clear()
    assert list(matches) == [first, last]
    assert batch.get_matching(circuit=circuit).results == [first, last]
    assert list(batch) == [first, other, last]
    assert len(batch) == 3
    assert batch.get_matching(circuit=QCircuit(0)).results == []
    assert batch.get_matching().results == [first, other, last]
    assert BatchResult([]).get_matching().results == []


def test_chained_filtering_retains_snapshots_duplicates_and_order():
    circuit = QCircuit([H(0), BasisMeasure([0], label="readout")], label="input")
    original = deepcopy(circuit)
    other = QCircuit([X(0), BasisMeasure([0], label="readout")], label="other")
    results = [
        make_result(circuit),
        make_result(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR),
        make_result(other),
        make_result(circuit),
    ]
    batch = BatchResult(results)
    circuit.label = "edited"
    circuit.measurements[0].label = "edited readout"
    results[0].job.device = AWSDevice.BRAKET_LOCAL_SIMULATOR

    by_device = batch.get_matching(device=IBMDevice.AER_SIMULATOR)
    refined = by_device.get_matching(measurement=original.measurements[0])
    refined = refined.get_matching(circuit=original)
    assert isinstance(refined, BatchResult)
    assert list(refined) == [results[0], results[3]]
    assert refined[0] is results[0]
    assert refined[-1] is results[3]
    assert refined[:] == [results[0], results[3]]
    assert (
        refined.results
        == batch.get_matching(
            circuit=original,
            device=IBMDevice.AER_SIMULATOR,
            measurement=original.measurements[0],
        ).results
    )
    assert len(batch) == 4
    assert len(by_device) == 3


def test_full_and_empty_selections_are_independent_batches():
    result = make_result(QCircuit([H(0)]))
    batch = BatchResult([result])
    selected = batch.get_matching()
    assert isinstance(selected, BatchResult)
    assert selected is not batch
    assert selected[0] is result
    selected.results.clear()
    assert list(selected) == list(batch) == [result]

    empty = batch.get_matching(circuit=QCircuit([X(0)]))
    assert isinstance(empty, BatchResult)
    assert not empty
    assert list(empty) == []
    assert len(empty.get_matching(device=IBMDevice.AER_SIMULATOR)) == 0
    assert len(batch) == 1


def test_snapshot_survives_nested_mutation():
    circuit = QCircuit([Rx(0.5, 0), BasisMeasure([0], shots=100)])
    original = deepcopy(circuit)
    result = make_result(circuit)
    batch = BatchResult([result])
    circuit.instructions[0].targets[0] = 1
    circuit.measurements[0].shots = 200
    circuit.add(X(0))
    assert batch.get_matching(circuit=original).results == [result]
    assert batch.get_matching(circuit=circuit).results == []
    assert batch.get_matching(measurement=original.measurements[0]).results == [result]
    assert batch.get_matching(measurement=circuit.measurements[0]).results == []


@pytest.mark.parametrize(
    "value",
    [
        QCircuit([H(0)]),
        H(0),
        CNOT(0, 1),
        Rx(0.5, 0),
        CustomGate(np.eye(2), [0]),
        BasisMeasure([0]),
        ExpectationMeasure(Observable(pZ), [0]),
        Observable(pZ),
        Barrier(1),
        Breakpoint(),
    ],
)
def test_object_labels_are_part_of_keys(value: Any):
    value = deepcopy(value)
    value.label = "first"
    equivalent = deepcopy(value)
    assert object_key(equivalent) == object_key(value)
    equivalent.label = "second"
    assert object_key(equivalent) != object_key(value)
    value.label = None
    equivalent.label = ""
    assert object_key(equivalent) != object_key(value)


def test_circuit_gate_and_measurement_labels_distinguish_runs():
    original = QCircuit([H(0), BasisMeasure([0], label="readout")], label="input")
    circuit_label, gate_label, measurement_label = [
        deepcopy(original) for _ in range(3)
    ]
    circuit_label.label = "other input"
    gate_label.instructions[0].label = "other gate"
    measurement_label.measurements[0].label = "other readout"
    circuits = [original, circuit_label, gate_label, measurement_label]
    results = [make_result(circuit) for circuit in circuits]
    batch = BatchResult(results)
    for circuit, result in zip(circuits, results):
        assert batch.get_matching(circuit=deepcopy(circuit)).results == [result]
    assert (
        batch.get_matching(measurement=original.measurements[0]).results == results[:3]
    )
    assert batch.get_matching(
        measurement=measurement_label.measurements[0]
    ).results == [results[3]]
    before = deepcopy(original)
    original.label = "renamed"
    assert batch.get_matching(circuit=before).results == [results[0]]
    assert batch.get_matching(circuit=original).results == []


@pytest.mark.parametrize("label", [None, "", "input"])
def test_runner_preserves_submitted_labels(label: Optional[str]):
    circuit = QCircuit([H(0)], label=label)
    before = deepcopy(circuit)
    with patch(
        "mpqp.execution.runner._run_single",
        side_effect=lambda circ, *args: make_result(circ),
    ):
        batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    assert circuit.label == label
    assert batch.get_matching(circuit=before).results == [batch[0]]
    assert isinstance(batch[0].job.circuit, QCircuit)
    assert batch[0].job.circuit.label == ("circuit 1" if label is None else label)


@pytest.mark.parametrize(
    "use_circuit,use_device,use_measurement,positions",
    [
        (False, False, False, [0, 1, 2, 3, 4, 5]),
        (True, False, False, [0, 2, 4]),
        (False, True, False, [0, 3, 4, 5]),
        (False, False, True, [0, 1, 2, 4]),
        (True, True, False, [0, 4]),
        (True, False, True, [0, 2, 4]),
        (False, True, True, [0, 4]),
        (True, True, True, [0, 4]),
    ],
)
def test_criteria_combine_with_and_in_execution_order(
    use_circuit: bool, use_device: bool, use_measurement: bool, positions: list[int]
):
    first = QCircuit([H(0), BasisMeasure([0], shots=16)])
    second = QCircuit([X(0), BasisMeasure([0], shots=16)])
    third = QCircuit([H(0), BasisMeasure([0], shots=32)])
    ibm, aws = IBMDevice.AER_SIMULATOR, AWSDevice.BRAKET_LOCAL_SIMULATOR
    results = [
        make_result(first, ibm),
        make_result(second, aws),
        make_result(deepcopy(first), aws),
        make_result(third, ibm),
        make_result(deepcopy(first), ibm),
        make_result(QCircuit([H(0)]), ibm),
    ]
    criteria: dict[str, Any] = {}
    if use_circuit:
        criteria["circuit"] = deepcopy(first)
    if use_device:
        criteria["device"] = ibm
    if use_measurement:
        criteria["measurement"] = deepcopy(first.measurements[0])
    batch = BatchResult(results)
    assert batch.get_matching(**criteria).results == [
        results[index] for index in positions
    ]
    assert batch.get_matching(measurement=None).results == [results[5]]
    assert batch.get_matching(circuit=first, measurement=None).results == []
    assert batch.get_matching(circuit=first, device=aws).results == [results[2]]
    assert batch.get_matching(device=IBMDevice.AER_SIMULATOR_STATEVECTOR).results == []


@pytest.mark.parametrize("criterion", ["circuit", "device", "measurement"])
def test_invalid_criteria_fail_explicitly(criterion: str):
    with pytest.raises(TypeError, match=criterion):
        BatchResult([]).get_matching(**{criterion: cast(Any, "invalid")})


def test_submitted_metadata_survives_job_mutation():
    circuit = QCircuit([H(0), BasisMeasure([0], shots=16)])
    original = deepcopy(circuit)
    result = make_result(circuit)
    batch = BatchResult([result])
    result.job.device = AWSDevice.BRAKET_LOCAL_SIMULATOR
    circuit.measurements[0].shots = 32
    assert batch.get_matching(device=IBMDevice.AER_SIMULATOR).results == [result]
    assert batch.get_matching(device=result.job.device).results == []
    assert batch.get_matching(measurement=original.measurements[0]).results == [result]


def test_submitted_key_serialization_and_legacy_fallback():
    circuit = QCircuit([H(0), BasisMeasure([0], shots=16)])
    device = IBMDevice.AER_SIMULATOR
    key = result_key(circuit, device)
    assert decode_result_key(encode_result_key(key), device, circuit) == key
    recorded = deepcopy(circuit)
    recorded.label = "recorded execution"
    assert decode_result_key(
        encode_key(object_key(circuit)), device, recorded
    ) == result_key(recorded, device)
    legacy = legacy_payload(key, 2)
    assert decode_result_key(legacy, device, recorded) == result_key(recorded, device)
    without_measurement = QCircuit([H(0)])
    assert decode_result_key(
        encode_key(object_key(without_measurement)), device, without_measurement
    ) == result_key(without_measurement, device)
    with pytest.raises(ValueError, match="version"):
        decode_result_key('[99, []]', device, circuit)


def test_no_measurement_is_distinct_from_zero_shots():
    implicit = make_result(QCircuit([H(0)]))
    explicit = make_result(QCircuit([H(0), BasisMeasure([0], shots=0)]))
    batch = BatchResult([implicit, explicit])
    assert batch.get_matching().results == [implicit, explicit]
    assert batch.get_matching(measurement=None).results == [implicit]
    assert batch.get_matching(measurement=explicit.job.measure).results == [explicit]


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
    assert object_key(circuit) != key
    circuit.label = None
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
    assert batch.get_matching(circuit=original).results == list(batch)
    assert len(batch) == 2
    assert batch.get_matching(circuit=circuit).results == []


@pytest.mark.provider("qiskit")
def test_real_local_batch_and_multiple_devices():
    circuit = QCircuit([H(0), CNOT(0, 1)])
    original = deepcopy(circuit)
    batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    assert circuit.label is None
    assert isinstance(batch[0].job.circuit, QCircuit)
    assert batch[0].job.circuit.label == "circuit 1"
    np.testing.assert_allclose(
        batch.get_matching(circuit=circuit)[0].amplitudes,
        [2**-0.5, 0, 0, 2**-0.5],
        atol=1e-14,
    )
    assert batch.get_matching(circuit=original).results == [batch[0]]
    multiple = run(circuit, [IBMDevice.AER_SIMULATOR, IBMDevice.AER_SIMULATOR])
    assert isinstance(multiple, BatchResult)
    assert len(multiple.get_matching(circuit=circuit)) == 2


@pytest.mark.provider("qiskit")
def test_sampling_lookup():
    circuit = QCircuit([X(0), BasisMeasure([0], shots=16)])
    original = deepcopy(circuit)
    batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    assert batch.get_matching(circuit=original)[0].counts == [0, 16]
    assert batch.get_matching(circuit=circuit).results == [batch[0]]


@pytest.mark.provider("qiskit")
def test_observable_lookup_uses_submitted_measurement():
    circuit = QCircuit(
        [X(1), ExpectationMeasure(Observable(pZ), [1], shots=0)], nb_qubits=2
    )
    original = deepcopy(circuit)
    batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    assert batch.get_matching(circuit=original)[0].expectation_values == pytest.approx(
        -1
    )
    assert batch.get_matching(circuit=circuit).results == [batch[0]]
    assert batch.get_matching(measurement=original.measurements[0]).results == [
        batch[0]
    ]


@pytest.mark.provider("qiskit")
def test_symbolic_lookup_uses_submitted_parameters():
    theta = Symbol("theta")
    circuit = QCircuit([Rx(theta, 0)])
    values: dict[Expr | str, Complex] = {theta: cast(Complex, np.pi)}
    batch = run([circuit], IBMDevice.AER_SIMULATOR, values)
    assert isinstance(batch, BatchResult)
    np.testing.assert_allclose(
        np.abs(batch.get_matching(circuit=circuit)[0].amplitudes), [0, 1], atol=1e-14
    )


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
    assert restored.get_matching(circuit=original).results == [restored[0]]
    assert restored.get_matching(circuit=QCircuit([Rx(0.5, 0)])).results == []
    with connect(isolated_storage) as connection:
        assert connection.execute("SELECT COUNT(*) FROM result_keys").fetchone() == (1,)
        connection.execute("DELETE FROM results")
        assert connection.execute("SELECT COUNT(*) FROM result_keys").fetchone() == (0,)


def test_legacy_storage_falls_back_to_job_circuit(isolated_storage: Path):
    from mpqp.local_storage.save import insert_results

    circuit = QCircuit([X(0)])
    ids = insert_results(make_result(circuit))
    restored = BatchResult.load_by_local_ids(ids)
    assert restored.get_matching(circuit=circuit).results == [restored[0]]
    assert len(restored) == 1


@pytest.mark.parametrize("legacy_version", [None, 1, 2])
def test_storage_uses_submitted_or_legacy_recorded_criteria(
    isolated_storage: Path, legacy_version: Optional[int]
):
    from mpqp.local_storage.save import insert_results

    original = QCircuit(
        [X(1), ExpectationMeasure(Observable(pZ), [1], shots=0)], nb_qubits=2
    )
    adapted = QCircuit([X(1), ExpectationMeasure(Observable(pZ @ pZ), [0, 1], shots=0)])
    executed = Result(Job(JobType.OBSERVABLE, adapted, IBMDevice.AER_SIMULATOR), -1.0)
    if legacy_version is not None:
        # Legacy formats did not reliably preserve original labels.
        ids = insert_results(executed)
        old_key = result_key(original, executed.device)
        data = (
            encode_key(old_key.circuit)
            if legacy_version == 1
            else legacy_payload(old_key, 2)
        )
        with connect(isolated_storage) as connection:
            connection.execute(
                "CREATE TABLE result_keys (result_id INTEGER PRIMARY KEY, circuit_key TEXT)"
            )
            connection.execute(
                "INSERT INTO result_keys VALUES (?, ?)",
                (ids[0], data),
            )
    else:
        with patch("mpqp.execution.runner._run_single", return_value=executed):
            batch = run([original], IBMDevice.AER_SIMULATOR)
        assert isinstance(batch, BatchResult)
        ids = batch.save()
    restored = BatchResult.load_by_local_ids(ids)
    expected = adapted if legacy_version is not None else original
    assert restored.get_matching(
        circuit=expected,
        device=IBMDevice.AER_SIMULATOR,
        measurement=expected.measurements[0],
    ).results == [restored[0]]
    other = original if legacy_version is not None else adapted
    assert restored.get_matching(measurement=other.measurements[0]).results == []


def test_storage_preserves_submitted_labels(isolated_storage: Path):
    circuit = QCircuit([H(0), BasisMeasure([0], label="readout")], label="input")
    circuit.instructions[0].label = "preparation"
    batch = BatchResult([make_result(circuit)])
    restored = BatchResult.load_by_local_ids(batch.save())
    assert restored.get_matching(circuit=deepcopy(circuit)).results == [restored[0]]
    assert restored.get_matching(
        measurement=deepcopy(circuit.measurements[0])
    ).results == [restored[0]]
    different = deepcopy(circuit)
    different.instructions[0].label = "other preparation"
    assert restored.get_matching(circuit=different).results == []
    different = deepcopy(circuit.measurements[0])
    different.label = "other readout"
    assert restored.get_matching(measurement=different).results == []


def test_refined_batch_can_be_saved_and_loaded(isolated_storage: Path):
    circuit = QCircuit([H(0), BasisMeasure([0], label="readout")], label="input")
    original = deepcopy(circuit)
    results = [make_result(circuit), make_result(QCircuit([X(0)]))]
    batch = BatchResult(results)
    circuit.label = "edited"
    circuit.measurements[0].label = "edited readout"
    refined = batch.get_matching(device=IBMDevice.AER_SIMULATOR).get_matching(
        circuit=original, measurement=original.measurements[0]
    )
    restored = BatchResult.load_by_local_ids(refined.save())
    assert len(restored) == 1
    assert restored.get_matching(circuit=original)[0] is restored[0]
    assert restored.get_matching(measurement=original.measurements[0])[0] is restored[0]
    assert not restored.get_matching(circuit=circuit)


def legacy_payload(key: Any, version: int) -> str:
    """Reproduce pre-packed-array snapshots, including their scalar tuples."""

    def unpack(value: Any) -> Any:
        if not isinstance(value, tuple):
            return value
        if value and value[0] == "array-v2":
            _, shape, data, exceptions = value
            overrides = dict(exceptions)
            items = tuple(
                overrides.get(i, object_key(item))
                for i, item in enumerate(np.frombuffer(data, dtype="<c16"))
            )
            return ("array", shape, items)
        return tuple(unpack(item) for item in value)

    return json.dumps((version, unpack((key.circuit, key.device, key.measurement))))


@pytest.mark.parametrize(
    "dtype",
    [
        np.int32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        object,
    ],
)
def test_packed_array_numeric_equality(dtype: Any):
    value = np.array([[0, 1], [-0.0, 2]], dtype=dtype)
    expected = object_key(np.array([[0, 1], [0, 2]], dtype=np.complex128))
    assert object_key(value) == expected
    assert hash(object_key(value)) == hash(expected)
    assert decode_key(encode_key(object_key(value))) == expected


@pytest.mark.parametrize(
    "value",
    [
        np.array(1.0),
        np.array([], dtype=float),
        np.arange(24).reshape(4, 6)[:, ::2],
        np.array([1j, -0.0, -1j], dtype=">c16"),
        np.array([2**63 - 1], dtype=np.int64),
        np.array([2**64 - 1], dtype=np.uint64),
        np.array([2**60], dtype=np.int64),
        np.array([2**53 + 1, 2**53], dtype=np.uint64),
        np.array([Symbol("theta"), 1], dtype=object),
        np.array([True, False]),
        np.array([10**400], dtype=object),
    ],
)
def test_packed_arrays_preserve_layout_independence_and_exact_values(value: Any):
    key = object_key(value)
    assert key == object_key(value.astype(object))
    assert key == object_key(np.asfortranarray(value).reshape(value.shape))
    assert decode_key(encode_key(key)) == key
    saved = value.copy()
    if value.size:
        value.flat[0] = 0
        assert key == object_key(saved)


def test_packed_arrays_do_not_round_large_integers():
    integer = np.array([2**53 + 1], dtype=np.int64)
    rounded = integer.astype(float)
    assert object_key(integer) != object_key(rounded)
    assert object_key(np.array([2**60])) == object_key(np.array([float(2**60)]))
    assert object_key(np.array([True])) != object_key(np.array([1]))
    assert object_key(np.array([0.0])) == object_key(np.array([-0.0 + 0j]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, complex(1, np.inf)])
def test_packed_arrays_reject_nonfinite_values(bad: Any):
    with pytest.raises(ValueError, match="Non-finite"):
        object_key(np.array([bad]))


def test_snapshot_reuses_shared_arrays_without_caching_mutable_sources():
    value = np.eye(4)
    snapshot = Snapshot()
    first = snapshot(value)
    assert snapshot(value) is first
    assert snapshot([value, value])[1] == (first, first)
    value[0, 0] = 2
    assert object_key(value) != first


def test_version_three_matrix_keys_preserve_original_submitted_inputs():
    circuit = QCircuit(
        [H(0), ExpectationMeasure(Observable(np.diag([1.0, -1.0])), [0], shots=0)],
        label="submitted",
    )
    key = result_key(circuit, IBMDevice.AER_SIMULATOR)
    payload = legacy_payload(key, 3)
    recorded = deepcopy(circuit)
    recorded.label = "adapted"
    restored = decode_result_key(payload, IBMDevice.AER_SIMULATOR, recorded)
    assert restored == key
    assert hash(restored) == hash(key)
    assert restored != result_key(recorded, IBMDevice.AER_SIMULATOR)


def test_cached_hash_is_recomputed_after_serialization():
    key = result_key(QCircuit([H(0)]), IBMDevice.AER_SIMULATOR)
    expected_hash = hash(key)
    object.__setattr__(key, "_hash", expected_hash + 1)
    restored = pickle.loads(pickle.dumps(key))
    assert restored == key
    assert hash(restored) == expected_hash
    assert (
        hash(
            decode_result_key(
                encode_result_key(key), IBMDevice.AER_SIMULATOR, QCircuit(1)
            )
        )
        == expected_hash
    )


def test_packed_keys_resolve_hash_collisions_by_exact_content():
    first = QCircuit(
        [ExpectationMeasure(Observable(np.diag([1.0, -1.0])), [0], shots=0)]
    )
    second = QCircuit(
        [
            ExpectationMeasure(
                Observable(np.diag([1.0, np.nextafter(-1.0, 0.0)])), [0], shots=0
            )
        ]
    )
    results = [make_result(first), make_result(second)]
    for result in results:
        assert isinstance(result.job.circuit, QCircuit)
        key = result_key(result.job.circuit, result.device)
        object.__setattr__(key, "_hash", 0)
        result._submitted_key = key  # pyright: ignore[reportPrivateUsage]
    batch = BatchResult(results)
    assert len(batch) == 2
    assert batch.get_matching(circuit=first).results == [results[0]]
    assert batch.get_matching(circuit=second).results == [results[1]]


@pytest.mark.parametrize(
    "gate,fields",
    [
        (H(0), ("targets", "label")),
        (SWAP(0, 1), ("targets", "label")),
        (Rx(0.5, 0), ("targets", "parameters", "label")),
        (Rx(Symbol("theta"), 0), ("targets", "parameters", "label")),
        (Rxx(0.25, 0, 1), ("targets", "parameters", "label")),
        (U(0.2, 0.3, 0.4, 0), ("targets", "parameters", "label")),
        (PRX(0.2, 0.4, 0), ("targets", "parameters", "label")),
        (CNOT(0, 1), ("targets", "controls", "non_controlled_gate", "label")),
        (TOF([0, 1], 2), ("targets", "controls", "non_controlled_gate", "label")),
        (
            CP(0.3, 0, 1),
            ("targets", "controls", "non_controlled_gate", "parameters", "label"),
        ),
        (
            CRk(3, 0, 1),
            ("targets", "controls", "non_controlled_gate", "parameters", "label"),
        ),
    ],
)
def test_native_gate_keys_preserve_the_stored_structural_format(
    gate: Any, fields: tuple[str, ...]
):
    expected = (
        "object",
        type(gate).__module__,
        type(gate).__qualname__,
        tuple((field, object_key(getattr(gate, field))) for field in fields),
    )
    key = object_key(gate)
    assert key == expected
    assert decode_key(encode_key(key)) == expected
    original = deepcopy(gate)
    gate.label = "edited gate"
    assert object_key(gate) != key
    assert object_key(original) == key


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("targets", [2]),
        ("targets", (0,)),
        ("targets", [True]),
        ("targets", np.array([0])),
        ("parameters", [0]),
        ("parameters", [-0.0]),
        ("parameters", [0.5 + 0j]),
        ("parameters", [np.float64(0.5)]),
        ("parameters", [2**100]),
        ("parameters", [Symbol("theta")]),
        ("parameters", [complex(0.1, 0.2)]),
    ],
)
def test_native_gate_fields_keep_general_numeric_and_symbolic_semantics(
    attribute: str, value: Any
):
    gate = Rx(0.5, 0)
    setattr(gate, attribute, value)
    expected = (
        "object",
        type(gate).__module__,
        type(gate).__qualname__,
        (
            ("targets", object_key(gate.targets)),
            ("parameters", object_key(gate.parameters)),
            ("label", object_key(gate.label)),
        ),
    )
    assert object_key(gate) == expected


@pytest.mark.parametrize("bad", [np.nan, np.inf, complex(1, np.inf)])
def test_native_gate_fast_path_rejects_nonfinite_parameters(bad: Any):
    gate = Rx(0.5, 0)
    gate.parameters = [bad]
    with pytest.raises(ValueError, match="Non-finite"):
        object_key(gate)


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly", "big_endian"])
@pytest.mark.parametrize("negative_zero", [False, True])
def test_array_fast_path_preserves_source_bits_and_snapshot_independence(
    layout: str, negative_zero: bool
):
    values = np.array([[1 + 2j, 0], [0, -1 - 2j]], dtype=np.complex128)
    if negative_zero:
        values.view(np.float64).flat[2] = -0.0
        values.view(np.float64).flat[5] = -0.0
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = np.repeat(values, 2, axis=1)[:, ::2]
    elif layout == "readonly":
        values.flags.writeable = False
    elif layout == "big_endian":
        values = values.astype(">c16")
    before = values.tobytes()
    key = object_key(values)
    assert values.tobytes() == before
    expected = object_key(np.array([[1 + 2j, 0], [0, -1 - 2j]], dtype=object))
    assert key == expected
    assert hash(key) == hash(expected)
    if values.flags.writeable:
        values[0, 0] = 3
        assert key != object_key(values)
    assert decode_key(encode_key(key)) == expected


@pytest.mark.parametrize("gate_class,name", [(H, "H"), (CNOT, "CNOT"), (Rx, "Rx")])
def test_gate_classes_use_type_keys(gate_class: type, name: str):
    expected = ("type", "mpqp.core.instruction.gates.native_gates", name)
    key = object_key(gate_class)
    assert key == expected
    assert decode_key(encode_key(key)) == expected
    assert object_key((gate_class,)) == ("sequence", (expected,))
    assert key != object_key(H(0))


def test_noisy_circuit_matching_distinguishes_gate_classes():
    circuit = QCircuit([H(0), Depolarizing(0.1, gates=[H])])
    original = deepcopy(circuit)
    other = QCircuit([H(0), Depolarizing(0.1, gates=[X])])
    first, second = make_result(circuit), make_result(other)
    batch = BatchResult([first, second])
    assert batch.get_matching(circuit=original).results == [first]
    assert batch.get_matching(circuit=other).results == [second]
    circuit.noises[0].gates[0] = X
    assert batch.get_matching(circuit=original).results == [first]
    assert batch.get_matching(circuit=circuit).results == [second]


def test_storage_preserves_gate_class_selectors(isolated_storage: Path):
    circuit = QCircuit([H(0), Depolarizing(0.1, gates=[H])])
    original = deepcopy(circuit)
    batch = BatchResult([make_result(circuit)])
    circuit.noises[0].gates[0] = X
    restored = BatchResult.load_by_local_ids(batch.save())
    assert len(restored) == 1
    assert restored.get_matching(circuit=original).results == [restored[0]]
    assert not restored.get_matching(circuit=circuit)


def test_separate_job_measurement_matching_and_storage(isolated_storage: Path):
    circuit = QCircuit([H(0)])
    measurement = BasisMeasure([0], shots=16)
    original = deepcopy(measurement)
    result = Result(
        Job(JobType.SAMPLE, circuit, IBMDevice.AER_SIMULATOR, measurement=measurement),
        [Sample(1, index=0, count=16)],
        shots=16,
    )
    batch = BatchResult([result])
    assert batch.get_matching(circuit=circuit, measurement=original).results == [result]
    assert not batch.get_matching(measurement=None)
    restored = BatchResult.load_by_local_ids(batch.save())
    assert restored.get_matching(measurement=original).results == [restored[0]]
    assert not restored.get_matching(measurement=None)
    measurement.shots = 32
    assert batch.get_matching(measurement=original).results == [result]
    assert not batch.get_matching(measurement=measurement)


def test_legacy_keys_fall_back_to_separate_job_measurement():
    circuit = QCircuit([H(0)])
    measurement = BasisMeasure([0], shots=16)
    key = result_key(circuit, IBMDevice.AER_SIMULATOR, measurement=measurement)
    restored = decode_result_key(
        legacy_payload(key, 2),
        IBMDevice.AER_SIMULATOR,
        circuit,
        measurement=measurement,
    )
    assert restored.measurement == object_key(measurement)
