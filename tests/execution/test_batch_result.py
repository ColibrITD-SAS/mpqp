"""Lazy filtering uses current result metadata and retains every run."""

from __future__ import annotations

import json
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
    CircuitBinding,
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
from mpqp.core._keys import Snapshot, object_key
from mpqp.core.binding import BindingMode
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


def test_chained_filtering_preserves_duplicates_order_and_current_metadata():
    circuit = QCircuit([H(0), BasisMeasure([0], label="readout")], label="input")
    other = QCircuit([X(0), BasisMeasure([0], label="readout")], label="other")
    results = [
        make_result(circuit),
        make_result(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR),
        make_result(other),
        make_result(circuit),
    ]
    batch = BatchResult(results)
    refined = (
        batch.get_matching(device=IBMDevice.AER_SIMULATOR)
        .get_matching(measurement=circuit.measurements[0])
        .get_matching(circuit=deepcopy(circuit))
    )
    assert list(refined) == [results[0], results[3]]
    assert refined[0] is results[0]
    assert refined[-1] is results[3]
    assert refined[:] == [results[0], results[3]]
    assert list(refined) == list(
        batch.get_matching(
            circuit=circuit,
            device=IBMDevice.AER_SIMULATOR,
            measurement=circuit.measurements[0],
        )
    )
    before = deepcopy(circuit)
    circuit.label = "edited"
    assert len(refined.get_matching(circuit=before)) == 0
    assert list(refined.get_matching(circuit=circuit)) == [results[0], results[3]]


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
    assert len(empty) == 0
    assert list(empty) == []
    assert len(empty.get_matching(device=IBMDevice.AER_SIMULATOR)) == 0
    assert len(batch) == 1


def test_queries_follow_nested_mutation_before_and_after_first_query():
    circuit = QCircuit([Rx(0.5, 0), BasisMeasure([0], shots=100)])
    before = deepcopy(circuit)
    result = make_result(circuit)
    batch = BatchResult([result])
    circuit.instructions[0].label = "edited before first query"
    assert len(batch.get_matching(circuit=before)) == 0
    assert batch.get_matching(circuit=deepcopy(circuit)).results == [result]
    before = deepcopy(circuit)
    circuit.measurements[0].shots = 200
    circuit.add(X(0))
    assert len(batch.get_matching(circuit=before)) == 0
    assert batch.get_matching(circuit=circuit).results == [result]
    assert len(batch.get_matching(measurement=before.measurements[0])) == 0
    assert batch.get_matching(measurement=circuit.measurements[0]).results == [result]


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
    assert len(batch.get_matching(circuit=before)) == 0
    assert batch.get_matching(circuit=original).results == [results[0]]


@pytest.mark.parametrize("label", [None, "", "input"])
def test_runner_keeps_display_names_separate_from_circuit_labels(label: Optional[str]):
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
    assert batch[0].job.circuit.label == label
    assert batch[0].label == ("circuit 1" if label is None else None)


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


def test_queries_follow_job_device_and_measurement_changes():
    circuit = QCircuit([H(0), BasisMeasure([0], shots=16)])
    original = deepcopy(circuit)
    result = make_result(circuit)
    batch = BatchResult([result])
    assert batch.get_matching(device=IBMDevice.AER_SIMULATOR).results == [result]
    result.job.device = AWSDevice.BRAKET_LOCAL_SIMULATOR
    circuit.measurements[0].shots = 32
    assert len(batch.get_matching(device=IBMDevice.AER_SIMULATOR)) == 0
    assert batch.get_matching(device=result.job.device).results == [result]
    assert len(batch.get_matching(measurement=original.measurements[0])) == 0
    assert batch.get_matching(measurement=circuit.measurements[0]).results == [result]


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
def test_component_comparisons_are_independent(value: Any):
    key = object_key(value)
    assert key == object_key(deepcopy(value))
    assert {key: "found"}[object_key(deepcopy(value))] == "found"


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


def test_runner_and_nonstructural_queries_do_not_capture_inputs():
    circuit = QCircuit([H(0)])
    with patch(
        "mpqp.core._keys.Snapshot.__call__",
        side_effect=AssertionError("Unnecessary capture"),
    ), patch(
        "mpqp.execution.runner._run_single",
        side_effect=lambda circ, *args: make_result(circ),
    ):
        batch = run([circuit, circuit], IBMDevice.AER_SIMULATOR)
        assert isinstance(batch, BatchResult)
        assert len(batch) == 2
        assert batch[:1] == [batch[0]]
        assert list(batch.get_matching()) == list(batch)
        assert list(batch.get_matching(device=IBMDevice.AER_SIMULATOR)) == list(batch)
        assert list(batch.get_matching(measurement=None)) == list(batch)


@pytest.mark.provider("qiskit")
def test_real_local_batch_and_multiple_devices():
    circuit = QCircuit([H(0), CNOT(0, 1)])
    original = deepcopy(circuit)
    batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    assert circuit.label is None
    assert isinstance(batch[0].job.circuit, QCircuit)
    assert batch[0].job.circuit.label is None
    assert "circuit 1" in str(batch[0])
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
def test_observable_lookup_uses_recorded_measurement():
    circuit = QCircuit(
        [X(1), ExpectationMeasure(Observable(pZ), [1], shots=0)], nb_qubits=2
    )
    original = deepcopy(circuit)
    batch = run([circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    recorded = batch[0].job.circuit
    assert isinstance(recorded, QCircuit)
    assert len(batch.get_matching(circuit=original)) == 0
    assert batch.get_matching(circuit=deepcopy(recorded))[
        0
    ].expectation_values == pytest.approx(-1)
    assert batch.get_matching(measurement=deepcopy(batch[0].job.measure)).results == [
        batch[0]
    ]


@pytest.mark.provider("qiskit")
def test_symbolic_lookup_uses_recorded_parameters():
    theta = Symbol("theta")
    circuit = QCircuit([Rx(theta, 0)])
    values: dict[Expr | str, Complex] = {theta: cast(Complex, np.pi)}
    batch = run([circuit], IBMDevice.AER_SIMULATOR, values)
    assert isinstance(batch, BatchResult)
    assert len(batch.get_matching(circuit=circuit)) == 0
    recorded = batch[0].job.circuit
    assert isinstance(recorded, QCircuit)
    matches = batch.get_matching(circuit=deepcopy(recorded))
    assert list(matches) == [batch[0]]
    np.testing.assert_allclose(np.abs(matches[0].amplitudes), [0, 1], atol=1e-14)


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mpqp.environment import env_manager
    from mpqp.local_storage.setup import setup_local_storage

    monkeypatch.setattr(env_manager, "MPQP_ENV", tmp_path / "mpqp.env")
    monkeypatch.setenv("DB_PATH", "")
    path = tmp_path / "results.db"
    setup_local_storage(str(path))
    return path


@pytest.mark.provider("qiskit")
def test_binding_matches_input_and_recorded_circuits():
    circuits = [QCircuit([X(0)]), QCircuit([H(0)])]
    measurement = ExpectationMeasure(Observable(pZ))
    binding = CircuitBinding(circuits, measurements=measurement)
    with patch(
        "mpqp.core._keys.Snapshot.__call__",
        side_effect=AssertionError("Unnecessary capture"),
    ):
        batch = run(binding, IBMDevice.AER_SIMULATOR)

    assert len(batch) == 2
    assert [r.expectation_values for r in batch] == pytest.approx([-1, 0])
    for circuit, result in zip(circuits, batch):
        assert isinstance(result.job.circuit, QCircuit)
        assert result.input_circuit is circuit
        assert batch.get_matching(circuit=deepcopy(circuit)).results == [result]
        assert batch.get_matching(circuit=deepcopy(result.job.circuit)).results == [
            result
        ]
        assert batch.get_matching(
            circuit=circuit, device=result.device, measurement=result.job.measure
        ).results == [result]
        assert len(batch.get_matching(circuit=circuit, measurement=None)) == 0
        assert len(circuit.measurements) == 0

    selected = batch.get_matching(circuit=circuits[0])
    original = deepcopy(circuits[0])
    circuits[0].label = "edited input"
    assert len(selected.get_matching(circuit=original)) == 0
    assert selected.get_matching(circuit=deepcopy(circuits[0])).results == [batch[0]]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("mode", [BindingMode.PRODUCT, BindingMode.ZIP])
@pytest.mark.parametrize("embedded", [False, True])
def test_binding_inputs_follow_nested_sweeps_and_repeated_circuits(
    mode: BindingMode, embedded: bool
):
    theta = Symbol("theta")
    first, second = QCircuit([Rx(theta, 0)]), QCircuit([Rx(theta, 0), X(0)])
    if embedded:
        first.add(ExpectationMeasure(Observable(pZ)))
        second.add(ExpectationMeasure(Observable(pZ)))
    values = (
        [{"theta": 0.0}, {"theta": np.pi}]
        if mode == BindingMode.PRODUCT
        else [{"theta": 0.0}, {"theta": np.pi}, {"theta": np.pi / 2}]
    )
    binding = CircuitBinding(
        [CircuitBinding(first), CircuitBinding(second), CircuitBinding(first)],
        values=values,
        measurements=None if embedded else ExpectationMeasure(Observable(pZ)),
        mode=mode,
    )
    # The public unroll contract remains a sequence of three-item executions.
    assert all(len(execution) == 3 for execution in binding.unroll())
    batch = run(binding, IBMDevice.AER_SIMULATOR)
    first_positions = [0, 1, 4, 5] if mode == BindingMode.PRODUCT else [0, 2]
    second_positions = [2, 3] if mode == BindingMode.PRODUCT else [1]
    assert batch.get_matching(circuit=deepcopy(first)).results == [
        batch[i] for i in first_positions
    ]
    assert batch.get_matching(circuit=deepcopy(second)).results == [
        batch[i] for i in second_positions
    ]
    expected = [1, -1, -1, 1, 1, -1] if mode == BindingMode.PRODUCT else [1, 1, 0]
    assert [r.expectation_values for r in batch] == pytest.approx(expected, abs=1e-14)
    assert len(first.measurements) == int(embedded)


@pytest.mark.provider("qiskit")
def test_binding_input_matching_survives_storage(isolated_storage: Path):
    circuit = QCircuit([X(0)])
    measured = QCircuit([X(0), ExpectationMeasure(Observable(pZ))])
    binding = CircuitBinding(
        [
            CircuitBinding(measured),
            CircuitBinding(circuit, measurements=ExpectationMeasure(Observable(pZ))),
        ]
    )
    batch = run(binding, IBMDevice.AER_SIMULATOR)
    restored = BatchResult.load_by_local_ids(batch.save())
    assert isinstance(restored[0].job.circuit, QCircuit)
    assert restored.get_matching(circuit=deepcopy(circuit)).results == [restored[1]]
    assert restored.get_matching(circuit=deepcopy(measured)).results == list(restored)
    assert restored.get_matching(
        circuit=deepcopy(restored[0].job.circuit)
    ).results == list(restored)
    assert restored.get_matching(
        circuit=circuit, measurement=deepcopy(batch[1].job.measure)
    ).results == [restored[1]]
    assert restored[0].input_circuit is not None
    assert restored[1].input_circuit is not None
    assert len(restored[0].input_circuit.measurements) == 1
    assert len(restored[1].input_circuit.measurements) == 0


@pytest.mark.parametrize(
    "device,provider",
    [
        (IBMDevice.AER_SIMULATOR, "run_ibm"),
        (AWSDevice.BRAKET_LOCAL_SIMULATOR, "run_braket"),
    ],
)
def test_common_binding_runner_attaches_inputs(device: AvailableDevice, provider: str):
    circuits = [QCircuit([X(0)]), QCircuit([H(0)])]
    binding = CircuitBinding(circuits)
    results = [make_result(deepcopy(circuit), device) for circuit in circuits]
    with patch(f"mpqp.execution.runner.{provider}", return_value=BatchResult(results)):
        batch = run(binding, device)
    assert [result.input_circuit for result in batch] == circuits
    assert all(
        result.input_circuit is circuit for result, circuit in zip(batch, circuits)
    )


def test_binding_result_count_must_match_inputs():
    binding = CircuitBinding(QCircuit([X(0)]))
    with patch("mpqp.execution.runner.run_ibm", return_value=BatchResult([])):
        with pytest.raises(ValueError, match="one result per resolved execution"):
            run(binding, IBMDevice.AER_SIMULATOR)


@pytest.mark.parametrize(
    "device,kind",
    [
        pytest.param(
            IBMDevice.AER_SIMULATOR, "observable", marks=pytest.mark.provider("qiskit")
        ),
        pytest.param(
            IBMDevice.AER_SIMULATOR, "sample", marks=pytest.mark.provider("qiskit")
        ),
        pytest.param(
            IBMDevice.AER_SIMULATOR, "state", marks=pytest.mark.provider("qiskit")
        ),
        pytest.param(
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            "observable",
            marks=pytest.mark.provider("braket"),
        ),
        pytest.param(
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            "state",
            marks=pytest.mark.provider("braket"),
        ),
    ],
)
def test_binding_resolves_once_per_run(device: AvailableDevice, kind: str):
    if kind == "observable":
        measurement = ExpectationMeasure(
            Observable(pZ), shots=16 if isinstance(device, AWSDevice) else 0
        )
    elif kind == "sample":
        measurement = BasisMeasure([0], shots=16)
    else:
        measurement = None
    circuit = QCircuit([X(0)])
    binding = CircuitBinding(circuit, measurements=measurement)
    resolve = getattr(CircuitBinding, "_unroll_with_inputs")
    calls = 0

    def count_resolution(instance: CircuitBinding):
        nonlocal calls
        calls += 1
        return resolve(instance)

    with patch.object(CircuitBinding, "_unroll_with_inputs", new=count_resolution):
        first = run(binding, device)
        assert calls == 1
        binding.circuits.append(QCircuit([H(0)]))
        second = run(binding, device)
        assert calls == 2
    assert len(first) == 1
    assert len(second) == 2
    assert second.get_matching(circuit=circuit).results == [second[0]]


def test_circuit_shape_rejection_avoids_structural_capture():
    source = QCircuit([H(0)], label="input")
    recorded = QCircuit([H(0), ExpectationMeasure(Observable(pZ))], label="prepared")
    result = make_result(recorded)
    result.input_circuit = source
    query = deepcopy(recorded)
    query.label = "unrelated"
    with patch(
        "mpqp.core._keys.Snapshot.__call__",
        side_effect=AssertionError("Unnecessary structural capture"),
    ):
        assert len(BatchResult([result]).get_matching(circuit=query)) == 0


def test_shared_inputs_are_serialized_once_per_save(isolated_storage: Path):
    source = QCircuit(1)
    independent = deepcopy(source)
    recorded = QCircuit([BasisMeasure([0], shots=0)])
    results = [make_result(recorded) for _ in range(3)]
    for result, input_circuit in zip(results, [source, independent, source]):
        result.input_circuit = input_circuit
    batch = BatchResult(results)
    represent = QCircuit.__repr__
    calls = 0

    def count_input_repr(circuit: QCircuit):
        nonlocal calls
        if circuit is source:
            calls += 1
        return represent(circuit)

    with patch.object(QCircuit, "__repr__", new=count_input_repr):
        first_ids = batch.save()
        assert calls == 1
        source.label = "edited input"
        second_ids = batch.save()
        assert calls == 2
    first = BatchResult.load_by_local_ids(first_ids)
    second = BatchResult.load_by_local_ids(second_ids)
    assert len(first.get_matching(circuit=QCircuit(1))) == 3
    assert second.get_matching(circuit=deepcopy(source)).results == [
        second[0],
        second[2],
    ]
    assert second.get_matching(circuit=deepcopy(independent)).results == [second[1]]


def test_job_save_initializes_local_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from mpqp.environment import env_manager
    from mpqp.local_storage.setup import setup_local_storage

    monkeypatch.setattr(env_manager, "MPQP_ENV", tmp_path / "mpqp.env")
    monkeypatch.setenv("DB_PATH", "")
    path = tmp_path / "results.db"
    monkeypatch.setattr(
        "mpqp.local_storage.setup.setup_local_storage",
        lambda: setup_local_storage(str(path)),
    )

    job = Job(JobType.STATE_VECTOR, QCircuit([X(0)]), IBMDevice.AER_SIMULATOR)
    job_id = job.save()

    assert path.is_file()
    assert Job.load_by_local_id(job_id) == job


def test_save_and_load_do_not_capture_inputs(isolated_storage: Path):
    original = QCircuit([Rx(Symbol("theta"), 0)])
    values: dict[Expr | str, Complex] = {"theta": cast(Complex, 0.5)}
    with patch(
        "mpqp.core._keys.Snapshot.__call__",
        side_effect=AssertionError("Unnecessary capture"),
    ), patch(
        "mpqp.execution.runner._run_single",
        side_effect=lambda circ, dev, values, *args: make_result(circ.subs(values)),
    ):
        batch = run([original], IBMDevice.AER_SIMULATOR, values)
        assert isinstance(batch, BatchResult)
        restored = BatchResult.load_by_local_ids(batch.save())
    assert len(restored) == 1
    assert len(restored.get_matching(circuit=original)) == 0
    assert restored.get_matching(circuit=QCircuit([Rx(0.5, 0)])).results == [
        restored[0]
    ]


def test_legacy_storage_falls_back_to_job_circuit(isolated_storage: Path):
    from mpqp.local_storage.save import insert_results

    circuit = QCircuit([X(0)])
    ids = insert_results(make_result(circuit))
    restored = BatchResult.load_by_local_ids(ids)
    assert restored.get_matching(circuit=circuit).results == [restored[0]]
    assert len(restored) == 1


def test_storage_preserves_generated_result_labels(isolated_storage: Path):
    circuit = QCircuit([X(0)])
    with patch(
        "mpqp.execution.runner._run_single",
        side_effect=lambda circ, *args: make_result(circ),
    ):
        batch = run([circuit, circuit], IBMDevice.AER_SIMULATOR)
    assert isinstance(batch, BatchResult)
    restored = BatchResult.load_by_local_ids(batch.save())

    assert [result.label for result in restored] == ["circuit 1", "circuit 2"]
    assert str(restored) == str(batch)
    assert circuit.label is None
    for result in restored:
        assert isinstance(result.job.circuit, QCircuit)
        assert result.job.circuit.label is None
    assert len(restored.get_matching(circuit=deepcopy(circuit))) == 2


@pytest.mark.parametrize("label", [None, "", "Alice's result"])
def test_storage_preserves_explicit_result_labels(
    isolated_storage: Path, label: Optional[str]
):
    from mpqp.local_storage.load import results_local_storage_to_mpqp
    from mpqp.local_storage.queries import fetch_results_with_id

    result = make_result(QCircuit([X(0)], label="circuit label"))
    result.label = label
    result_id = result.save()
    restored = Result.load_by_local_id(result_id)
    restored_record = results_local_storage_to_mpqp(
        fetch_results_with_id(result_id)[0]
    )[0]

    assert restored.label == restored_record.label == label
    assert str(restored) == str(restored_record) == str(result)


@pytest.mark.parametrize("entry_point", ["load", "save", "save_job", "setup"])
@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_storage_upgrades_legacy_result_labels(
    isolated_storage: Path, entry_point: str, version: str
):
    from mpqp.local_storage.load import results_local_storage_to_mpqp
    from mpqp.local_storage.save import insert_results
    from mpqp.local_storage.setup import (
        DATABASE_VERSION,
        get_database_version,
        setup_local_storage,
    )

    circuit = QCircuit([X(0)], label="legacy circuit")
    state = StateVector(np.array([0, 1], dtype=np.complex128))
    original = Result(
        Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR),
        state,
    )
    with connect(isolated_storage) as connection:
        connection.executescript("""
            DROP TABLE results;
            CREATE TABLE results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                data TEXT NOT NULL,
                error TEXT,
                shots INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            UPDATE version SET version = '1.0' WHERE id = 1;
            """)
        if version == "1.1":
            original.label = "legacy display name"
            connection.execute("ALTER TABLE results ADD COLUMN label TEXT")
            connection.execute("UPDATE version SET version = '1.1' WHERE id = 1")
        connection.execute(
            "INSERT INTO jobs (id, type, circuit, device) VALUES (?, ?, ?, ?)",
            (7, "STATE_VECTOR", json.dumps(repr(circuit)), str(original.device)),
        )
        connection.execute(
            "INSERT INTO results (id, job_id, data, shots, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (11, 7, json.dumps(repr(state)), 0, "2026-01-01 00:00:00"),
        )
        if version == "1.1":
            connection.execute(
                "UPDATE results SET label = ? WHERE id = 11", (original.label,)
            )
        cursor = connection.execute("SELECT * FROM results WHERE id = 11")
        legacy_record = dict(
            zip((column[0] for column in cursor.description), cursor.fetchone())
        )

    new_result = make_result(circuit)
    new_result.label = "new display name"
    if entry_point == "load":
        Result.load_by_local_id(11)
    elif entry_point == "save":
        insert_results(new_result, reuse_similar_job=False)
    elif entry_point == "save_job":
        original.job.save()
    else:
        setup_local_storage(str(isolated_storage))

    assert get_database_version() == DATABASE_VERSION
    restored = Result.load_by_local_id(11)
    assert restored.label == original.label
    assert str(restored) == str(original)
    np.testing.assert_array_equal(restored.amplitudes, [0, 1])
    assert results_local_storage_to_mpqp(legacy_record)[0].label == original.label
    with connect(isolated_storage) as connection:
        cursor = connection.execute("SELECT * FROM results WHERE id = 11")
        migrated_record = dict(
            zip((column[0] for column in cursor.description), cursor.fetchone())
        )
    assert migrated_record == {
        **legacy_record,
        "label": original.label,
        "input_circuit": None,
    }

    # Reopening and writing again must work after the one-time upgrade.
    setup_local_storage(str(isolated_storage))
    assert Result.load_by_local_id(new_result.save()).label == new_result.label


@pytest.mark.parametrize("entry_point", ["load", "save_job"])
def test_storage_rejects_unsupported_database_version(
    isolated_storage: Path, entry_point: str
):
    from mpqp.local_storage.setup import get_database_version

    with connect(isolated_storage) as connection:
        connection.execute("UPDATE version SET version = '2.0' WHERE id = 1")

    with pytest.raises(RuntimeError, match="Database version 2.0 is not supported"):
        if entry_point == "save_job":
            Job(JobType.STATE_VECTOR, QCircuit([X(0)]), IBMDevice.AER_SIMULATOR).save()
        else:
            Result.load_by_local_id(1)
    assert get_database_version() == "2.0"


@pytest.mark.parametrize("with_old_metadata", [False, True])
def test_storage_uses_recorded_metadata(
    isolated_storage: Path, with_old_metadata: bool
):
    from mpqp.local_storage.save import insert_results

    circuit = QCircuit([H(0)])
    ids = insert_results(make_result(circuit))
    if with_old_metadata:
        with connect(isolated_storage) as connection:
            connection.execute(
                "CREATE TABLE result_keys (result_id INTEGER PRIMARY KEY, circuit_key TEXT)"
            )
            connection.execute(
                "INSERT INTO result_keys VALUES (?, ?)",
                (ids[0], json.dumps((3, (("scalar", None),) * 3))),
            )
    restored = BatchResult.load_by_local_ids(ids)
    assert restored.get_matching(circuit=circuit).results == [restored[0]]


def test_storage_preserves_recorded_labels(isolated_storage: Path):
    circuit = QCircuit([H(0), BasisMeasure([0], label="readout")], label="input")
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
    batch = BatchResult([make_result(circuit), make_result(QCircuit([X(0)]))])
    refined = batch.get_matching(device=IBMDevice.AER_SIMULATOR).get_matching(
        circuit=original, measurement=original.measurements[0]
    )
    circuit.label = "edited"
    circuit.measurements[0].label = "edited readout"
    restored = BatchResult.load_by_local_ids(refined.save())
    assert len(restored) == 1
    assert restored.get_matching(circuit=circuit)[0] is restored[0]
    assert restored.get_matching(measurement=circuit.measurements[0])[0] is restored[0]
    assert len(restored.get_matching(circuit=original)) == 0


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


def test_matrix_queries_distinguish_exact_content():
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
    batch = BatchResult(results)
    assert batch.get_matching(circuit=deepcopy(first)).results == [results[0]]
    assert batch.get_matching(circuit=deepcopy(second)).results == [results[1]]


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
def test_native_gate_keys_include_all_structural_fields(
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


@pytest.mark.parametrize("gate_class,name", [(H, "H"), (CNOT, "CNOT"), (Rx, "Rx")])
def test_gate_classes_use_type_keys(gate_class: type, name: str):
    expected = ("type", "mpqp.core.instruction.gates.native_gates", name)
    key = object_key(gate_class)
    assert key == expected
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
    assert len(batch.get_matching(circuit=original)) == 0
    assert batch.get_matching(circuit=circuit).results == [first, second]


def test_storage_preserves_gate_class_selectors(isolated_storage: Path):
    circuit = QCircuit([H(0), Depolarizing(0.1, gates=[H])])
    original = deepcopy(circuit)
    batch = BatchResult([make_result(circuit)])
    circuit.noises[0].gates[0] = X
    restored = BatchResult.load_by_local_ids(batch.save())
    assert len(restored) == 1
    assert len(restored.get_matching(circuit=original)) == 0
    assert restored.get_matching(circuit=circuit).results == [restored[0]]


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
    assert len(batch.get_matching(measurement=None)) == 0
    restored = BatchResult.load_by_local_ids(batch.save())
    assert restored.get_matching(measurement=original).results == [restored[0]]
    assert len(restored.get_matching(measurement=None)) == 0
    measurement.shots = 32
    assert len(batch.get_matching(measurement=original)) == 0
    assert batch.get_matching(measurement=measurement).results == [result]


def test_basic_batch_operations_do_not_capture_metadata():
    results = [
        make_result(QCircuit([H(0)])),
        make_result(QCircuit([X(0)]), AWSDevice.BRAKET_LOCAL_SIMULATOR),
    ]
    with patch(
        "mpqp.core._keys.Snapshot.__call__",
        side_effect=AssertionError("Unnecessary capture"),
    ):
        batch = BatchResult(results)
        assert len(batch) == 2
        assert batch[0] is results[0]
        assert batch[:] == results
        assert list(batch) == results
        assert batch.get_matching().results == results
        assert batch.get_matching(device=IBMDevice.AER_SIMULATOR).results == results[:1]


def test_measurement_queries_do_not_inspect_circuits():
    class UnsupportedGate(H):
        pass

    measurement = BasisMeasure([0], shots=16)
    circuit = QCircuit([UnsupportedGate(0), measurement])
    result = make_result(circuit)
    batch = BatchResult([result])
    assert batch.get_matching(measurement=deepcopy(measurement)).results == [result]
    with pytest.raises(TypeError, match="explicit structural key"):
        batch.get_matching(circuit=circuit)


def test_device_rejection_avoids_structural_capture():
    result = make_result(QCircuit([H(0)]))
    batch = BatchResult([result])
    with patch(
        "mpqp.core._keys.Snapshot.__call__",
        side_effect=AssertionError("Unnecessary capture"),
    ):
        assert (
            len(
                batch.get_matching(
                    device=AWSDevice.BRAKET_LOCAL_SIMULATOR,
                    circuit=QCircuit([X(0)]),
                    measurement=BasisMeasure([0]),
                )
            )
            == 0
        )


def test_repeated_queries_follow_current_metadata():
    circuit = QCircuit([Rx(0.5, 0)])
    before = deepcopy(circuit)
    batch = BatchResult([make_result(circuit), make_result(circuit)])
    assert len(batch.get_matching(circuit=before)) == 2
    gate = circuit.instructions[0]
    assert isinstance(gate, Rx)
    gate.parameters[0] = 0.8
    assert len(batch.get_matching(circuit=before)) == 0
    assert len(batch.get_matching(circuit=deepcopy(circuit))) == 2
