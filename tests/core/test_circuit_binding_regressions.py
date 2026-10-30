from copy import deepcopy
from importlib import import_module
from itertools import permutations, product
from unittest.mock import patch

import numpy as np
import pytest
from sympy import Symbol

from mpqp import (
    AWSDevice,
    BasisMeasure,
    ExpectationMeasure,
    IBMDevice,
    Observable,
    QCircuit,
    Ry,
    Rz,
    X,
    pI,
    pX,
    pZ,
    run,
)
from mpqp.core.binding import BindingMode, CircuitBinding
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.execution.job import JobType
from mpqp.execution.providers.providers_params import QiskitParams
from mpqp.execution.result import BatchResult


def test_public_binding_exports_use_renamed_module():
    import mpqp

    assert mpqp.CircuitBinding is CircuitBinding
    assert mpqp.BindingMode is BindingMode


def test_zip_keeps_observables_on_their_own_register():
    circuits = [QCircuit(3), QCircuit(4)]
    measurements = [
        ExpectationMeasure(Observable(pI @ pI @ pZ)),
        ExpectationMeasure(Observable(pI @ pI @ pI @ pZ)),
    ]
    original = deepcopy(measurements)
    binding = CircuitBinding(circuits, measurements=measurements, mode=BindingMode.ZIP)
    for _ in range(2):
        executions = binding.unroll()
        assert [c.nb_qubits for c, _, _ in executions] == [3, 4]
        assert [
            m.observables[  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                0
            ].nb_qubits
            for _, _, m in executions
        ] == [
            3,
            4,
        ]
    assert measurements == original
    assert all(not c.measurements for c in circuits)


@pytest.mark.parametrize("mode", list(BindingMode))
@pytest.mark.parametrize("nested", [False, True])
def test_shared_measure_is_embedded_per_resolved_circuit(
    mode, nested  # pyright: ignore[reportMissingParameterType]
):
    circuits = [QCircuit(3), QCircuit(4)]
    # Reordered targets also exercise adjust_measure's in-place reordering path.
    measurement = ExpectationMeasure(Observable(pZ @ pX), targets=[2, 0])
    original = deepcopy(measurement)
    branches = [CircuitBinding(c) for c in circuits] if nested else circuits
    binding = CircuitBinding(branches, measurements=measurement, mode=mode)
    for _ in range(2):
        executions = binding.unroll()
        assert len(executions) == 2
        for (circuit, _, adjusted), expected in zip(
            executions, [pX @ pI @ pZ, pX @ pI @ pZ @ pI]
        ):
            assert (
                adjusted.targets  # pyright: ignore[reportOptionalMemberAccess]
                == list(range(circuit.nb_qubits))
            )
            np.testing.assert_allclose(
                adjusted.observables[  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                    0
                ].matrix,
                Observable(expected).matrix,
            )
    assert measurement == original


def test_product_rejects_measurement_too_wide_for_one_circuit():
    binding = CircuitBinding(
        [QCircuit(3), QCircuit(4)],
        measurements=ExpectationMeasure(Observable(pI @ pI @ pI @ pZ)),
        mode=BindingMode.PRODUCT,
    )
    with pytest.raises(ValueError, match="resolved circuit"):
        binding.unroll()


@pytest.mark.parametrize("lengths", list(product((1, 2), repeat=3)))
@pytest.mark.parametrize("nested", [False, True])
def test_zip_broadcasts_each_axis(lengths: tuple[int, int, int], nested: bool):
    theta = Symbol("theta")
    circuits = [QCircuit([Ry(theta, 0)]), QCircuit([Ry(theta, 0), X(0)])]
    values = [{"theta": 0.0}, {"theta": np.pi}]
    measurements = [
        ExpectationMeasure(Observable(pZ), label="Z"),
        ExpectationMeasure(Observable(pX), label="X"),
    ]
    nc, nv, nm = lengths
    branches = [CircuitBinding(circuit) for circuit in circuits] if nested else circuits
    binding = CircuitBinding(
        branches[:nc],
        values=values[:nv],
        measurements=measurements[:nm],
        mode=BindingMode.ZIP,
    )

    executions = binding.unroll()

    assert len(executions) == max(lengths)
    for i, (circuit, parameters, measurement) in enumerate(executions):
        assert circuit is circuits[0 if nc == 1 else i]
        assert parameters == values[0 if nv == 1 else i]
        assert measurement is not None
        assert measurement.label == measurements[0 if nm == 1 else i].label


@pytest.mark.parametrize("omit_values", [False, True])
@pytest.mark.parametrize("omit_measurements", [False, True])
def test_zip_broadcasts_omitted_axes(omit_values: bool, omit_measurements: bool):
    values = None if omit_values else [{"theta": 0.0}, {"theta": np.pi}]
    measurements = (
        None
        if omit_measurements
        else [ExpectationMeasure(Observable(pZ)), ExpectationMeasure(Observable(pX))]
    )
    binding = CircuitBinding(
        [QCircuit(1), QCircuit(1)],
        values=values,
        measurements=measurements,
        mode=BindingMode.ZIP,
    )

    executions = binding.unroll()

    assert len(executions) == 2
    for i, (_, parameters, measurement) in enumerate(executions):
        assert parameters == (None if values is None else values[i])
        if measurements is None:
            assert measurement is None
        else:
            assert isinstance(measurement, ExpectationMeasure)
            assert measurement.observables == measurements[i].observables


@pytest.mark.parametrize(
    "lengths",
    list(permutations((1, 2, 3))) + [(2, 3, None), (2, None, 3), (1, 0, 1), (1, 1, 0)],
)
def test_zip_rejects_incompatible_axes_at_construction(
    lengths: tuple[int, int | None, int | None],
):
    nc, nv, nm = lengths
    with pytest.raises(ValueError, match="each axis must have length 1"):
        CircuitBinding(
            [QCircuit(1) for _ in range(nc)],
            values=None if nv is None else [{} for _ in range(nv)],
            measurements=(
                None
                if nm is None
                else [ExpectationMeasure(Observable(pZ)) for _ in range(nm)]
            ),
            mode=BindingMode.ZIP,
        )


@pytest.mark.parametrize("lengths", list(product((1, 2), repeat=3)))
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(IBMDevice.AER_SIMULATOR, marks=pytest.mark.provider("qiskit")),
        pytest.param(
            AWSDevice.BRAKET_LOCAL_SIMULATOR, marks=pytest.mark.provider("braket")
        ),
    ],
)
def test_zip_broadcast_execution_order_and_values(
    lengths: tuple[int, int, int], device: IBMDevice | AWSDevice
):
    theta = Symbol("theta")
    nc, nv, nm = lengths
    circuits = [QCircuit([Ry(theta, 0)]), QCircuit([Ry(theta, 0), X(0)])]
    binding = CircuitBinding(
        circuits[:nc],
        values=[{"theta": 0.0}, {"theta": np.pi}][:nv],
        measurements=[
            ExpectationMeasure(Observable(pZ), shots=1000),
            ExpectationMeasure(Observable(pI), shots=1000),
        ][:nm],
        mode=BindingMode.ZIP,
    )
    # At position 1, X and Ry(pi) each flip Z's sign when their axes vary.
    # The second measurement is identity, whose expectation is always one.
    expected = [1.0]
    if max(lengths) == 2:
        expected.append(1.0 if nm == 2 else (-1.0) ** ((nc == 2) + (nv == 2)))

    result = run(binding, device)

    assert isinstance(result, BatchResult)
    assert [r.expectation_values for r in result.results] == pytest.approx(expected)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("mode", list(BindingMode))
@pytest.mark.parametrize("symbol_keys", [False, True])
def test_parameter_order_and_observable_association(
    mode: BindingMode, symbol_keys: bool
):
    a, z = Symbol("a"), Symbol("z")
    circuit = QCircuit([Ry(z, 0), Rz(a, 0)])
    values = [{a: 0.0, z: 0.0}, {a: 0.0, z: np.pi}]
    if not symbol_keys:
        values = [{str(k): v for k, v in item.items()} for item in values]
    binding = CircuitBinding(circuit, values=values, measurements=None, mode=mode)  # type: ignore
    # Put the measurement on an outer level to broadcast it in both modes.
    binding = CircuitBinding(
        binding,
        measurements=ExpectationMeasure(
            [Observable(pZ, label="Z"), Observable(pX, label="X")]
        ),
    )
    # Deliberately reverse the set's possible order to catch positional binding.
    with patch.object(circuit, "variables", return_value=[z, a]):
        result = run(binding, IBMDevice.AER_SIMULATOR)
    assert len(result.results) == 2
    assert result.results[0].expectation_values == pytest.approx({"Z": 1, "X": 0})
    assert result.results[1].expectation_values == pytest.approx({"Z": -1, "X": 0})


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("targets", [[0], [1], [1, 0]])
@pytest.mark.parametrize("explicit_cbits", [False, True])
def test_explicit_measurement_registers(targets: list[int], explicit_cbits: bool):
    measure = BasisMeasure(
        targets,
        c_targets=list(range(len(targets))) if explicit_cbits else None,
        shots=16,
    )
    original = deepcopy(measure)
    binding = CircuitBinding(QCircuit([X(0)], nb_qubits=2), measurements=measure)
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert isinstance(result, BatchResult)
    execution = result.results[0]
    expected = int("".join("1" if target == 0 else "0" for target in targets), 2)
    assert execution.counts[expected] == 16
    assert sum(execution.counts) == 16
    assert measure == original


@pytest.mark.parametrize("many_circuits", [False, True])
@pytest.mark.parametrize("many_devices", [False, True])
def test_provider_parameters_forwarded(many_circuits: bool, many_devices: bool):
    runner = import_module("mpqp.execution.runner")
    circuits = [QCircuit(1), QCircuit(1)] if many_circuits else QCircuit(1)
    devices = (
        [IBMDevice.AER_SIMULATOR, IBMDevice.AER_SIMULATOR_STATEVECTOR]
        if many_devices
        else IBMDevice.AER_SIMULATOR
    )
    params = QiskitParams(instance="selected-instance")
    with patch.object(runner, "_run_single", return_value=None) as execute:
        runner.run(circuits, devices, provider_params=params)
    assert execute.call_count == (2 if many_circuits else 1) * (
        2 if many_devices else 1
    )
    assert all(call.args[4] is params for call in execute.call_args_list)


def test_binding_runs_on_every_device():
    runner = import_module("mpqp.execution.runner")
    binding = CircuitBinding(QCircuit(1))
    devices = [IBMDevice.AER_SIMULATOR, IBMDevice.AER_SIMULATOR_STATEVECTOR]
    with patch.object(
        runner, "_run_circuit_binding", side_effect=[BatchResult([]), BatchResult([])]
    ) as execute:
        result = runner.run(binding, devices)
    assert isinstance(result, BatchResult)
    assert [call.args[1] for call in execute.call_args_list] == devices


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "mode, expected",
    [
        (BindingMode.ZIP, [1.0, 0.0]),
        (BindingMode.PRODUCT, [1.0, 0.0, -1.0, 0.0]),
    ],
)
def test_grouped_binding_modes(mode: BindingMode, expected: float):
    theta = Symbol("theta")
    circuit = QCircuit([Ry(theta, 0)])
    binding = CircuitBinding(
        circuit,
        values=[{theta: 0.0}, {theta: np.pi}],
        measurements=[
            ExpectationMeasure(Observable(pZ)),
            ExpectationMeasure(Observable(pX)),
        ],
        mode=mode,
    )
    with patch.object(binding, "unroll", wraps=binding.unroll) as unroll:
        pubs = binding.to_other_device(IBMDevice.AER_SIMULATOR)
    assert unroll.call_count == 1
    assert len(pubs) == 1
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert [r.expectation_values for r in result.results] == pytest.approx(expected)


@pytest.mark.provider("qiskit")
def test_grouped_pub_without_parameters_and_ragged_observables():
    circuit = QCircuit([X(0)])
    binding = CircuitBinding(
        [circuit, circuit],
        measurements=[
            ExpectationMeasure(Observable(pZ, label="Z")),
            ExpectationMeasure([Observable(pZ, label="Z"), Observable(pX, label="X")]),
        ],
        mode=BindingMode.ZIP,
    )
    pubs = binding.to_other_device(IBMDevice.AER_SIMULATOR)
    assert len(pubs) == 1
    assert len(pubs[0][0]) == 2  # type: ignore
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert len(result.results) == 2
    assert result.results[0].expectation_values == pytest.approx(-1)
    assert result.results[1].expectation_values == pytest.approx({"Z": -1, "X": 0})


@pytest.mark.provider("qiskit")
def test_binding_preserves_pre_measurement_basis():
    from mpqp import H
    from mpqp.core.instruction.measurement.basis import HadamardBasis

    measure = BasisMeasure(shots=16, basis=HadamardBasis())
    binding = CircuitBinding(QCircuit([H(0)]), measurements=measure)
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert isinstance(result, BatchResult)
    assert result.results[0].counts == [16, 0]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("nested", [False, True])
def test_grouping_preserves_execution_associations(nested: bool):
    first = QCircuit([X(0)])
    second = QCircuit(1)
    binding = CircuitBinding(
        [first, second, first],
        measurements=[
            ExpectationMeasure(Observable(pZ), label="first_Z"),
            ExpectationMeasure(Observable(pZ), label="second_Z"),
            ExpectationMeasure(Observable(pX), label="first_X"),
        ],
        mode=BindingMode.ZIP,
    )
    if nested:
        binding = CircuitBinding(binding)
    pubs = binding.to_other_device(IBMDevice.AER_SIMULATOR)
    assert len(pubs) == 2
    for (_, contexts), expected_circuit in zip(pubs, [first, second]):
        for context in contexts:
            assert isinstance(context.job.circuit, QCircuit)
            assert context.job.circuit.gates == expected_circuit.gates
    assert [[context.index for context in contexts] for _, contexts in pubs] == [
        [0, 2],
        [1],
    ]
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert len(result.results) == 3
    # Associate values through their execution metadata, independently of PUB order.
    actual = {}
    for execution in result.results:
        assert execution.job.measure is not None
        actual[execution.job.measure.label] = execution.expectation_values
    assert actual == pytest.approx({"first_Z": -1, "second_Z": 1, "first_X": 0})
    assert [r.expectation_values for r in result.results] == pytest.approx([-1, 1, 0])


@pytest.mark.parametrize(
    "shots, override, expected",
    [
        (0, None, JobType.STATE_VECTOR),
        (16, None, JobType.SAMPLE),
        (1024, 0, JobType.STATE_VECTOR),
        (1024, 16, JobType.SAMPLE),
    ],
)
def test_binding_basis_measure_job_type(
    shots: int, override: int | None, expected: JobType
):
    binding = CircuitBinding(
        QCircuit([X(0)]), measurements=BasisMeasure(shots=shots), shots=override
    )
    assert binding.job_type == expected


@pytest.mark.parametrize("shots", [0, 16])
def test_binding_expectation_measure_job_type(shots: int):
    binding = CircuitBinding(
        QCircuit(1), measurements=ExpectationMeasure(Observable(pZ), shots=shots)
    )
    assert binding.job_type == JobType.OBSERVABLE


@pytest.mark.parametrize("expectation_first", [False, True])
def test_binding_rejects_mixed_measurement_types(expectation_first: bool):
    measurements = [BasisMeasure(shots=0), ExpectationMeasure(Observable(pZ))]
    if expectation_first:
        measurements.reverse()
    with pytest.raises(ValueError, match="same type"):
        CircuitBinding(QCircuit(1), measurements=measurements)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_binding_zero_shots_returns_state_vector(embedded: bool, override: bool):
    measure = BasisMeasure() if override else BasisMeasure(shots=0)
    shots = 0 if override else None
    binding = (
        CircuitBinding(QCircuit([X(0), measure]), shots=shots)
        if embedded
        else CircuitBinding(QCircuit([X(0)]), measurements=measure, shots=shots)
    )
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert len(result.results) == 1
    execution = result.results[0]
    assert execution.job.job_type == JobType.STATE_VECTOR
    np.testing.assert_allclose(execution.amplitudes, [0, 1], atol=1e-12)
    assert execution.shots == 0


@pytest.mark.skip(reason="Re-enable after merging the branch with the VQA one.")
@pytest.mark.parametrize("widths", [(1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize(
    "measure",
    [ExpectationMeasure(Observable(pZ), targets=[0]), BasisMeasure(targets=[0])],
)
def test_binding_rejects_mixed_widths_with_fixed_measurement(
    widths: tuple[int, int], measure: Measure
):
    circuits = [QCircuit(width) for width in widths]
    if widths[0] == widths[1]:
        CircuitBinding(circuits, measurements=measure)
    else:
        with pytest.raises(ValueError, match="(?i)(qubit|width)"):
            CircuitBinding(circuits, measurements=measure)


@pytest.mark.skip(reason="Re-enable after merging the branch with the VQA one.")
@pytest.mark.parametrize("widths", [(1, 2), (2, 1)])
@pytest.mark.parametrize("mode", list(BindingMode))
def test_binding_accepts_mixed_widths_with_dynamic_measurement(
    widths: tuple[int, int], mode: BindingMode
):
    binding = CircuitBinding(
        [QCircuit(width) for width in widths],
        measurements=BasisMeasure(),
        mode=mode,
    )

    executions = binding.unroll()
    assert [circuit.nb_qubits for circuit, _, _ in executions] == list(widths)
    assert all(isinstance(measure, BasisMeasure) for _, _, measure in executions)


@pytest.mark.parametrize("expectation", [False, True])
@pytest.mark.parametrize("many", [False, True])
def test_binding_shot_override_preserves_input_measurement(
    expectation: bool, many: bool
):
    measure = ExpectationMeasure(Observable(pZ)) if expectation else BasisMeasure()
    original = deepcopy(measure)
    original_shots = measure.shots
    measurements = [measure, deepcopy(measure)] if many else [measure]
    binding = CircuitBinding(
        QCircuit(1), measurements=measurements if many else measure, shots=16
    )
    reused = CircuitBinding(QCircuit(1), measurements=measure)

    assert measure.shots == original_shots
    assert measure == original
    assert binding.shots == 16
    assert binding.measurements is not None
    assert binding.measurements[0] is not measure
    assert binding.measurements[0].shots == 16
    assert len(binding.measurements) == len(measurements)
    for supplied, bound in zip(measurements, binding.measurements):
        assert supplied == original
        assert supplied.shots == original_shots
        assert bound is not supplied
        assert bound.shots == 16
    assert reused.shots == original_shots
    assert reused.measurements is not None
    assert reused.measurements[0].shots == original_shots


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("mode", list(BindingMode))
def test_interleaved_repeated_circuits_keep_input_order(mode: BindingMode):
    first = QCircuit([X(0)])
    second = QCircuit(1)
    binding = CircuitBinding(
        [first, second, first],
        measurements=ExpectationMeasure(Observable(pZ)),
        mode=mode,
    )

    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert len(result.results) == 3
    assert [r.expectation_values for r in result.results] == pytest.approx([-1, 1, -1])


@pytest.mark.provider("qiskit")
def test_interleaved_parameterized_circuits_keep_input_order():
    theta = Symbol("theta")
    first = QCircuit([Ry(theta, 0)])
    second = QCircuit([Rz(theta, 0)])
    binding = CircuitBinding(
        [first, second, first],
        values=[{theta: 0.0}, {theta: 0.0}, {theta: np.pi}],
        measurements=[ExpectationMeasure(Observable(pZ)) for _ in range(3)],
        mode=BindingMode.ZIP,
    )

    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert len(result.results) == 3
    assert [r.expectation_values for r in result.results] == pytest.approx([1, 1, -1])
