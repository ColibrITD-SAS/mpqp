from copy import deepcopy
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest
from sympy import Symbol

from mpqp import (
    BasisMeasure,
    ExpectationMeasure,
    IBMDevice,
    Observable,
    QCircuit,
    Ry,
    Rz,
    X,
    pX,
    pZ,
    run,
)
from mpqp.core.binding import BindingMode, CircuitBinding
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.execution.job import JobType
from mpqp.execution.providers.providers_params import QiskitParams
from mpqp.execution.result import BatchResult


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
def test_grouping_preserves_execution_associations():
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
    pubs = binding.to_other_device(IBMDevice.AER_SIMULATOR)
    assert len(pubs) == 2
    assert pubs[0][0][0] is first.transpiled_circuit  # pyright: ignore
    assert pubs[1][0][0] is second.transpiled_circuit  # pyright: ignore
    assert [len(contexts) for _, contexts in pubs] == [2, 1]
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert len(result.results) == 3
    # Associate values through their execution metadata, independently of PUB order.
    actual = {}
    for execution in result.results:
        assert execution.job.measure is not None
        actual[execution.job.measure.label] = execution.expectation_values
    assert actual == pytest.approx({"first_Z": -1, "second_Z": 1, "first_X": 0})


@pytest.mark.parametrize(
    "shots, expected", [(0, JobType.STATE_VECTOR), (16, JobType.SAMPLE)]
)
def test_binding_basis_measure_job_type(shots: int, expected: JobType):
    binding = CircuitBinding(QCircuit([X(0)]), measurements=BasisMeasure(shots=shots))
    assert binding.job_type == expected


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "binding",
    [
        CircuitBinding(QCircuit([X(0), BasisMeasure(shots=0)])),
        CircuitBinding(QCircuit([X(0)]), measurements=BasisMeasure(shots=0)),
    ],
)
def test_binding_zero_shots_returns_state_vector(binding: CircuitBinding):
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert len(result.results) == 1
    execution = result.results[0]
    assert execution.job.job_type == JobType.STATE_VECTOR
    np.testing.assert_allclose(execution.amplitudes, [0, 1], atol=1e-12)
    assert execution.shots == 0


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


@pytest.mark.parametrize(
    "measure", [ExpectationMeasure(Observable(pZ)), BasisMeasure()]
)
def test_binding_shot_override_preserves_input_measurement(measure: Measure):
    original_shots = measure.shots
    binding = CircuitBinding(QCircuit(1), measurements=measure, shots=16)
    reused = CircuitBinding(QCircuit(1), measurements=measure)

    assert measure.shots == original_shots
    assert binding.shots == 16
    assert binding.measurements is not None
    assert binding.measurements[0].shots == 16
    assert reused.shots == original_shots
    assert reused.measurements is not None
    assert reused.measurements[0].shots == original_shots
