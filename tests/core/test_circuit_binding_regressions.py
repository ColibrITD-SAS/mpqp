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
    pI,
    pX,
    pZ,
    run,
)
from mpqp.core.circuitbinding import BindingMode, CircuitBinding
from mpqp.execution.providers.providers_params import QiskitParams
from mpqp.execution.result import BatchResult


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
        assert [m.observables[0].nb_qubits for _, _, m in executions] == [3, 4]
    assert measurements == original
    assert all(not c.measurements for c in circuits)


@pytest.mark.parametrize("mode", list(BindingMode))
@pytest.mark.parametrize("nested", [False, True])
def test_shared_measure_is_embedded_per_resolved_circuit(mode, nested):
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
            assert adjusted.targets == list(range(circuit.nb_qubits))
            np.testing.assert_allclose(
                adjusted.observables[0].matrix, Observable(expected).matrix
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


@pytest.mark.provider("qiskit")
def test_mixed_register_ibm_pubs_validate_without_submission():
    from mpqp.execution.providers.ibm import _normalize_estimator_pubs
    from qiskit.primitives.containers.estimator_pub import EstimatorPub

    binding = CircuitBinding(
        [QCircuit([X(0)], nb_qubits=3), QCircuit([X(0)], nb_qubits=4)],
        measurements=[
            ExpectationMeasure(Observable(pI @ pI @ pZ)),
            ExpectationMeasure(Observable(pI @ pI @ pI @ pZ)),
        ],
        mode=BindingMode.ZIP,
    )
    translated = binding.to_other_device(IBMDevice.AER_SIMULATOR)
    pubs = _normalize_estimator_pubs([pub for pub, _ in translated])
    assert [pub[0].num_qubits for pub in pubs] == [3, 4]
    for pub in pubs:
        EstimatorPub.coerce(pub)  # Validate shapes locally; never submit a job.


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
    assert all(
        call.kwargs["provider_params"] is params for call in execute.call_args_list
    )


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
def test_grouping_keeps_circuit_and_execution_order():
    first = QCircuit([X(0)])
    second = QCircuit(1)
    binding = CircuitBinding(
        [first, second, first],
        measurements=[
            ExpectationMeasure(Observable(pZ)),
            ExpectationMeasure(Observable(pZ)),
            ExpectationMeasure(Observable(pX)),
        ],
        mode=BindingMode.ZIP,
    )
    pubs = binding.to_other_device(IBMDevice.AER_SIMULATOR)
    assert len(pubs) == 2
    assert pubs[0][0][0] is first.transpiled_circuit[IBMDevice.AER_SIMULATOR]
    assert pubs[1][0][0] is second.transpiled_circuit[IBMDevice.AER_SIMULATOR]
    assert [len(contexts) for _, contexts in pubs] == [2, 1]
    result = run(binding, IBMDevice.AER_SIMULATOR)
    assert [r.expectation_values for r in result.results] == pytest.approx([-1, 0, 1])
