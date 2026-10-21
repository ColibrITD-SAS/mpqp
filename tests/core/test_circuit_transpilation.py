"""Regression tests for reusable, device-specific circuit preparation."""

import numpy as np
import pytest
from sympy import Symbol

from mpqp import QCircuit
from mpqp.execution.devices import IBMDevice
from mpqp.execution.runner import run
from mpqp.gates import Ry, X


@pytest.mark.provider("qiskit")
def test_prepared_circuit_reuses_template_for_multiple_values():
    theta = Symbol("theta")
    circuit = QCircuit([Ry(theta, 0)])
    device = IBMDevice.AER_SIMULATOR_STATEVECTOR
    template = circuit.transpiled_for_device(device)
    assert circuit.transpiled_for_device(device) is template

    for value in (0.5, 1.0, 0.5):
        result = run(circuit, device, values={theta: value})
        assert abs(result.state_vector.vector[1]) ** 2 == pytest.approx(
            np.sin(value / 2) ** 2
        )
        assert circuit.transpiled_circuit[device] is template
        assert len(template.parameters) == 1


@pytest.mark.provider("qiskit")
def test_prepared_circuit_cache_is_per_device():
    circuit = QCircuit([X(0)])
    devices = (
        IBMDevice.AER_SIMULATOR_STATEVECTOR,
        IBMDevice.AER_SIMULATOR,
    )
    templates = [circuit.transpiled_for_device(device) for device in devices]
    assert set(circuit.transpiled_circuit) == set(devices)
    assert templates[0] is not templates[1]
    circuit.add(X(0))
    assert circuit.transpiled_circuit == {}


@pytest.mark.provider("qiskit")
def test_shallow_copy_has_independent_preparation_cache():
    circuit = QCircuit([X(0)])
    device = IBMDevice.AER_SIMULATOR_STATEVECTOR
    template = circuit.transpiled_for_device(device)
    clone = circuit.without_measurements(deep_copy=False)
    clone.add(X(0))
    assert clone.transpiled_circuit == {}
    assert circuit.transpiled_circuit[device] is template
