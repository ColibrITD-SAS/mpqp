import pytest
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.core.instruction.measurement.pauli_string import I, X, Y, Z
from mpqp.execution.devices import (
    AWSDevice,
    AvailableDevice,
    GOOGLEDevice,
    IBMDevice,
)
from mpqp.execution.runner import run
from mpqp.tools.circuit import random_circuit
from mpqp.tools.pauli_grouping import run_optimized_multi_observables


@pytest.mark.parametrize(
    "device",
    [
        (IBMDevice.AER_SIMULATOR),
        (GOOGLEDevice.CIRQ_LOCAL_SIMULATOR),
        (AWSDevice.BRAKET_LOCAL_SIMULATOR),
    ],
)
def test_expectation_values_devices(device: AvailableDevice):
    circuit = random_circuit(nb_qubits=3)
    string = X @ I @ Z + X @ Z @ Z + I @ Z @ Z - 2 * Y @ Z @ Z + 3 * X @ Y @ Z
    true_result: float = run(circuit + QCircuit([ExpectationMeasure(Observable(string))]), device).expectation_values  # type: ignore
    decomposed_pauli = run_optimized_multi_observables(circuit, string, device)
    assert round(true_result, 6) == round(decomposed_pauli, 6)
