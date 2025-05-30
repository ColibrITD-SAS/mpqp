import pytest
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.core.instruction.measurement.pauli_string import I, X, Y, Z
from mpqp.execution.devices import (
    ATOSDevice,
    AWSDevice,
    AvailableDevice,
    GOOGLEDevice,
    IBMDevice,
)
from mpqp.execution.runner import run
from mpqp.tools.circuit import random_circuit


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
    string = X @ I @ Z + X @ Z @ Z + I @ Z @ Z
    str2 = I @ Z @ Z - 2 * Y @ Z @ Z + 3 * X @ Y @ Z
    str3 = X @ X @ X + X @ I @ X + I @ X @ X
    obs = [Observable(string), Observable(str2), Observable(str3)]
    true_result = run(circuit + QCircuit([ExpectationMeasure(obs)]), device).expectation_values  # type: ignore
    single_exp_values = []
    for observable in obs:
        single_exp_values.append(
            run(
                circuit + QCircuit([ExpectationMeasure(observable)]), device
            ).expectation_values  # type: ignore
        )
    equal = True
    print(single_exp_values)
    assert isinstance(true_result, dict)
    for i in range(len(true_result)):
        equal &= round(true_result[f"observable_{i}"], 6) == round(
            single_exp_values[i], 6
        )
    assert equal
