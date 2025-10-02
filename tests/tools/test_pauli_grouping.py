import pytest
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.core.instruction.measurement.pauli_string import pI, pX, pY, pZ
from mpqp.execution.devices import (
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
    string = pX @ pI @ pZ + pX @ pZ @ pZ + pI @ pZ @ pZ
    str2 = pI @ pZ @ pZ - 2 * pY @ pZ @ pZ + 3 * pX @ pY @ pZ
    str3 = pX @ pX @ pX + pX @ pI @ pX + pI @ pX @ pX
    obs = [Observable(string), Observable(str2), Observable(str3)]
    true_result = run(
        circuit + QCircuit([ExpectationMeasure(obs)]), device
    ).expectation_values
    single_exp_values = []
    for observable in obs:
        single_exp_values.append(
            run(
                circuit + QCircuit([ExpectationMeasure(observable)]),
                device,
            ).expectation_values
        )
    assert isinstance(true_result, dict)
    assert all(
        round(true_result[f"observable_{i}"], 6) == round(single_exp_values[i], 6)
        for i in range(len(true_result))
    )
