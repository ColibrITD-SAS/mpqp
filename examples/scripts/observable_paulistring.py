"""Example 2: Expectation value of an observable"""

import numpy as np
from mpqp.gates import H, Rx
from mpqp import QCircuit
from mpqp.execution import run
from mpqp.measures import ExpectationMeasure, Observable
from mpqp.execution.devices import AWSDevice

from mpqp.core.instruction.measurement.pauli_string import I, Z

obs = Observable(
    np.array(
        [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
)

ps_obs = Observable(1 * I @ Z + 1 * I @ I)

mat = ps_obs.matrix

print(obs)
print(ps_obs)

# Declaration of the circuit with the right size
circuit = QCircuit(2, label="Observable test")
# Constructing the circuit by adding gates and measurements
circuit.add(H(0))
circuit.add(Rx(1.76, 1))
circuit.add(ExpectationMeasure([0, 1], observable=ps_obs, shots=0))

print(circuit)

result = run(circuit, [AWSDevice.BRAKET_LOCAL_SIMULATOR])
print(result)

circuit2 = circuit.without_measurements()
circuit2.add(ExpectationMeasure([0, 1], observable=obs, shots=0))
result = run(circuit2, [AWSDevice.BRAKET_LOCAL_SIMULATOR])
print(result)

