"""Example 2: Expectation value of an observable"""

import numpy as np
from mpqp.gates import H, Rx
from mpqp import QCircuit
from mpqp.execution import run
from mpqp.measures import ExpectationMeasure, Observable
from mpqp.execution.devices import ATOSDevice, IBMDevice, AWSDevice
from mpqp.core.instruction.measurement.pauli_string import I, Z

obs = Observable(
    np.array(
        [
            [0.63, 0.5, 1, 1],
            [0.5, 0.82, 1, 1],
            [1, 1, 1, 0.33],
            [1, 1, 0.33, 0.3],
        ],
        dtype=float,
    )
)

obs2 = Observable(1 * I @ Z + 1 * I @ I)

# Observable can be constructed from a Pauli string or a matrix
print("Observable2:")
print(obs2.pauli_string)
print(obs2.matrix)

# Declaration of the circuit with the right size
circuit = QCircuit(2, label="Observable test")
# Constructing the circuit by adding gates and measurements
circuit.add(H(0))
circuit.add(Rx(1.76, 1))
circuit.add(ExpectationMeasure([0, 1], observable=obs, shots=1000))

print(circuit)

# Running the computation on myQLM and on Aer simulator, then retrieving the results
results = run(
    circuit,
    [
        ATOSDevice.MYQLM_PYLINALG,
        IBMDevice.AER_SIMULATOR,
        ATOSDevice.MYQLM_CLINALG,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
    ],
)
print(results)


# Declaration of the circuit with the right size
circuit = QCircuit(2, label="Observable test 2")
# Constructing the circuit by adding gates and measurements
circuit.add(H(0))
circuit.add(H(1))
circuit.add(ExpectationMeasure([0, 1], observable=obs2, shots=0))

print(circuit)

# Running the computation on myQLM and on Aer simulator, then retrieving the results
results = run(
    circuit,
    [
        ATOSDevice.MYQLM_PYLINALG,
        ATOSDevice.MYQLM_CLINALG,
        IBMDevice.AER_SIMULATOR,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
    ],
)
print(results)
