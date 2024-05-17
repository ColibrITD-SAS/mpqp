"""Demonstration MPQP"""

import numpy as np

from mpqp import QCircuit
from mpqp.execution import run
from mpqp.execution.devices import ATOSDevice, AWSDevice, GOOGLEDevice, IBMDevice
from mpqp.gates import *
from mpqp.measures import BasisMeasure

# Constructing the circuit
meas = BasisMeasure(list(range(3)), shots=2000)
c = QCircuit([T(0), CNOT(0, 1), Ry(np.pi / 2, 2), S(1), CZ(2, 1), SWAP(2, 0), meas])
print(c)

# Run the circuit on a selected device
results = run(
    c,
    [
        IBMDevice.AER_SIMULATOR,
        ATOSDevice.MYQLM_PYLINALG,
        ATOSDevice.MYQLM_CLINALG,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
    ],
)

# Print the results
print(results)

# Display the circuit
results.plot()

c = QCircuit([T(0), CNOT(0, 1), Ry(np.pi / 2, 2), S(1), CZ(2, 1), SWAP(2, 0)])
res = run(
    c,
    [
        IBMDevice.AER_SIMULATOR_STATEVECTOR,
        ATOSDevice.MYQLM_PYLINALG,
        ATOSDevice.MYQLM_CLINALG,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
    ],
)
# Print the results
print(res)
