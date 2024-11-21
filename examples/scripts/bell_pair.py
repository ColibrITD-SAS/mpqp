"""Example 1: Building a Bell state"""

import matplotlib.pyplot as plt

from mpqp import QCircuit
from mpqp.execution import run
from mpqp.execution.devices import AWSDevice, IBMDevice
from mpqp.gates import CNOT, H
from mpqp.measures import BasisMeasure

# Declaration of the circuit with the right size
circuit = QCircuit(2, label="Bell pair")
# Constructing the circuit by adding gates and measurements
circuit.add(H(0))
circuit.add(CNOT(0, 1))
circuit.add(BasisMeasure(shots=1000))
results = run(circuit, [IBMDevice.AER_SIMULATOR, AWSDevice.BRAKET_LOCAL_SIMULATOR])
print(results)

results.plot(show=False)
circuit.display()
plt.show()
