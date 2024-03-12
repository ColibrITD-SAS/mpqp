"""Example 1: Building a Bell state"""

from mpqp.gates import H, CNOT
from mpqp import QCircuit
from mpqp.measures import BasisMeasure
from mpqp.execution.devices import IBMDevice, AWSDevice
from mpqp.execution import run
from mpqp.tools.visualization import plot_results_sample_mode
import matplotlib.pyplot as plt

# Declaration of the circuit with the right size
circuit = QCircuit(2, label="Bell pair")
# Constructing the circuit by adding gates and measurements
circuit.add(H(0))
circuit.add(CNOT(0, 1))
circuit.add(BasisMeasure([0, 1], shots=1000))
results = run(circuit, [IBMDevice.AER_SIMULATOR, AWSDevice.BRAKET_LOCAL_SIMULATOR])
print(results)

plot_results_sample_mode(results)
circuit.display()
plt.show()
