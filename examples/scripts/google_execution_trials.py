from mpqp.gates import H, Rx, Ry, Rz
from mpqp import QCircuit
from mpqp.measures import BasisMeasure
from mpqp.execution.devices import GOOGLEDevice, IBMDevice, AWSDevice
from mpqp.execution import run
from mpqp.execution.providers_execution.google_execution import circuit_to_processor_cirq_Circuit
from mpqp.qasm import qasm2_to_cirq_Circuit
from mpqp.tools.visualization import plot_results_sample_mode
import matplotlib.pyplot as plt

circuit = QCircuit(3)
circuit.add(H(0))
circuit.add(H(1))
circuit.add(H(2))
circuit.add(Rx(1.76, 1))
circuit.add(Ry(1.76, 1))
circuit.add(Rz(1.987, 0))
circuit.add(BasisMeasure([0, 1, 2], shots=10000))
print(f"MPQP circuit:\n{circuit}\n")

#####################################
cirq_circuit = qasm2_to_cirq_Circuit(circuit.to_qasm2())
print(f"Cirq circuit:\n{cirq_circuit}\n")
#####################################

results = run(circuit, [GOOGLEDevice.CIRQ])
print(results)

#####################################
# @title Choose a processor ("rainbow" or "weber")
processor_id = "rainbow"
shots = circuit.get_measurements()[0].shots

grid_circuit, simulator = circuit_to_processor_cirq_Circuit(processor_id, cirq_circuit)
print(f"circuit for processor {processor_id}:\n{grid_circuit}\n")
#####################################
 

results = run(circuit, [GOOGLEDevice.CIRQ, GOOGLEDevice.PROCESSOR_RAINBOW, GOOGLEDevice.PROCESSOR_WEBER, IBMDevice.AER_SIMULATOR, AWSDevice.BRAKET_LOCAL_SIMULATOR])
print(results)

plot_results_sample_mode(results)
plt.show()