import numpy as np

from mpqp import QCircuit
from mpqp.execution import run
from mpqp.execution.devices import ATOSDevice
from mpqp.gates import CNOT, H, Rx, Ry, Rz

circuit = QCircuit(5)

circuit.add(H(0))
circuit.add(H(1))
circuit.add(Rz(np.pi / 3, 1))
circuit.add(H(2))
circuit.add(Rx(np.pi / 3, 2))
circuit.add(H(3))
circuit.add(H(4))
circuit.add(CNOT(0, 1))
circuit.add(CNOT(1, 2))
circuit.add(CNOT(2, 3))
circuit.add(CNOT(3, 4))
circuit.add(Ry(np.pi / 3, 4))

print(run(circuit, ATOSDevice.MYQLM_PYLINALG).state_vector)


circuit2 = QCircuit(5)

circuit2.add(H(0))
circuit2.add(H(1))
circuit2.add(Rz(np.pi / 3, 1))
circuit2.add(H(2))
circuit2.add(Rx(np.pi / 3, 2))
circuit2.add(H(3))
circuit2.add(H(4))
circuit2.add(CNOT(0, 1))
circuit2.add(CNOT(2, 3))
circuit2.add(CNOT(1, 2))
circuit2.add(CNOT(3, 4))
circuit2.add(Ry(np.pi / 3, 4))

print(run(circuit2, ATOSDevice.MYQLM_PYLINALG).state_vector)
