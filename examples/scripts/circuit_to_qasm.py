from mpqp.gates import *
from mpqp import QCircuit
from mpqp.measures import BasisMeasure

# Declaration of the circuit with the right size
circuit = QCircuit(4)

# Constructing the circuit by adding gates
circuit.add(T(0))
circuit.add(CNOT(0, 1))
circuit.add(X(0))
circuit.add(H(1))
circuit.add(Z(2))
circuit.add(CZ(2, 1))
circuit.add(SWAP(2, 0))
circuit.add(CNOT(0, 2))
circuit.add(Ry(3.14 / 2, 2))
circuit.add(S(1))
circuit.add(H(3))
circuit.add(CNOT(1, 2))
circuit.add(Rx(3.14, 1))
circuit.add(CNOT(3, 0))
circuit.add(Rz(3.14, 0))
# Add measurement
circuit.add(BasisMeasure([0, 1, 2, 3], shots=2000))

print(circuit.to_qasm2())
print(circuit.to_qasm3())


# Declaration of the circuit with the right size
circuit = QCircuit(2)

# Constructing the circuit by adding gates
circuit.add(X(0))
circuit.add(CNOT(0, 1))
# Add measurement
circuit.add(BasisMeasure([1], shots=2000))

print(circuit.to_qasm2())
print(circuit.to_qasm3())
