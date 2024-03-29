from mpqp import QCircuit
from mpqp.core.instruction.gates import Gate

# Wish list
# Apply noise model to the whole circuit --> DepolarizingNoise on whole circuit
# Apply noise to specific qubits --> DepolarizingNoise on qubit 0
# Apply noise to specific gate, all qubits --> Noise on Hadamard on all qubits
# Apply noise to specific gate, specific qubits --> Noise on Hadamard(0)
# Predefined noise models : with focus on DepolarizingNoise, GateNoise
# Which providers ? The four of them is possible

################# USAGE PART #################

circuit = QCircuit()

# Remark : the precision of the target list, when None, has to be done at execution

# Remark 2: Hardware model could be Custom noise (combination of several noise models)


# depolarizing noise on the whole circuit
circuit.add(Depolarizing(0.03))

# depolarizing noise on the two first qubits
circuit.add(Depolarizing(0.03, [0, 1]))
circuit.add(Depolarizing(0.03, [0, 1], gates=[H]))

# adding several noises at the same time to the circuit
circuit.add([Depolarizing(0.02, [0]), BitFlip(0.1, [2, 3])])

# adding noise to specific gates, all qubits
circuit.add(Depolarizing(0.03, [0, 1], gates=[H, X ,Y, Z]))

# adding several noise to different gates, specific qubits
circuit.add([Depolarizing(0.02, [0], gates=[H, X, Y, Z]), BitFlip(0.1, [2, 3], gates=[CZ, CNOT])])






