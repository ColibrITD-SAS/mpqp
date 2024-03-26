from mpqp import QCircuit

# Wish list
# Apply noise model to the whole circuit --> DepolarizingNoise on whole circuit
# Apply noise to specific qubits --> DepolarizingNoise on qubit 0
# Apply noise to specific gate, all qubits --> Noise on Hadamard on all qubits
# Apply noise to specific gate, specific qubits --> Noise on Hadamard(0)
# Predefined noise models : with focus on DepolarizingNoise, GateNoise
# Wish providers ? The four of them
#

################# USAGE PART #################
from mpqp.core.instruction.gates import Gate

circuit = QCircuit()

# depolarizing noise on the whole circuit
circuit.add(Depolarizing(0.03))

# depolarizing noise on the two first qubits
circuit.add(Depolarizing(0.03, [0,1]))

# adding several noises at the same time to the circuit
circuit.add([Depolarizing(0.02, 0), BitFlip(0.1, [2, 3])])

# adding noise to specific gates, all qubits
circuit.add(GateNoise(([H, X, Y, Z], Depolarizing(0.03))))

# adding several noise to different gates, all qubits
circuit.add(GateNoise(
    [
        ([H, X, Y, Z], Depolarizing(0.03)),
        ([CNOT, CZ], Bitflip(0.5))
    ])
)

# adding noise to specific gates, specific qubits
circuit.add(GateNoise(([H], Depolarizing(0.03, [0,1, 2]))))




################# CONCEPTION PART #################


class NoiseModel:
    def __init__(self, p_error: float, target: int | list[int] = None):
        # if target is None, it has to be set from circuit.add() as targets = list(range(circuit.nb_qubits))
        self.target = target
        pass


class Depolarizing(NoiseModel):

    pass


class BitFlip(NoiseModel):
    pass


class GateNoise():
    def __init__(self, gate_noise_spec: tuple[list[Gate], NoiseModel] | list[tuple[list[Gate], NoiseModel]]):
        pass


class CustomNoise(NoiseModel):
    """# 3M-TODO : implement and comment"""
    pass


class KrausOperator():
    """# 3M-TODO : implement and comment"""
    pass


class KrausRepresentation():
    """# 3M-TODO : implement and comment"""

    def __init__(self, k_ops: list[KrausOperator]):
        self.pp = 1
