from mpqp import QCircuit

# Wish list
# Apply noise model to the whole circuit --> DepolarizingNoise on whole circuit
# Apply noise to specific qubits --> DepolarizingNoise on qubit 0
# Apply noise to specific gate, all qubits --> Noise on Hadamard on all qubits
# Apply noise to specific gate, specific qubits --> Noise on Hadamard(0)
# Predefined noise models : with focus on DepolarizingNoise
# Wish providers ? The four of them
#

################# USAGE PART #################

circuit = QCircuit()

circuit.add() # calls circuit._add_noise()


################# CONCEPTION PART #################


class NoiseModel:
    def __init__(self, p_error: float, target: list[int] = None):
        # if target is None, it has to be set from circuit.add() as targets = list(range(circuit.nb_qubits))
        self.target = target
        pass


class DepolarizingNoise(NoiseModel):

    pass


class BitFlip(NoiseModel):
    pass


class GateNoise(NoiseModel):
    # needs
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
