from mpqp.core.instruction.gates import Gate


class NoiseModel:
    def __init__(self, target: list[int] = None, gates: list[Gate] = None):
        # if target is None, it has to be set from circuit.add() as targets = list(range(circuit.nb_qubits))
        self.target = target
        self.gates = gates
        pass

class Depolarizing(NoiseModel):
    self.e1
    self.e2
    pass


class BitFlip(NoiseModel):
    self.flip_proba
    pass