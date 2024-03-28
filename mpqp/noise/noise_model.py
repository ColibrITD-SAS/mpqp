from abc import ABC, abstractmethod
from typing import Union

from sympy import Expr

from mpqp.core.instruction.gates import Gate
from mpqp.noise.custom_noise import KrausRepresentation


class NoiseModel(ABC):
    def __init__(self, targets: list[int], gates: list[Gate] = None):
        # if target is None, it has to be set from circuit.add() as targets = list(range(circuit.nb_qubits))
        self.targets = targets
        self.gates = gates
        pass

    def connections(self) -> set[int]:
        return set(self.targets)

    @abstractmethod
    def to_kraus_representation(self) -> KrausRepresentation:
        pass


class Depolarizing(NoiseModel):
    # TODO: NOISE - define the __init__ --> need to know what are all the parameters for defining Depolarizing Noise ?
    #  Error on one qubit, two qubits ?

    # TODO : NOISE - Is this a single qubit Depolarizing that we can put on several qubit ? Or it also captures the Two-qubit
    #  Depolarizing channel ? --> look at d parameter ?

    def __init__(
        self,
        proba: Union[float, Expr],
        targets: list[int],
        dimension: int = 1,
        gates: list[Gate] = None,
    ):
        # TODO check that the parameter verifies 0 <= proba <= 1+1/(d**2-1), with d dimension of the depolarizing channel
        super().__init__()
        self.proba

    # TODO

    def to_kraus_representation(self):
        # TODO
        pass


class BitFlip(NoiseModel):
    """3M-TODO"""

    pass


class Pauli(NoiseModel):
    """3M-TODO"""

    pass


class Dephasing(NoiseModel):
    """3M-TODO"""

    pass


class PhaseFlip(NoiseModel):
    """3M-TODO"""

    pass


class AmplitudeDamping(NoiseModel):
    """3M-TODO"""

    pass


class PhaseDamping(NoiseModel):
    """3M-TODO"""

    pass
