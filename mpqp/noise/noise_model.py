from abc import ABC, abstractmethod
from typing import Union

from sympy import Expr

from mpqp.core.instruction.gates import Gate
from mpqp.noise.custom_noise import KrausRepresentation


class NoiseModel(ABC):
    # TODO: to define the 'docstring'
    def __init__(self, targets: list[int], gates: list[Gate] = None):
        self.targets = targets
        self.gates = gates if gates is not None else []

    def connections(self) -> set[int]:
        return set(self.targets)

    @abstractmethod
    def to_kraus_representation(self) -> KrausRepresentation:
        pass


class Depolarizing(NoiseModel):
    # TODO: to define the 'docstring'
    def __init__(
        self,
        proba: Union[float, Expr],
        targets: list[int],
        dimension: int = 1,
        gates: list[Gate] = None,
    ):

        super().__init__(targets, gates)

        if not (0 <= proba <= 1 + 1 / (dimension**2 - 1)):
            raise ValueError(
                f"Invalid probability; {proba} must be between 0 and {1 + 1 / (dimension**2 - 1)}"
            )

        self.proba = proba
        self.dimension = dimension

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
