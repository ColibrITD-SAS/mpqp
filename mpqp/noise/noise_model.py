from abc import ABC, abstractmethod
from typing import Union

from sympy import Expr

from mpqp.core.instruction.gates import Gate
from mpqp.noise.custom_noise import KrausRepresentation


class NoiseModel(ABC):
    # TODO: to define the 'docstring'
    def __init__(self, targets: list[int], gates: list[Gate] = None):
        # TODO: check that targets do not contain negative indices (inspire from what is done in instructions ?)
        self.targets = targets
        self.gates = gates if gates is not None else []

    def connections(self) -> set[int]:
        """Returns the indices of the qubits connected to the noise model (affected by the noise).

        Returns:
            Set of qubit indices on which this NoiseModel is connected (applied on).
        """
        return set(self.targets)

    @abstractmethod
    def to_kraus_representation(self) -> KrausRepresentation:
        """
        TODO: doc
        Returns:

        """
        pass


class Depolarizing(NoiseModel):
    """
    TODO: doc
    blablblabl

    Examples:
        >>>
        >>>
        >>>

    Args:
        proba:
        targets:
        dimension:
        gates:

    Raises:

    """
    def __init__(
        self,
        proba: Union[float, Expr],
        targets: list[int],
        dimension: int = 1,
        gates: list[Gate] = None,
    ):
        proba_upper_bound = 1 + 1/(dimension**2 - 1)
        if not (0 <= proba <= proba_upper_bound):
            raise ValueError(
                f"Invalid probability: {proba} must have been between 0 and {proba_upper_bound}"
            )

        if dimension <= 0:
            raise ValueError(
                f"Dimension of the depolarizing channel must be strictly greater than 1, but got {dimension} instead."
            )

        self.proba = proba
        self.dimension = dimension
        super().__init__(targets, gates)

    def to_kraus_representation(self):
        # TODO
        pass

    def __repr__(self):
        #TODO
        pass


class BitFlip(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation:
        pass

class Pauli(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation:
        pass


class Dephasing(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation:
        pass


class PhaseFlip(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation:
        pass


class AmplitudeDamping(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation:
        pass


class PhaseDamping(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation:
        pass
