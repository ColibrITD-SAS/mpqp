from abc import ABC, abstractmethod
from typing import Union

from sympy import Expr

from mpqp.core.instruction.gates import Gate
from mpqp.noise.custom_noise import KrausRepresentation


class NoiseModel(ABC):
    # TODO: to define the 'docstring'
    def __init__(self, targets: list[int], gates: list[Gate] = None):
        if len(set(targets)) != len(targets):
            raise ValueError(f"Duplicate registers in targets: {targets}")
        if any(index < 0 for index in targets):
            raise ValueError(f"Target indices must be non-negative, but got: {targets}")
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
        proba_upper_bound = 1 + 1 / (dimension**2 - 1)
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
        # generate Kraus operators for depolarizing noise
        kraus_operators = [...]  # list of Kraus operators
        return KrausRepresentation(kraus_operators)

    def __repr__(self):
        #TODO
        pass


class BitFlip(NoiseModel):
    """3M-TODO"""

    # def __init__(
    #     self,
    #     proba: Union[float, Expr],
    #     targets: List[int],
    #     dimension: int = 1,
    #     gates: List[Gate] = None):

    #     super().__init__(proba, targets, dimension, gates)

    def to_kraus_representation(self) -> KrausRepresentation:
        # generate Kraus operators for bit flip noise
        # kraus_operators = [
        #     np.sqrt(1 - self.proba) * np.array([[1, 0], [0, 1]]),  # Identity
        #     np.sqrt(self.proba) * np.array([[0, 1], [1, 0]])      # Bit flip
        # ]
        # return KrausRepresentation(kraus_operators)
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
