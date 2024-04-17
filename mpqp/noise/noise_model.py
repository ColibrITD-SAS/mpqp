from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

from braket.circuits.noises import Depolarizing as BraketDepolarizing
from braket.circuits.noises import Noise as BraketNoise
from qiskit_aer.noise import NoiseModel as QiskitNoise
from sympy import Expr

from mpqp.core.instruction.gates import Gate
from mpqp.core.languages import Language
from mpqp.noise.custom_noise import KrausRepresentation


class NoiseModel(ABC):
    """Abstract class used to represent a generic noise model, specifying
    criteria for applying different noise type to a quantum circuit, or some of
    its qubits.

    It allows to specify which qubits (targets) and which gates of the circuit
    will be affected with this noise model.

    Args:
        targets: List of qubit indices affected by this noise.
        gates: List of :class:`Gates<mpqp.core.instructions.gates.gate.Gate>`
            affected by this noise.

    Raises:
        ValueError: When target list is empty, or target indices are duplicated
            or negative.
    """

    def __init__(
        self, targets: list[int], gates: Optional[list[Gate | type[Gate]]] = None
    ):
        if len(targets) == 0:
            raise ValueError("Expected non-empty target list")

        if len(set(targets)) != len(targets):
            raise ValueError(f"Duplicate indices in targets: {targets}")

        if any(index < 0 for index in targets):
            raise ValueError(f"Target indices must be non-negative, but got: {targets}")

        if gates is not None:
            for gate in gates:
                nb_qubits = gate.nb_qubits
                if isinstance(nb_qubits, property):
                    # TODO: set class attribute for all native gates
                    raise ValueError(
                        "If you want to pass a custom gate class to specify"
                        " the noise target, please add `nb_qubits` to this "
                        "class as a class attribute."
                    )
                if nb_qubits > len(targets):  # pyright: ignore[reportOperatorIssue]
                    raise ValueError(
                        "Size mismatch between gate and noise: gate size is "
                        f"{gate.nb_qubits} but noise size is {len(targets)}"
                    )

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

    @abstractmethod
    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> BraketNoise | QiskitNoise:
        """
        TODO: doc
        Returns:

        """
        pass

    # @abstractmethod
    # def subs(self):
    #     #TODO: implement the possibilty of having a parameterized noise
    #     pass


class Depolarizing(NoiseModel):
    """Class representing the depolarizing noise channel, which maps a state onto a linear combination of itself and
    the maximally mixed state. It can applied to a single or multiple qubits, and depends on a single parameter
    (probability or error rate).

    When the number of qubits in the target is higher than the dimension, the noise will be applied to all possible
    combinations of indices of size ``dimension``.

    Examples:
        >>> circuit.add(Depolarizing(0.32, list(range(circuit.nb_qubits)))
        >>> circuit.add(Depolarizing(0.05, [0, 1], dimension=2)
        >>> circuit.add(Depolarizing(0.05, [0, 1, 2], dimension=2)
        >>> circuit.add(Depolarizing(0.12, [2], gates=[H, Rx, Ry, Rz])

    Args:
        prob: Depolarizing error probability or error rate.
        targets: List of qubit indices affected by this noise.
        dimension: Dimension of the depolarizing channel.
        gates: List of :class:`Gates<mpqp.core.instructions.gates.gate.Gate>` affected by this noise.

    Raises:
        ValueError: when a wrong dimension (negative) or probability (outside of the expected interval) is input.
    """

    def __init__(
        self,
        prob: Union[float, Expr],
        targets: list[int],
        dimension: int = 1,
        gates: Optional[list[Gate | type[Gate]]] = None,
    ):
        prob_upper_bound = 1 if dimension == 1 else 1 + 1 / (dimension**2 - 1)
        if not (0 <= prob <= prob_upper_bound):  # pyright: ignore[reportOperatorIssue]
            raise ValueError(
                f"Invalid probability: {prob} must have been between 0 and {prob_upper_bound}"
            )

        if dimension <= 0:
            raise ValueError(
                f"Dimension of the depolarizing channel must be strictly greater than 1, but got {dimension} instead."
            )

        nb_targets = len(targets)
        if nb_targets < dimension:
            raise ValueError(
                f"Number of target qubits {nb_targets} should be higher than the dimension {dimension}. "
            )

        super().__init__(targets, gates)
        self.proba = prob
        self.dimension = dimension

    def to_kraus_representation(self):
        # TODO
        # generate Kraus operators for depolarizing noise
        kraus_operators = [...]  # list of Kraus operators
        return KrausRepresentation(kraus_operators)

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.proba}, {self.targets}, {self.dimension}"
            + (", " + str(self.gates) if self.gates else "")
            + ")"
        )

    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> BraketNoise | QiskitNoise:
        if language == Language.BRAKET:
            return BraketDepolarizing(probability=self.proba)
        else:
            # TODO: add other providers
            raise NotImplementedError


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
        ...


class Pauli(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation: ...


class Dephasing(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation: ...


class PhaseFlip(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation: ...


class AmplitudeDamping(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation: ...


class PhaseDamping(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation: ...
