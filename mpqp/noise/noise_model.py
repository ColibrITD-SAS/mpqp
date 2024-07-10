from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from braket.circuits.noises import Noise as BraketNoise
    from braket.circuits.noises import TwoQubitDepolarizing
    from qat.quops.class_concepts import QuantumChannel as QLMNoise

from typeguard import typechecked

from mpqp.core.instruction.gates import Gate
from mpqp.core.languages import Language
from mpqp.noise.custom_noise import KrausRepresentation


@typechecked
class NoiseModel(ABC):
    """Abstract class used to represent a generic noise model, specifying
    criteria for applying different noise type to a quantum circuit, or some of
    its qubits.

    It allows to specify which qubits (targets) and which gates of the circuit
    will be affected with this noise model. If you don't specify a target, the
    operation will apply to all qubits.

    Args:
        targets: List of qubit indices affected by this noise.
        gates: List of :class:`Gates<mpqp.core.instructions.gates.gate.Gate>`
            affected by this noise.

    Raises:
        ValueError: When target list is empty, or target indices are duplicated
        or negative. When the size of the gate is higher than the number of target qubits.
    """

    def __init__(
        self,
        targets: Optional[list[int]] = None,
        gates: Optional[list[type[Gate]]] = None,
    ):
        if targets is None:
            targets = []
        if len(set(targets)) != len(targets):
            raise ValueError(f"Duplicate indices in targets: {targets}")

        if any(index < 0 for index in targets):
            raise ValueError(f"Target indices must be non-negative, but got: {targets}")

        if gates is not None:
            for gate in gates:
                nb_qubits = gate.nb_qubits
                if isinstance(nb_qubits, property):
                    raise ValueError(
                        "If you want to pass a custom gate class to specify"
                        " the noise target, please add `nb_qubits` to this "
                        "class as a class attribute."
                    )
                if len(targets) != 0 and nb_qubits > len(
                    targets
                ):  # pyright: ignore[reportOperatorIssue]
                    raise ValueError(
                        "Size mismatch between gate and noise: gate size is "
                        f"{nb_qubits} but noise size is {len(targets)}"
                    )

        self.targets = targets
        """List of target qubits that will be affected by this noise model."""
        self.gates = gates if gates is not None else []
        """List of specific gates after which this noise model will be applied."""

    def connections(self) -> set[int]:
        """Returns the indices of the qubits connected to the noise model (affected by the noise).

        Returns:
            Set of qubit indices on which this NoiseModel is connected (applied on).
        """
        return set(self.targets)

    @abstractmethod
    def to_kraus_representation(self) -> KrausRepresentation:
        """3M-TODO: to be implemented"""
        pass

    @abstractmethod
    def to_other_language(self, language: Language) -> BraketNoise | QLMNoise:
        """Transforms this noise model into the corresponding object in the
        language specified in the ``language`` arg.

        In the current version, only Braket and my_QLM are available for conversion.

        Args:
            language: Enum representing the target language.

        Returns:
            The corresponding noise model (or channel) in the target language.
        """
        pass

    # 3M-TODO: implement the possibility of having a parameterized noise
    # @abstractmethod
    # def subs(self):
    #     pass


@typechecked
class Depolarizing(NoiseModel):
    """Class representing the depolarizing noise channel, which maps a state
    onto a linear combination of itself and the maximally mixed state. It can
    applied to a single or multiple qubits, and depends on a single parameter
    (probability or error rate).

    When the number of qubits in the target is higher than the dimension, the
    noise will be applied to all possible combinations of indices of size
    ``dimension``.

    Args:
        prob: Depolarizing error probability or error rate.
        targets: List of qubit indices affected by this noise.
        dimension: Dimension of the depolarizing channel.
        gates: List of :class:`Gates<mpqp.core.instruction.gates.gate.Gate>`
            affected by this noise.

    Raises:
        ValueError: When a wrong dimension (negative) or probability (outside of
            the expected interval) is input.
        ValueError: When the size of the specified gates is not coherent with
            the number of targets or the dimension.

    Examples:
        >>> circuit = QCircuit([H(i) for i in range(3)])
        >>> d1 = Depolarizing(0.32, list(range(circuit.nb_qubits)))
        >>> d2 = Depolarizing(0.01)
        >>> d3 = Depolarizing(0.05, [0, 1], dimension=2)
        >>> d4 = Depolarizing(0.12, [2], gates=[H, Rx, Ry, Rz])
        >>> d5 = Depolarizing(0.05, [0, 1, 2], dimension=2, gates=[CNOT, CZ])
        >>> circuit.add([d1, d2, d3, d4, d5])
        >>> print(circuit)  # doctest: +NORMALIZE_WHITESPACE
             ┌───┐
        q_0: ┤ H ├
             ├───┤
        q_1: ┤ H ├
             ├───┤
        q_2: ┤ H ├
             └───┘
        NoiseModel:
            Depolarizing(0.32, [0, 1, 2])
            Depolarizing(0.01)
            Depolarizing(0.05, [0, 1], 2)
            Depolarizing(0.12, [2], gates=[H, Rx, Ry, Rz])
            Depolarizing(0.05, [0, 1, 2], 2, [CNOT, CZ])

    """

    def __init__(
        self,
        prob: float,
        targets: Optional[list[int]] = None,
        dimension: int = 1,
        gates: Optional[list[type[Gate]]] = None,
    ):
        if dimension <= 0:
            raise ValueError(
                "Dimension of the depolarizing channel must be strictly greater"
                f" than 1, but got {dimension} instead."
            )

        # 3M-TODO: implement the possibility of having a parameterized noise,
        # param: Union[float, Expr]
        prob_upper_bound = 1 if dimension == 1 else 1 + 1 / (dimension**2 - 1)
        if not (0 <= prob <= prob_upper_bound):  # pyright: ignore[reportOperatorIssue]
            print(dimension, prob, prob_upper_bound)
            raise ValueError(
                f"Invalid probability: {prob} but should have been between 0 "
                f"and {prob_upper_bound}."
            )

        if gates is not None:
            if any(
                gate.nb_qubits
                != dimension  # pyright: ignore[reportUnnecessaryComparison]
                for gate in gates
            ):
                raise ValueError(
                    f"Dimension of Depolarizing is {dimension}, but got specified gate(s) of different size."
                )

        if targets and len(targets) < dimension:
            raise ValueError(
                f"Number of target qubits {len(targets)} should be higher than the dimension {dimension}."
            )

        super().__init__(targets, gates)
        self.proba = prob
        """Probability, or error rate, of the depolarizing noise model."""
        self.dimension = dimension
        """Dimension of the depolarizing noise model."""

    def to_kraus_representation(self):
        """3M-TODO"""
        # generate Kraus operators for depolarizing noise
        kraus_operators = [...]  # list of Kraus operators
        return KrausRepresentation(kraus_operators)

    def __repr__(self):
        target = ", " + str(self.targets) if len(self.targets) != 0 else ""
        dimension = (
            f", {'dimension=' if not target else ''}" + str(self.dimension)
            if self.dimension != 1
            else ""
        )
        return (
            f"{type(self).__name__}({self.proba}{target}{dimension}"
            + (
                f", {'gates=' if not target or not dimension else ''}" + str(self.gates)
                if len(self.gates) != 0
                else ""
            )
            + ")"
        )

    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> BraketNoise | TwoQubitDepolarizing | QLMNoise:
        """See documentation of this method in abstract mother class :class:`NoiseModel`.

        Args:
            language: Enum representing the target language.

        Examples:
            >>> braket_depo = Depolarizing(0.3, [0,1], dimension=1).to_other_language(Language.BRAKET)
            >>> braket_depo
            Depolarizing('probability': 0.3, 'qubit_count': 1)
            >>> type(braket_depo)
            <class 'braket.circuits.noises.Depolarizing'>
            >>> qlm_depo = Depolarizing(0.3, [0,1], dimension=1).to_other_language(Language.MY_QLM)
            >>> print(qlm_depo)  # doctest: +NORMALIZE_WHITESPACE
            Depolarizing channel, p = 0.3:
            [[0.83666003 0.        ]
             [0.         0.83666003]]
            [[0.        +0.j 0.31622777+0.j]
             [0.31622777+0.j 0.        +0.j]]
            [[0.+0.j         0.-0.31622777j]
             [0.+0.31622777j 0.+0.j        ]]
            [[ 0.31622777+0.j  0.        +0.j]
             [ 0.        +0.j -0.31622777+0.j]]
            >>> type(qlm_depo)
            <class 'qat.quops.quantum_channels.QuantumChannelKraus'>

        """
        if language == Language.BRAKET:
            if self.dimension > 2:
                raise NotImplementedError(
                    f"Depolarizing channel is not implemented in Braket for more than 2 qubits."
                )
            elif self.dimension == 2:
                from braket.circuits.noises import TwoQubitDepolarizing

                return TwoQubitDepolarizing(probability=self.proba)
            else:
                from braket.circuits.noises import Depolarizing as BraketDepolarizing

                return BraketDepolarizing(probability=self.proba)

        elif language == Language.MY_QLM:
            if self.dimension > 2:
                raise NotImplementedError(
                    f"Depolarizing channel is not implemented in the QLM for more than 2 qubits."
                )
            elif self.dimension == 2 and len(self.gates) == 0:
                raise ValueError(
                    "Depolarizing channel of dimension 2 for idle qubits is not supported by the QLM."
                )

            from qat.quops import (
                make_depolarizing_channel,  # pyright: ignore[reportAttributeAccessIssue]
            )

            return make_depolarizing_channel(
                prob=self.proba,
                nqbits=self.dimension,
                method_2q="equal_probs",
                depol_type="pauli",
            )
        else:
            raise NotImplementedError(
                f"Conversion of Depolarizing noise for language {language.name} is not supported"
            )


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
    """Class representing the amplitude damping noise channel, which can model both
    the standard and generalized amplitude damping processes. It can be applied
    to a single qubit and depends on two parameters: the decay rate `gamma` and the
    probability of excitation `prob`.

    Args:
        gamma: Decaying rate of the amplitude damping noise channel.
        prob: Probability of excitation in the generalized amplitude damping noise channel
            When set to 1, indicating standard amplitude damping.
        targets: List of qubit indices affected by this noise.
        gates: List of :class:`Gates<mpqp.core.instruction.gates.gate.Gate>`
            affected by this noise.

    Raises:
        ValueError: When the gamma or prob parameters are outside of the expected interval [0, 1].
        ValueError: When no target qubits are specified.

    Examples:
        >>> circuit = QCircuit([H(i) for i in range(3)])
        >>> ad1 = AmplitudeDamping(0.2, 0, [0])
        >>> ad2 = AmplitudeDamping(0.4, 0.1, [1, 2])
        >>> circuit.add([ad1, ad2])
        >>> print(circuit)
             ┌───┐
        q_0: ┤ H ├
             ├───┤
        q_1: ┤ H ├
             ├───┤
        q_2: ┤ H ├
             └───┘
        NoiseModel:
            AmplitudeDamping(gamma=0.2, prob=0, targets=[0])
            AmplitudeDamping(gamma=0.4, prob=0.1, targets=[1, 2])

    """

    def __init__(
        self,
        gamma: float,
        prob: float = 1,
        targets: Optional[list[int]] = None,
        gates: Optional[list[type[Gate]]] = None,
    ):
        if not (0 <= gamma <= 1):
            raise ValueError(
                f"Invalid decaying rate: {gamma}. It should be between 0 and 1."
            )

        if not (0 <= prob <= 1):
            raise ValueError(
                f"Invalid excitation probability: {prob}. It should be between 0 and 1."
            )

        nb_targets = len(targets) if targets else 0
        if nb_targets < 1:
            raise ValueError("Number of target qubits should be at least 1.")

        super().__init__(targets, gates)

        self.gamma = gamma
        """Decaying rate, of the amplitude damping noise channel."""
        self.prob = prob
        """Excitation probability, of the generalized amplitude damping noise channel."""

    def __repr__(self):
        return (
            f"{type(self).__name__}(gamma={self.gamma}, prob={self.prob}, targets={self.targets}"
            + (", gates=" + str(self.gates) if self.gates else "")
            + ")"
        )

    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> BraketNoise | QLMNoise:
        """See documentation of this method in abstract mother class :class:`NoiseModel`.

        Args:
            language: Enum representing the target language.

        Examples:
            >>> braket_ad = AmplitudeDamping(0.2, 0, [1]).to_other_language(Language.BRAKET)
            >>> braket_ad
            AmplitudeDamping('gamma': 0.2, 'qubit_count': 1)
            >>> type(braket_ad)
            <class 'braket.circuits.noises.AmplitudeDamping'>
            >>> braket_gad = AmplitudeDamping(0.15, 0.2, [0]).to_other_language(Language.BRAKET)
            >>> braket_gad
            GeneralizedAmplitudeDamping('gamma': 0.15, 'probability': 0.2, 'qubit_count': 1)
            >>> type(braket_gad)
            <class 'braket.circuits.noises.GeneralizedAmplitudeDamping'>

        """
        if language == Language.BRAKET:
            if self.prob == 1:
                from braket.circuits.noises import (
                    AmplitudeDamping as BraketAmplitudeDamping,
                )

                return BraketAmplitudeDamping(self.gamma)
            else:
                from braket.circuits.noises import GeneralizedAmplitudeDamping

                return GeneralizedAmplitudeDamping(self.gamma, float(self.prob))

        # TODO: MY_QLM implmentation

        else:
            raise NotImplementedError(
                f"Conversion of Amplitude Damping noise for language {language} is not supported."
            )

    def to_kraus_representation(self) -> KrausRepresentation: ...


class PhaseDamping(NoiseModel):
    """3M-TODO"""

    def to_kraus_representation(self) -> KrausRepresentation: ...
