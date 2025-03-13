from __future__ import annotations

import inspect
import sys
from abc import ABC, abstractmethod
from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import numpy.typing as npt

from mpqp.measures import I, X, Y, Z
from mpqp.tools.generics import T

if TYPE_CHECKING:
    from braket.circuits.noises import Noise as BraketNoise
    from braket.circuits.noises import TwoQubitDepolarizing
    from qat.quops.class_concepts import QuantumChannel as QLMNoise
    from qiskit_aer.noise.errors.quantum_error import QuantumError

from typeguard import typechecked

from mpqp.core.instruction.gates.native_gates import NativeGate
from mpqp.core.languages import Language
from mpqp.tools.generics import T


def _plural_marker(items: Sequence[T]):
    """Returns the stringified version of a group of items, with a plural
    marker for the previous word if needed.

    Args:
        items: The group of items to be stringified.

    Returns:
        The stringified version of the group.

    Examples:
        >>> print(f"In the 3rd question, you picked the number{_plural_marker([1])}.")
        In the 3rd question, you picked the number 1.
        >>> print(f"In the 3rd question, you picked the number{_plural_marker([1, 3])}.")
        In the 3rd question, you picked the numbers [1, 3].

    """
    if len(items) > 1:
        return f"s {items}"
    return f" {items[0]}"


@typechecked
class NoiseModel(ABC):
    """Abstract class used to represent a generic noise model, specifying
    criteria for applying different noise types to a quantum circuit or some of
    its qubits.

    This class allows one to specify which qubits (targets) and which gates of the circuit
    will be affected by this noise model. If you do not specify a target, the
    operation will apply to all qubits.

    Args:
        targets: Qubits affected by this noise. Defaults to all qubits.
        gates: Gates affected by this noise. Defaults to all gates.

    Raises:
        ValueError: When the target list is empty, or the target indices are duplicated
            or negative. When the size of the gate is higher than the number of target qubits.
    """

    def __init__(
        self,
        targets: Optional[list[int]] = None,
        gates: Optional[list[type[NativeGate]]] = None,
    ):
        if targets is None:
            targets = []
            self._dynamic = True
        else:
            self._dynamic = False
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
        """See parameter description."""
        self.gates = gates if gates is not None else []
        """See parameter description."""

    def connections(self) -> set[int]:
        """Qubits to which this is connected (applied to)."""
        return set(self.targets)

    @abstractmethod
    def to_kraus_operators(self) -> list[npt.NDArray[np.complex64]]:
        r"""Noise models can be represented by Kraus operators. They represent how the
        state is affected by the noise following the formula

        `\rho \leftarrow \sum_{K \in \mathcal{K}} K \rho K^\dagger`

        Where `\mathcal{K}` is the set of Kraus operators corresponding to the
        noise model and `\rho` is the state (as a density matrix).

        Returns:
            The Kraus operators of the noise. Note that it is not a unique
            representation.
        """
        pass

    def to_adjusted_kraus_operators(
        self, targets: set[int], size: int
    ) -> list[npt.NDArray[np.complex64]]:
        r"""In some cases, you may prefer the Kraus operators to match the size
        of your circuit, and the targets involved. In particular, the targets of
        the noise application may not match the noise targets, because the noise
        targets signifies all the qubits that the noise is applicable on, but if
        the noise happens at a gate execution, it would only actually impact the
        targets qubits of the gate.

        Note:
            This generic method considers that the default Kraus operators of
            the noise are for one qubit noises. If this is not the case, this
            method should be overloaded in the corresponding class.

        Args:
            targets: Qubits actually affected by the noise.
            size: Size of the desired Kraus operators.

        Returns:
            The Kraus operators adjusted to the targets of the gate on which the
            noise acts and the size of the circuit.
        """
        K = self.to_kraus_operators()
        return [
            reduce(np.kron, ops)
            for ops in product(
                *[K if t in targets else [I.matrix] for t in range(size)]
            )
        ]

    @abstractmethod
    def to_other_language(
        self, language: Language
    ) -> "BraketNoise | QLMNoise | QuantumError":
        """Transforms this noise model into the corresponding object in the
        language specified in the ``language`` arg.

        In the current version, only Braket and my_QLM are available for conversion.

        Args:
            language: Enum representing the target language.

        Returns:
            The corresponding noise model (or channel) in the target language.
        """
        pass

    def info(self) -> str:
        """For usage of pretty prints, this method displays in a string all
        information relevant to the noise at matter.

        Returns:
            The string displaying the noise information in a human readable
            manner.
        """
        noise_info = f"{type(self).__name__} noise:"
        if not self._dynamic:
            noise_info += f" on qubit{_plural_marker(self.targets)}"
        if len(self.gates) != 0:
            noise_info += f" for gate{_plural_marker(self.gates)}"

        return noise_info

    # 3M-TODO: implement the possibility of having a parameterized noise
    # param: Union[float, Expr]
    # @abstractmethod
    # def subs(self):
    #     pass


@typechecked
class DimensionalNoiseModel(NoiseModel, ABC):
    """Abstract class representing a multi-dimensional NoiseModel.

    Args:
        targets: List of qubit indices affected by this noise.
        dimension: Dimension of the noise model.
        gates: List of :class:`~mpqp.core.instructions.gates.native_gates.NativeGate`
            affected by this noise.

    Raises:
        ValueError: When a negative or zero dimension is input.
        ValueError: When the size of the specified gates is not coherent with
            the number of targets or the dimension.
    """

    def __init__(
        self,
        targets: Optional[list[int]] = None,
        dimension: int = 1,
        gates: Optional[list[type[NativeGate]]] = None,
    ):
        if dimension <= 0:
            raise ValueError(
                "Dimension of a multi-dimensional NoiseModel must be strictly greater"
                f" than 1, but got {dimension} instead."
            )

        if gates is not None:
            if any(
                gate.nb_qubits
                != dimension  # pyright: ignore[reportUnnecessaryComparison]
                for gate in gates
            ):
                raise ValueError(
                    f"Dimension of the noise model is {dimension}, but got specified gate(s) of different size."
                )

        super().__init__(targets, gates)
        self.dimension = dimension
        """Dimension of the depolarizing noise model."""
        self.check_dimension()

    def check_dimension(self):
        if 0 < len(self.targets) < self.dimension:
            raise ValueError(
                f"Number of target qubits {len(self.targets)} should be higher than the dimension {self.dimension}."
            )


@typechecked
class Depolarizing(DimensionalNoiseModel):
    """Class representing the depolarizing noise channel, which maps a state
    onto a linear combination of itself and the maximally mixed state. It can
    be applied to a single or multiple qubits, and depends on a single parameter
    (probability or error rate).

    When the number of qubits in the target is higher than the dimension, the
    noise will be applied to all possible combinations of indices of size
    ``dimension``.

    Args:
        prob: Depolarizing error probability (also called error rate).
        targets: Qubits affected by this noise. Defaults to all qubits.
        dimension: Dimension of the depolarizing channel.
        gates: Gates affected by this noise. Defaults to all gates.

    Raises:
        ValueError: When a wrong dimension (negative) or probability (outside of
            the expected interval) is input. When the size of the specified
            gates is not consistent with the number of targets or the dimension.

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
            Depolarizing(0.05, [0, 1], dimension=2)
            Depolarizing(0.12, [2], gates=[H, Rx, Ry, Rz])
            Depolarizing(0.05, [0, 1, 2], dimension=2, gates=[CNOT, CZ])
        >>> print(circuit.to_other_language(Language.BRAKET))  # doctest: +NORMALIZE_WHITESPACE
        T  : │                        0                         │
              ┌───┐ ┌────────────┐ ┌────────────┐
        q0 : ─┤ H ├─┤ DEPO(0.01) ├─┤ DEPO(0.32) ├────────────────
              └───┘ └────────────┘ └────────────┘
              ┌───┐ ┌────────────┐ ┌────────────┐
        q1 : ─┤ H ├─┤ DEPO(0.01) ├─┤ DEPO(0.32) ├────────────────
              └───┘ └────────────┘ └────────────┘
              ┌───┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
        q2 : ─┤ H ├─┤ DEPO(0.12) ├─┤ DEPO(0.01) ├─┤ DEPO(0.32) ├─
              └───┘ └────────────┘ └────────────┘ └────────────┘
        T  : │                        0                         │

    """

    def __init__(
        self,
        prob: float,
        targets: Optional[list[int]] = None,
        dimension: int = 1,
        gates: Optional[list[type[NativeGate]]] = None,
    ):
        super().__init__(targets, dimension, gates)
        self.prob = prob
        """Probability or error rate of the depolarizing noise model."""

        prob_upper_bound = 1 if dimension == 1 else 1 + 1 / (dimension**2 - 1)
        if not (0 <= prob <= prob_upper_bound):
            raise ValueError(
                f"Invalid probability: {prob} but should have been between 0 "
                f"and {prob_upper_bound}."
            )

    def to_kraus_operators(self) -> list[npt.NDArray[np.complex64]]:
        return [
            np.sqrt(1 - 3 * self.prob / 4) * I.matrix,
            np.sqrt(self.prob / 4) * X.matrix,
            np.sqrt(self.prob / 4) * Y.matrix,
            np.sqrt(self.prob / 4) * Z.matrix,
        ]

    def __repr__(self):
        target = f", {self.targets}" if not self._dynamic else ""
        dimension = f", dimension={self.dimension}" if self.dimension != 1 else ""
        gates = f", gates={self.gates}" if len(self.gates) != 0 else ""
        return f"Depolarizing({self.prob}{target}{dimension}{gates})"

    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> "BraketNoise | TwoQubitDepolarizing | QLMNoise | QuantumError":
        """See the documentation for this method in the abstract mother class :class:`NoiseModel`.

        Args:
            language: Enum representing the target language.

        Examples:
            >>> Depolarizing(0.3, [0,1], dimension=1).to_other_language(Language.BRAKET)
            Depolarizing('probability': 0.3, 'qubit_count': 1)

            >>> Depolarizing(0.3, [0,1], dimension=1).to_other_language(Language.QISKIT).to_quantumchannel()
            SuperOp([[0.85+0.j, 0.  +0.j, 0.  +0.j, 0.15+0.j],
                     [0.  +0.j, 0.7 +0.j, 0.  +0.j, 0.  +0.j],
                     [0.  +0.j, 0.  +0.j, 0.7 +0.j, 0.  +0.j],
                     [0.15+0.j, 0.  +0.j, 0.  +0.j, 0.85+0.j]],
                    input_dims=(2,), output_dims=(2,))

            >>> print(Depolarizing(0.3, [0,1], dimension=1).to_other_language(Language.MY_QLM))  # doctest: +NORMALIZE_WHITESPACE
            Depolarizing channel, p = 0.3:
            [[0.83666003 0.        ]
             [0.         0.83666003]]
            [[0.        +0.j 0.31622777+0.j]
             [0.31622777+0.j 0.        +0.j]]
            [[0.+0.j         0.-0.31622777j]
             [0.+0.31622777j 0.+0.j        ]]
            [[ 0.31622777+0.j  0.        +0.j]
             [ 0.        +0.j -0.31622777+0.j]]

        """
        if language == Language.BRAKET:
            if self.dimension > 2:
                raise NotImplementedError(
                    f"Depolarizing channel is not implemented in Braket for more than 2 qubits."
                )
            elif self.dimension == 2:
                from braket.circuits.noises import TwoQubitDepolarizing

                return TwoQubitDepolarizing(probability=self.prob)
            else:
                from braket.circuits.noises import Depolarizing as BraketDepolarizing

                return BraketDepolarizing(probability=self.prob)

        elif language == Language.QISKIT:
            from qiskit_aer.noise.errors.standard_errors import depolarizing_error

            return depolarizing_error(self.prob, self.dimension)

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
                prob=self.prob,
                nqbits=self.dimension,
                method_2q="equal_probs",
                depol_type="pauli",
            )
        else:
            raise NotImplementedError(f"Depolarizing is not implemented for {language}")

    def info(self) -> str:
        dimension = f" and dimension {self.dimension}" if self.dimension != 1 else ""
        return f"{super().info()} with probability {self.prob}{dimension}"


@typechecked
class BitFlip(NoiseModel):
    """Class representing the bit flip noise channel, which flips the state of
    a qubit with a certain probability. It can be applied to single and
    multi-qubit gates and depends on a single parameter (probability or error
    rate).

    Args:
        prob: Bit flip error probability or error rate (must be within
            ``[0, 0.5]``).
        targets: Qubits affected by this noise. Defaults to all qubits.
        gates: Gates affected by this noise. If multi-qubit gates is passed,
            single-qubit bitflip will be added for each qubit connected (target,
            control) with the gates. Defaults to all gates.

    Raises:
        ValueError: When the probability is outside of the expected interval
            ``[0, 0.5]``.

    Examples:
        >>> circuit = QCircuit(
        ...     [H(i) for i in range(3)]
        ...     + [
        ...         BitFlip(0.1, [0]),
        ...         BitFlip(0.3, [1, 2]),
        ...         BitFlip(0.05, [0], gates=[H]),
        ...         BitFlip(0.3),
        ...     ]
        ... )
        >>> print(circuit)
             ┌───┐
        q_0: ┤ H ├
             ├───┤
        q_1: ┤ H ├
             ├───┤
        q_2: ┤ H ├
             └───┘
        NoiseModel:
            BitFlip(0.1, [0])
            BitFlip(0.3, [1, 2])
            BitFlip(0.05, [0], gates=[H])
            BitFlip(0.3)
        >>> print(circuit.to_other_language(Language.BRAKET)) # doctest: +NORMALIZE_WHITESPACE
        T  : │                    0                     │
              ┌───┐ ┌─────────┐ ┌──────────┐ ┌─────────┐
        q0 : ─┤ H ├─┤ BF(0.3) ├─┤ BF(0.05) ├─┤ BF(0.1) ├─
              └───┘ └─────────┘ └──────────┘ └─────────┘
              ┌───┐ ┌─────────┐ ┌─────────┐
        q1 : ─┤ H ├─┤ BF(0.3) ├─┤ BF(0.3) ├──────────────
              └───┘ └─────────┘ └─────────┘
              ┌───┐ ┌─────────┐ ┌─────────┐
        q2 : ─┤ H ├─┤ BF(0.3) ├─┤ BF(0.3) ├──────────────
              └───┘ └─────────┘ └─────────┘
        T  : │                    0                     │

    """

    def __init__(
        self,
        prob: float,
        targets: Optional[list[int]] = None,
        gates: Optional[list[type[NativeGate]]] = None,
    ):

        if not (0 <= prob <= 0.5):
            raise ValueError(
                f"Invalid probability: {prob} but should be between 0 and 0.5"
            )

        super().__init__(targets, gates)
        self.prob = prob
        """See parameter description."""

    def to_kraus_operators(self) -> list[npt.NDArray[np.complex64]]:
        return [np.sqrt(1 - self.prob) * I.matrix, np.sqrt(self.prob) * X.matrix]

    def __repr__(self):
        targets = f", {self.targets}" if not self._dynamic else ""
        gates = f", gates={self.gates}" if self.gates else ""
        return f"BitFlip({self.prob}{targets}{gates})"

    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> "BraketNoise | QLMNoise | QuantumError":
        """See documentation of this method in abstract mother class :class:`NoiseModel`.

        Args:
            language: Enum representing the target language.

        Examples:
            >>> BitFlip(0.3, [0,1]).to_other_language(Language.BRAKET)
            BitFlip('probability': 0.3, 'qubit_count': 1)

            >>> BitFlip(0.3, [0,1]).to_other_language(Language.QISKIT).to_quantumchannel()
            SuperOp([[0.7+0.j, 0. +0.j, 0. +0.j, 0.3+0.j],
                     [0. +0.j, 0.7+0.j, 0.3+0.j, 0. +0.j],
                     [0. +0.j, 0.3+0.j, 0.7+0.j, 0. +0.j],
                     [0.3+0.j, 0. +0.j, 0. +0.j, 0.7+0.j]],
                    input_dims=(2,), output_dims=(2,))

        """

        if language == Language.BRAKET:
            from braket.circuits.noises import BitFlip as BraketBitFlip

            return BraketBitFlip(probability=self.prob)

        elif language == Language.QISKIT:
            from qiskit_aer.noise.errors.standard_errors import pauli_error

            return pauli_error([("X", self.prob), ("I", 1 - self.prob)])

        # TODO: MY_QLM implementation

        else:
            raise NotImplementedError(f"{language.name} not yet supported.")

    def info(self) -> str:
        return f"{super().info()} with probability {self.prob}"


@typechecked
class AmplitudeDamping(NoiseModel):
    r"""Class representing the amplitude damping noise channel, which can model
    both the standard and generalized amplitude damping processes. It can be
    applied to a single qubit and depends on two parameters: the decay rate
    ``gamma`` and the probability of excitation ``prob``.

    We recall below the associated representation, in terms of Kraus operators,
    where we denote by `\gamma` the decay rate ``gamma``, and by `p` the
    excitation probability ``prob``:

    `E_0=\sqrt{p}\begin{pmatrix}1&0\\0&\sqrt{1-\gamma}\end{pmatrix}`,
    `~ ~ E_1=\sqrt{p}\begin{pmatrix}0&\sqrt{\gamma}\\0&0\end{pmatrix}`,
    `~ ~ E_2=\sqrt{1-p}\begin{pmatrix}\sqrt{1-\gamma}&0\\0&1\end{pmatrix}` and
    `~ E_3=\sqrt{1-p}\begin{pmatrix}0&0\\\sqrt{\gamma}&0\end{pmatrix}`.

    Args:
        gamma: Decaying rate of the amplitude damping noise channel.
        prob: Probability of excitation in the generalized amplitude damping
            noise channel. A value of 1, corresponds to the standard amplitude
            damping. It must be in the ``[0, 1]`` interval.
        targets: Qubits affected by this noise. Defaults to all qubits.
        gates: Gates affected by this noise. Defaults to all gates.

    Raises:
        ValueError: When the gamma or prob parameters are outside of the
            expected interval ``[0, 1]``.

    Examples:
        >>> circuit = QCircuit(
        ...     [H(i) for i in range(3)]
        ...     + [
        ...         AmplitudeDamping(0.2, 0, [0]),
        ...         AmplitudeDamping(0.4, 0.1, [1, 2]),
        ...         AmplitudeDamping(0.1, 1, [0, 1, 2]),
        ...         AmplitudeDamping(0.1, 1),
        ...         AmplitudeDamping(0.7, targets=[0, 1]),
        ...     ]
        ... )
        >>> print(circuit)
             ┌───┐
        q_0: ┤ H ├
             ├───┤
        q_1: ┤ H ├
             ├───┤
        q_2: ┤ H ├
             └───┘
        NoiseModel:
            AmplitudeDamping(0.2, 0, targets=[0])
            AmplitudeDamping(0.4, 0.1, targets=[1, 2])
            AmplitudeDamping(0.1, targets=[0, 1, 2])
            AmplitudeDamping(0.1)
            AmplitudeDamping(0.7, targets=[0, 1])
        >>> print(circuit.to_other_language(Language.BRAKET)) # doctest: +NORMALIZE_WHITESPACE
        T  : │                               0                               │
              ┌───┐ ┌─────────┐ ┌─────────┐   ┌─────────┐     ┌────────────┐
        q0 : ─┤ H ├─┤ AD(0.7) ├─┤ AD(0.1) ├───┤ AD(0.1) ├─────┤ GAD(0.2,0) ├──
              └───┘ └─────────┘ └─────────┘   └─────────┘     └────────────┘
              ┌───┐ ┌─────────┐ ┌─────────┐   ┌─────────┐    ┌──────────────┐
        q1 : ─┤ H ├─┤ AD(0.7) ├─┤ AD(0.1) ├───┤ AD(0.1) ├────┤ GAD(0.4,0.1) ├─
              └───┘ └─────────┘ └─────────┘   └─────────┘    └──────────────┘
              ┌───┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐
        q2 : ─┤ H ├─┤ AD(0.1) ├─┤ AD(0.1) ├─┤ GAD(0.4,0.1) ├──────────────────
              └───┘ └─────────┘ └─────────┘ └──────────────┘
        T  : │                               0                               │

    """

    def __init__(
        self,
        gamma: float,
        prob: float = 1,
        targets: Optional[list[int]] = None,
        gates: Optional[list[type[NativeGate]]] = None,
    ):
        if not (0 <= gamma <= 1):
            raise ValueError(
                f"Invalid decaying rate: {gamma}. It should be between 0 and 1."
            )

        if not (0 <= prob <= 1):
            raise ValueError(
                f"Invalid excitation probability: {prob}. It should be between 0 and 1."
            )

        super().__init__(targets, gates)
        self.gamma = gamma
        """See parameter description."""
        self.prob = prob
        """See parameter description."""

    def to_kraus_operators(self) -> list[npt.NDArray[np.complex64]]:
        return [
            np.diag(1, np.sqrt(1 - self.prob)),
            np.array([[0, np.sqrt(self.prob)], [0, 0]]),
        ]

    def __repr__(self):
        prob = f", {self.prob}" if self.prob != 1 else ""
        targets = f", targets={self.targets}" if not self._dynamic else ""
        gates = f", gates={self.gates}" if len(self.gates) != 0 else ""
        return f"AmplitudeDamping({self.gamma}{prob}{targets}{gates})"

    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> "BraketNoise | QLMNoise | QuantumError":
        """See documentation of this method in abstract mother class :class:`NoiseModel`.

        Args:
            language: Enum representing the target language.

        Examples:
            >>> AmplitudeDamping(0.4, targets=[0, 1]).to_other_language(Language.BRAKET)
            AmplitudeDamping('gamma': 0.4, 'qubit_count': 1)

            >>> AmplitudeDamping(0.4, 0.2, [1]).to_other_language(Language.BRAKET)
            GeneralizedAmplitudeDamping('gamma': 0.4, 'probability': 0.2, 'qubit_count': 1)

            >>> AmplitudeDamping(0.2, 0.4, [0, 1]).to_other_language(Language.QISKIT).to_quantumchannel()
            SuperOp([[0.88      +0.j, 0.        +0.j, 0.        +0.j, 0.08      +0.j],
                     [0.        +0.j, 0.89442719+0.j, 0.        +0.j, 0.        +0.j],
                     [0.        +0.j, 0.        +0.j, 0.89442719+0.j, 0.        +0.j],
                     [0.12      +0.j, 0.        +0.j, 0.        +0.j, 0.92      +0.j]],
                    input_dims=(2,), output_dims=(2,))

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

        # TODO: MY_QLM implementation

        elif language == Language.QISKIT:
            from qiskit_aer.noise.errors.standard_errors import amplitude_damping_error

            return amplitude_damping_error(
                self.gamma, 1 - self.prob  # pyright: ignore[reportArgumentType]
            )

        else:
            raise NotImplementedError(
                f"Conversion of Amplitude Damping noise for language {language} is not supported."
            )

    def info(self) -> str:
        prob = f" and probability {self.prob}" if self.prob != 1 else ""
        return f"{super().info()} with gamma {self.gamma}{prob}"


@typechecked
class PhaseDamping(NoiseModel):
    """Class representing the phase damping noise channel. It can be applied to
    a single qubit and depends on the phase damping parameter ``gamma``. Phase
    damping happens when a quantum system loses its phase information due to
    interactions with the environment, leading to decoherence.

    Args:
        gamma: Probability of phase damping.
        targets: Qubits affected by this noise. Defaults to all qubits.
        gates: Gates affected by this noise. Defaults to all gates.

    Raises:
        ValueError: When the gamma parameter is outside of the expected interval
            ``[0, 1]``.

    Examples:
        >>> circuit = QCircuit(
        ...     [H(i) for i in range(3)]
        ...     + [
        ...         PhaseDamping(0.32, list(range(3))),
        ...         PhaseDamping(0.01),
        ...         PhaseDamping(0.45, [0, 1]),
        ...     ]
        ... )
        >>> print(circuit)
             ┌───┐
        q_0: ┤ H ├
             ├───┤
        q_1: ┤ H ├
             ├───┤
        q_2: ┤ H ├
             └───┘
        NoiseModel:
            PhaseDamping(0.32, [0, 1, 2])
            PhaseDamping(0.01)
            PhaseDamping(0.45, [0, 1])
        >>> print(circuit.to_other_language(Language.BRAKET)) # doctest: +NORMALIZE_WHITESPACE
        T  : │                     0                      │
              ┌───┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
        q0 : ─┤ H ├─┤ PD(0.45) ├─┤ PD(0.01) ├─┤ PD(0.32) ├─
              └───┘ └──────────┘ └──────────┘ └──────────┘
              ┌───┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
        q1 : ─┤ H ├─┤ PD(0.45) ├─┤ PD(0.01) ├─┤ PD(0.32) ├─
              └───┘ └──────────┘ └──────────┘ └──────────┘
              ┌───┐ ┌──────────┐ ┌──────────┐
        q2 : ─┤ H ├─┤ PD(0.01) ├─┤ PD(0.32) ├──────────────
              └───┘ └──────────┘ └──────────┘
        T  : │                     0                      │

    """

    def __init__(
        self,
        gamma: float,
        targets: Optional[list[int]] = None,
        gates: Optional[list[type[NativeGate]]] = None,
    ):
        if not (0 <= gamma <= 1):
            raise ValueError(
                f"Invalid phase damping probability: {gamma}. It should be between 0 and 1."
            )

        super().__init__(targets, gates)
        self.gamma = gamma
        """Probability of phase damping."""

    def to_kraus_operators(self) -> list[npt.NDArray[np.complex64]]:
        return [
            np.sqrt(1 - self.gamma) * I.matrix,
            np.diag([np.sqrt(self.gamma), 0]),
            np.diag([0, np.sqrt(self.gamma)]),
        ]

    def __repr__(self):
        targets = f", {self.targets}" if not self._dynamic else ""
        gates = f", gates={self.gates}" if self.gates else ""
        return f"PhaseDamping({self.gamma}{targets}{gates})"

    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> "BraketNoise | QLMNoise | QuantumError":
        """See documentation of this method in abstract mother class :class:`NoiseModel`.

        Args:
            language: Enum representing the target language.

        Examples:
            >>> PhaseDamping(0.4, [0, 1]).to_other_language(Language.BRAKET)
            PhaseDamping('gamma': 0.4, 'qubit_count': 1)

            >>> PhaseDamping(0.4, [0, 1]).to_other_language(Language.QISKIT).to_quantumchannel()
            SuperOp([[1.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j],
                     [0.        +0.j, 0.77459667+0.j, 0.        +0.j, 0.        +0.j],
                     [0.        +0.j, 0.        +0.j, 0.77459667+0.j, 0.        +0.j],
                     [0.        +0.j, 0.        +0.j, 0.        +0.j, 1.        +0.j]],
                    input_dims=(2,), output_dims=(2,))

            >>> print(PhaseDamping(0.4, [0, 1]).to_other_language(Language.MY_QLM))  # doctest: +NORMALIZE_WHITESPACE
            Phase Damping channel, gamma = 0.4:
            [[1.         0.        ]
             [0.         0.77459667]]
            [[0.         0.        ]
             [0.         0.77459667]]

        """
        if language == Language.BRAKET:
            from braket.circuits.noises import PhaseDamping as BraketPhaseDamping

            return BraketPhaseDamping(self.gamma)

        elif language == Language.QISKIT:
            from qiskit_aer.noise.errors.standard_errors import phase_damping_error

            return phase_damping_error(self.gamma)

        elif language == Language.MY_QLM:
            from qat.quops.quantum_channels import QuantumChannelKraus

            return QuantumChannelKraus(
                [
                    np.diag([1, np.sqrt(1 - self.gamma)]),
                    np.diag([0, np.sqrt(1 - self.gamma)]),
                ],
                "Phase Damping channel, gamma = " + str(self.gamma),
            )

        else:
            raise NotImplementedError(
                f"Conversion of Phase Damping noise for language {language} is not supported."
            )

    def info(self) -> str:
        return f"{super().info()} with gamma {self.gamma}"


class Pauli(NoiseModel):
    """3M-TODO"""

    def __init__(self):
        raise NotImplementedError(
            f"{type(self).__name__} noise model is not yet implemented."
        )


class Dephasing(NoiseModel):
    """3M-TODO"""

    def __init__(self):
        raise NotImplementedError(
            f"{type(self).__name__} noise model is not yet implemented."
        )


class PhaseFlip(NoiseModel):
    """3M-TODO"""

    def __init__(self):
        raise NotImplementedError(
            f"{type(self).__name__} noise model is not yet implemented."
        )


NOISE_MODELS = [
    cls
    for _, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if issubclass(cls, NoiseModel)
    and not (
        any("ABC" in base.__name__ for base in cls.__bases__)
        or "M-TODO" in (cls.__doc__ or "")
    )
]
"""All concrete noise models."""
