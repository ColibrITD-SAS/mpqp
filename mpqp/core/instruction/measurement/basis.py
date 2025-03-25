"""When we measure a quantum state, we project it in an orthonormal basis of the
associated Hilbert space. By default,
:class:`~mpqp.core.instruction.measurement.basis_measure.BasisMeasure`
operates in the computational basis, but you may want to measure the state in a
custom basis, like it can be the case in the Bell game. For this purpose, you
can use the :class:`Basis` class.

On the other hand, some common basis are available for you to use:
:class:`ComputationalBasis` and :class:`HadamardBasis`."""

from __future__ import annotations

from abc import abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

if TYPE_CHECKING:
    from mpqp import QCircuit

from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.tools.display import clean_1D_array, one_lined_repr
from mpqp.tools.maths import is_unitary


@typechecked
class Basis:
    """Represents a basis of the Hilbert space used for measuring a qubit.

    Args:
        basis_vectors: List of vector composing the basis.
        nb_qubits: Number of qubits associated with this basis. If not
            specified, it will be automatically inferred from
            ``basis_vectors``'s dimensions.

    Example:
        >>> custom_basis = Basis([np.array([1,0]), np.array([0,-1])], symbols=("↑", "↓"))
        >>> custom_basis.pretty_print()
        Basis: [
            [1, 0],
            [0, -1]
        ]
        >>> circ = QCircuit([X(0), H(0)])
        >>> circ.add(BasisMeasure([0], basis=custom_basis, shots=10000))
        >>> print(run(circ, IBMDevice.AER_SIMULATOR)) # doctest: +SKIP
        Result: IBMDevice, AER_SIMULATOR
         Counts: [5035, 4965]
         Probabilities: [0.5035, 0.4965]
         Samples:
          State: ↑, Index: 0, Count: 5035, Probability: 0.5035
          State: ↓, Index: 1, Count: 4965, Probability: 0.4965
         Error: None

    """

    def __init__(
        self,
        basis_vectors: list[npt.NDArray[np.complex64]],
        nb_qubits: Optional[int] = None,
        symbols: Optional[tuple[str, str]] = None,
        basis_vectors_labels: Optional[list[str]] = None,
    ):
        if symbols is not None and basis_vectors_labels is not None:
            raise ValueError(
                "You can only specify either symbols or basis_vectors_labels, "
                "not both."
            )

        if symbols is None:
            symbols = ("0", "1")
        self.symbols = symbols
        self.basis_vectors_labels = basis_vectors_labels

        if len(basis_vectors) == 0:
            if nb_qubits is None:
                raise ValueError(
                    "Empty basis and no number of qubits specified. Please at "
                    "least specify one of these two."
                )
            self.nb_qubits = nb_qubits
            self.basis_vectors = basis_vectors
            return
        if nb_qubits is None:
            nb_qubits = int(np.log2(len(basis_vectors[0])))

        if len(basis_vectors) != 2**nb_qubits:
            raise ValueError(
                "Incorrect number of vector for the basis: given "
                f"{len(basis_vectors)} but there should be {2**nb_qubits}."
            )
        if any(len(vector) != 2**nb_qubits for vector in basis_vectors):
            raise ValueError("All vectors of the given basis are not the same size.")
        if not is_unitary(np.array([vector for vector in basis_vectors])):
            raise ValueError(
                "The given basis is not orthogonal: the matrix of the "
                "concatenated vectors of the basis should be unitary."
            )

        self.nb_qubits = nb_qubits
        """See parameter description."""
        self.basis_vectors = basis_vectors
        """See parameter description."""

    def binary_to_custom(self, state: str) -> str:
        """Converts a binary string to a custom string representation.
        By default, it uses "0" and "1" but can be customized based on the provided `symbols`.

        Args:
            state: The binary string (e.g., "01") to be converted.

        Returns:
            The custom string representation of the binary state.

        Example:
            >>> basis = Basis([np.array([1,0]), np.array([0,-1])], symbols=("+", "-"))
            >>> custom_state = basis.binary_to_custom("01")
            >>> custom_state
            '+-'
        """
        if self.basis_vectors_labels is not None:
            return self.basis_vectors_labels[int(state, 2)]

        return ''.join(self.symbols[int(bit)] for bit in state)

    def pretty_print(self):
        """Nicer print for the basis, with human readable formatting."""
        joint_vectors = ",\n    ".join(map(clean_1D_array, self.basis_vectors))
        print(f"Basis: [\n    {joint_vectors}\n]")

    def __repr__(self) -> str:
        joint_vectors = "[" + ", ".join(map(one_lined_repr, self.basis_vectors)) + "]"
        args = []
        args.append(joint_vectors)
        if isinstance(self, VariableSizeBasis):
            args.append(f"nb_qubits={self.nb_qubits}")
        if self.symbols != ("0", "1"):
            args.append(f"symbols={self.symbols}")
        if self.basis_vectors_labels is not None:
            args.append(f"basis_vectors_labels={self.basis_vectors_labels}")
        return f"{type(self).__name__}({', '.join(args)})"

    def to_computational(self) -> QCircuit:
        """Converts the custom basis to the computational basis.

        This method creates a quantum circuit with a custom gate represented by
        a unitary transformation and applies it to all qubits before measurement.

        Returns:
            A quantum circuit representing the basis change circuit.

        Example:
            >>> basis = Basis([np.array([1, 0]), np.array([0, -1])])
            >>> circuit = basis.to_computational()
            >>> print(circuit)
               ┌─────────┐
            q: ┤ Unitary ├
               └─────────┘

        """

        from mpqp.core.circuit import QCircuit

        basis_change = np.array(self.basis_vectors).T.conjugate()
        return QCircuit(
            [
                CustomGate(
                    UnitaryMatrix(basis_change), targets=list(range(self.nb_qubits))
                )
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Basis):
            return False

        return (
            self.nb_qubits == other.nb_qubits
            and np.array_equal(self.basis_vectors, other.basis_vectors)
            and self.symbols == other.symbols
            and self.basis_vectors_labels == other.basis_vectors_labels
        )


@typechecked
class VariableSizeBasis(Basis):
    """A variable-size basis with a dynamically adjustable size to different qubit numbers
    during circuit execution.

    Args:
        nb_qubits: number of qubits in the basis. If not provided,
            the basis can be dynamically sized later using the `set_size` method.
        symbols: custom symbols for representing basis states, defaults to ("0", "1").

    Example:
        >>> custom_basis = VariableSizeBasis([np.array([1,0]), np.array([0,-1])], symbols=("↑", "↓"))
        >>> custom_basis.pretty_print()
        Basis: [
            [1, 0],
            [0, -1]
        ]
        >>> circ = QCircuit([X(0), H(1), CNOT(1, 2), Y(2)])
        >>> circ.add(BasisMeasure(basis=custom_basis, shots=10000))
        >>> print(run(circ, IBMDevice.AER_SIMULATOR)) # doctest: +SKIP
        Result: None, IBMDevice, AER_SIMULATOR
         Counts: [0, 0, 0, 0, 0, 4936, 5064, 0]
         Probabilities: [0, 0, 0, 0, 0, 0.4936, 0.5064, 0]
         Samples:
          State: ↓↑↓, Index: 5, Count: 4936, Probability: 0.4936
          State: ↓↓↑, Index: 6, Count: 5064, Probability: 0.5064
         Error: None

    """

    def __init__(
        self,
        basis_vectors: list[npt.NDArray[np.complex64]],
        nb_qubits: Optional[int] = None,
        symbols: Optional[tuple[str, str]] = None,
    ):
        super().__init__(basis_vectors, symbols=symbols)
        self._init_basis = Basis(basis_vectors, symbols=symbols)
        self._dynamic = True if nb_qubits is None else False
        nb_qubits = (
            int(np.log2(len(basis_vectors[0]))) if nb_qubits is None else nb_qubits
        )
        self.set_size(nb_qubits)

    @abstractmethod
    def set_size(self, nb_qubits: int):
        """To allow the user to use a basis without having to specify the size
        (because implicitly the size should be the number of qubits of the
        circuit), we use this method, that will only be called once the
        circuit's size is definitive (i.e. at the last moment before the circuit
        is ran)

        Args:
            nb_qubits: number of qubits in the basis
        """
        if self.nb_qubits == nb_qubits:
            return

        if nb_qubits < self._init_basis.nb_qubits:
            raise ValueError(
                f"Invalid number of qubits ({nb_qubits}): must be at least the "
                f"size of the initial basis ({self._init_basis.nb_qubits})."
            )
        if nb_qubits % self._init_basis.nb_qubits != 0:
            raise ValueError(
                f"Invalid number of qubits ({nb_qubits}): must be a multiple of "
                f"the initial basis size ({self._init_basis.nb_qubits})."
            )

        basis_matrix = reduce(
            np.kron,
            [self._init_basis.basis_vectors]
            * (nb_qubits // self._init_basis.nb_qubits),
            np.eye(1),
        )
        self.basis_vectors = [line for line in basis_matrix]
        self.nb_qubits = nb_qubits

    def __repr__(self) -> str:
        args = []
        args.append(f"{self._init_basis.basis_vectors}")
        if not self._dynamic:
            args.append(f"{self.nb_qubits}")
        if self.symbols != ("0", "1"):
            args.append(f"symbols={self.symbols}")

        return f"{type(self).__name__}({', '.join(args)})"


class ComputationalBasis(VariableSizeBasis):
    """Basis representing the computational basis, also called Z-basis or
    canonical basis.

    Args:
        nb_qubits: number of qubits to measure, if not specified, the size will
            be dynamic and automatically span across the whole circuit, even
            through dimension change of the circuit.

    Examples:
        >>> ComputationalBasis(3).pretty_print()
        Basis: [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ]
        >>> b = ComputationalBasis()
        >>> b.set_size(2)
        >>> b.pretty_print()
        Basis: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]

    """

    def __init__(self, nb_qubits: Optional[int] = None):
        super().__init__([np.array([1, 0]), np.array([0, 1])], nb_qubits=nb_qubits)

    def set_size(self, nb_qubits: int):
        if self.nb_qubits == nb_qubits:
            return
        self.basis_vectors = [
            np.array([0] * i + [1] + [0] * (2**nb_qubits - 1 - i), dtype=np.complex64)
            for i in range(2**nb_qubits)
        ]
        self.nb_qubits = nb_qubits

    def to_computational(self) -> QCircuit:
        from mpqp.core.circuit import QCircuit

        return QCircuit(self.nb_qubits)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.nb_qubits if not self._dynamic else ''})"


class HadamardBasis(VariableSizeBasis):
    """Basis representing the Hadamard basis, also called X-basis or +/- basis.

    Args:
        nb_qubits: number of qubits to measure, if not specified, the size will
            be dynamic and automatically span across the whole circuit, even
            through dimension change of the circuit.

    Example:
        >>> HadamardBasis(2).pretty_print()
        Basis: [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5]
        ]
        >>> circ = QCircuit([X(0), H(1), CNOT(1, 2), Y(2)])
        >>> circ.add(BasisMeasure(basis=HadamardBasis()))
        >>> print(run(circ, IBMDevice.AER_SIMULATOR)) # doctest: +SKIP
        Result: None, IBMDevice, AER_SIMULATOR
         Counts: [0, 261, 253, 0, 0, 244, 266, 0]
         Probabilities: [0, 0.25488, 0.24707, 0, 0, 0.23828, 0.25977, 0]
         Samples:
          State: ++-, Index: 1, Count: 261, Probability: 0.2548828
          State: +-+, Index: 2, Count: 253, Probability: 0.2470703
          State: -+-, Index: 5, Count: 244, Probability: 0.2382812
          State: --+, Index: 6, Count: 266, Probability: 0.2597656
         Error: None
    """

    def __init__(self, nb_qubits: Optional[int] = None):
        super().__init__(
            [np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)],
            nb_qubits=nb_qubits,
            symbols=('+', '-'),
        )

    def set_size(self, nb_qubits: int):
        if self.nb_qubits == nb_qubits:
            return
        H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        Hn = reduce(np.kron, [H] * nb_qubits, np.eye(1))
        self.basis_vectors = [line for line in Hn]
        self.nb_qubits = nb_qubits

    def to_computational(self) -> QCircuit:
        from mpqp.core.circuit import QCircuit
        from mpqp.core.instruction.gates.native_gates import H

        if self.nb_qubits == 0:
            return QCircuit(self.nb_qubits)
        return QCircuit([H(qb) for qb in range(self.nb_qubits)])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.nb_qubits if not self._dynamic else ''})"
