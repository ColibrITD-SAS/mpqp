"""When we measure a quantum state, we project it in an orthonormal basis of the
associated Hilbert space. By default,
:class:`BasisMeasure<mpqp.core.instruction.measurement.basis_measure.BasisMeasure>`
operates in the computational basis, but you may want to measure the state in a
custom basis, like it can be the case in the Bell game. For this purpose, you
can use the :class:`Basis` class.

On the other hand, some common basis are available for you to use:
:class:`ComputationalBasis` and :class:`HadamardBasis`."""

from __future__ import annotations
from abc import abstractmethod
from functools import reduce
from typing import Optional

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

from mpqp.tools.maths import matrix_eq, atol


@typechecked
class Basis:
    """Represents a basis of the Hilbert space used for measuring a qubit.

    Args:
        basis_vectors: List of vector composing the basis.
        nb_qubits: Number of qubits associated with this basis. If not
            specified, it will be automatically inferred from
            ``basis_vectors``'s dimensions.

    Example:
        >>> Basis([np.array([1,0]), np.array([0,-1])]).pretty_print()
        Basis: [
            [1 0],
            [ 0 -1]
        ]
    """

    def __init__(
        self,
        basis_vectors: list[npt.NDArray[np.complex64]],
        nb_qubits: Optional[int] = None,
    ):
        # 3M-TODO : add the possibility to give the symbols for the '0' and '1' of the custom basis. This should then
        #  appear in the Sample binary_representation of the basis state. For instance in the Hadamard basis, the
        #  symbols will be '+' and '-'. If the user wants '↑' and '↓' for his custom basis, when we print samples we
        #  would have something like:
        #  State: ↑↑↓, Index: 1, Count: 512, Probability: 0.512
        if len(basis_vectors) == 0:
            self.nb_qubits = nb_qubits
            self.basis_vectors = basis_vectors
            return
        if nb_qubits is None:
            nb_qubits = int(np.log2(len(basis_vectors[0])))

        if len(basis_vectors) != 2**nb_qubits:
            raise ValueError(
                "Incorrect number of vector for the basis: given "
                f"{len(basis_vectors)} but there should be {2**nb_qubits}"
            )
        if any(len(vector) != 2**nb_qubits for vector in basis_vectors):
            raise ValueError("All vectors of the given basis are not the same size")
        if any(
            abs(np.linalg.norm(vector) - 1)  # pyright: ignore[reportGeneralTypeIssues]
            > atol
            for vector in basis_vectors
        ):
            raise ValueError("All vectors of the given basis are not normalized")
        m = np.array([vector for vector in basis_vectors])
        if not matrix_eq(
            m.transpose().dot(m),
            np.eye(len(basis_vectors)),  # pyright: ignore[reportArgumentType]
        ):
            raise ValueError("The given basis is not orthogonal")

        self.nb_qubits = nb_qubits
        """See parameter description."""
        self.basis_vectors = basis_vectors
        """See parameter description."""

    def pretty_print(self):
        """Nicer print for the basis, with human readable formatting.

        Example:
            >>> Basis([np.array([1,0]), np.array([0,-1])]).pretty_print()
            Basis: [
                [ 1  0],
                [ 0 -1]
            ]
        """
        joint_vectors = ",\n    ".join(map(str, np.round(self.basis_vectors, 2)))
        print(f"Basis: [\n    {joint_vectors}\n]")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.basis_vectors}, {self.nb_qubits})"


@typechecked
class VariableSizeBasis(Basis):
    """3M-TODO"""

    @abstractmethod
    def __init__(self, nb_qubits: Optional[int] = None):
        pass

    @abstractmethod
    def set_size(self, nb_qubits: int):
        """
        To allow the user to use a basis without having to specify the size
        (because implicitly the size should be the number of qubits of the
        circuit), we use this method, that will only be called once the
        circuit's size is definitive (i.e. at the last moment before the circuit
        is ran)

        Args:
            nb_qubits: number of qubits in the basis
        """
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ComputationalBasis(VariableSizeBasis):
    """Basis representing the computational basis, also called Z-basis or
    canonical basis.

    Args:
        nb_qubits: number of qubits of the space, if not given as input (for
            example if unknown at the moment of creation) ``set_size`` will have
            to be executed before the basis is used (in a measure for example).

    Examples:
        >>> ComputationalBasis(3).pretty_print()
        Basis: [
            [1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j],
            [0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j],
            [0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j],
            [0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j],
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j],
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j],
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j],
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j]
        ]
        >>> b = ComputationalBasis()
        >>> b.set_size(2)
        >>> b.pretty_print()
        Basis: [
            [1.+0.j 0.+0.j 0.+0.j 0.+0.j],
            [0.+0.j 1.+0.j 0.+0.j 0.+0.j],
            [0.+0.j 0.+0.j 1.+0.j 0.+0.j],
            [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
        ]
    """

    def __init__(self, nb_qubits: Optional[int] = None):
        basis = []
        Basis.__init__(self, basis, nb_qubits)
        if nb_qubits is not None:
            self.set_size(nb_qubits)

    def set_size(self, nb_qubits: int):
        self.basis_vectors = [
            np.array([0] * i + [1] + [0] * (2**nb_qubits - 1 - i), dtype=np.complex64)
            for i in range(2**nb_qubits)
        ]
        self.nb_qubits = nb_qubits


class HadamardBasis(VariableSizeBasis):
    """Basis representing the Hadamard basis, also called X-basis or +/- basis.

    Args:
        nb_qubits: number of qubits in the basis

    Example:
        >>> HadamardBasis(2).pretty_print()
        Basis: [
            [0.5+0.j 0.5+0.j 0.5+0.j 0.5+0.j],
            [ 0.5+0.j -0.5+0.j  0.5+0.j -0.5+0.j],
            [ 0.5+0.j  0.5+0.j -0.5+0.j -0.5+0.j],
            [ 0.5+0.j -0.5+0.j -0.5+0.j  0.5-0.j]
        ]
    """

    def __init__(self, nb_qubits: Optional[int] = None):
        basis = []
        Basis.__init__(self, basis, nb_qubits)
        if nb_qubits is not None:
            self.set_size(nb_qubits)

    def set_size(self, nb_qubits: int):
        H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        Hn = reduce(np.kron, [H] * nb_qubits, np.eye(1))
        self.basis_vectors = [line for line in Hn]
        self.nb_qubits = nb_qubits
