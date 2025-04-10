"""Observables can be defined using (real valued) linear combinations of Pauli
operators, these are called "Pauli strings". In ``mpqp``, a :class:`PauliString`
is a linear combination of :class:`PauliStringMonomial` which are themselves
combinations (tensor products) of :class:`PauliStringAtom`. :class:`PauliString`
can be added, subtracted and tensored together, as well as multiplied by scalars."""

from __future__ import annotations

from copy import deepcopy
from functools import reduce
from numbers import Real
from operator import matmul, mul
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from sympy import Expr
from typeguard import typechecked

from mpqp.core.instruction.gates.native_gates import H, NativeGate, S_dagger
from mpqp.core.languages import Language
from mpqp.tools import NumberQubitsError, format_element
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import atol, is_power_of_two, rtol

if TYPE_CHECKING:
    from braket.circuits.observables import Observable as BraketObservable
    from braket.circuits.observables import Sum as BraketSum
    from braket.circuits.observables import TensorProduct
    from cirq.circuits.circuit import Circuit as CirqCircuit
    from cirq.ops.gate_operation import GateOperation as CirqGateOperation
    from cirq.ops.linear_combinations import PauliSum as CirqPauliSum
    from cirq.ops.pauli_string import PauliString as CirqPauliString
    from cirq.ops.raw_types import Qid
    from qat.core.wrappers.observable import Term
    from qiskit.quantum_info import SparsePauliOp

Coef = Union[Real, float, Expr]


@typechecked
class PauliString:
    """Represents a Pauli string, a linear combination of Pauli monomials.

    Note that as a user of our library, you would most likely never need to
    directly call this class. Instead, we advise you to build the Pauli strings
    you need from the atom we provide (see the example bellow).

    Args:
        monomials : List of Pauli monomials defining the PauliString.

    Example:
        >>> from mpqp.measures import I, X, Y, Z
        >>> I @ Z + 2 * Y @ I + X @ Z
        I@Z + 2*Y@I + X@Z

    Note:
        Pauli atoms are named ``I``, ``X``, ``Y``, and ``Z``. If you have
        conflicts with the gates of the same name, you could:

        - Rename the Pauli atoms:

        .. code-block:: python

            from mpqp.measures import X as Pauli_X,  Y as Pauli_Y
            ps = Pauli_X + Pauli_Y/2

        - Import the Pauli atoms directly from the module:

        .. code-block:: python

            from mpqp.measures import pauli_string
            ps = pauli_string.X + pauli_string.Y/2
    """

    def __init__(self, monomials: Optional[list["PauliStringMonomial"]] = None):
        self._monomials: list[PauliStringMonomial] = []

        if monomials is not None:
            for mono in monomials:
                if isinstance(mono, PauliStringAtom):
                    mono = PauliStringMonomial(1, [mono])
                self._monomials.append(mono)

        self._initial_nb_qubits = (
            self._monomials[0].nb_qubits if len(self._monomials) != 0 else 0
        )

        for mono in self._monomials:
            if mono.nb_qubits != self._initial_nb_qubits:
                raise ValueError(
                    f"Non homogeneous sizes for given PauliStrings: {monomials}"
                )

    @property
    def monomials(self) -> list[PauliStringMonomial]:
        """Monomials of the PauliString."""
        return self._monomials

    @property
    def nb_qubits(self) -> int:
        """Number of qubits associated with the PauliString."""
        return (
            self._monomials[0].nb_qubits
            if len(self._monomials) != 0
            else self._initial_nb_qubits
        )

    def _non_null_str(self):
        return str(self._monomials[0]) + "".join(
            (f" - {-m}" if not isinstance(m.coef, Expr) and m.coef < 0 else f" + {m}")
            for m in self._monomials[1:]
        )

    def __str__(self):
        sorted_ps = self.round().simplify().sorted_monomials()
        return "0" if len(sorted_ps._monomials) == 0 else sorted_ps._non_null_str()

    def __repr__(self):
        return "PauliString()" if len(self._monomials) == 0 else self._non_null_str()

    def __pos__(self) -> "PauliString":
        return deepcopy(self)

    def __neg__(self) -> "PauliString":
        return -1 * self

    def __iadd__(self, other: "PauliString") -> "PauliString":
        if len(self.monomials) != 0:
            if not all([mono.nb_qubits == self.nb_qubits for mono in other.monomials]):
                raise ValueError(
                    f"Non homogeneous sizes for given PauliStrings: {(self, other)}"
                )
        else:
            self._initial_nb_qubits = other.nb_qubits
        self._monomials.extend(deepcopy(other.monomials))
        return self

    def __add__(self, other: "PauliString") -> "PauliString":
        res = deepcopy(self)
        res += other
        return res

    def __isub__(self, other: "PauliString") -> "PauliString":
        self += -1 * other
        return self

    def __sub__(self, other: "PauliString") -> "PauliString":
        return self + (-1) * other

    def __imul__(self, other: Coef) -> "PauliString":
        for i, mono in enumerate(self._monomials):
            if isinstance(mono, PauliStringAtom):
                self.monomials[i] = PauliStringMonomial(atoms=[mono])
            self.monomials[i] *= other
        return self

    def __mul__(self, other: Coef) -> "PauliString":
        res = deepcopy(self)
        res *= other
        return res

    def __rmul__(self, other: Coef) -> "PauliString":
        return self * other

    def __itruediv__(self, other: Coef) -> "PauliString":
        self *= 1 / other  # pyright: ignore[reportOperatorIssue]
        return self

    def __truediv__(self, other: Coef) -> "PauliString":
        return self * (1 / other)  # pyright: ignore[reportOperatorIssue]

    def __imatmul__(self, other: "PauliString") -> "PauliString":
        self._monomials = [
            mono for s_mono in self.monomials for mono in (s_mono @ other).monomials
        ]
        return self

    def __matmul__(self, other: "PauliString") -> "PauliString":
        res = deepcopy(self)
        res @= other
        return res

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliString):
            return False

        return self.to_dict() == other.to_dict()

    def subs(
        self, values: dict[Expr | str, Real], remove_symbolic: bool = True
    ) -> PauliString:
        r"""Substitute the coef of the pauli sting with values for each of the
        specified coef. Optionally also remove all symbolic variables such
        as `\pi` (needed for example for circuit execution).

        Since we use ``sympy`` for the gate parameters, the ``values`` can in fact be
        anything the ``subs`` method from ``sympy`` would accept.

        Args:
            values: Mapping between the variables and the replacing values.
            remove_symbolic: Whether symbolic values should be replaced by their
                numeric counterparts.

        Returns:
            The pauli sting with the replaced parameters.

        Examples:
            >>> from mpqp.measures import I, X, Y, Z
            >>> theta, k = symbols("θ k")
            >>> ps = theta * I @ X + k * Z @ Y
            >>> print(ps)
            (θ)*I@X + (k)*Z@Y
            >>> print(ps.subs({theta: np.pi, k: 1}))
            3.1416*I@X + Z@Y

        """
        new_pauli_string = PauliString()
        for monomial in self.monomials:
            substituted_monomial = monomial.subs(values, remove_symbolic)
            new_pauli_string += substituted_monomial
        return new_pauli_string

    def simplify(self, inplace: bool = False) -> PauliString:
        """Simplifies the Pauli string by combining identical terms and removing
        terms with null coefficients. When all terms annihilate themselves, we
        return an empty PauliString with a number of qubits corresponding to the
        initial one.

        Args:
            inplace: Indicates if ``self`` should be updated in addition of a
                new Pauli string being returned.

        Returns:
            The simplified version of the Pauli string.

        Example:
            >>> from mpqp.measures import I, X, Y, Z
            >>> (I @ I - 2 *I @ I + Z @ I - Z @ I).simplify()
            -1*I@I

        """
        res = PauliString()
        res._initial_nb_qubits = self.nb_qubits
        for unique_mono_atoms in {tuple(mono.atoms) for mono in self.monomials}:
            coef: Coef = format_element(
                sum(  # pyright: ignore[reportCallIssue, reportAssignmentType]
                    [
                        mono.coef
                        for mono in self.monomials
                        if mono.atoms == list(unique_mono_atoms)
                    ]  # pyright: ignore[reportArgumentType]
                )
            )
            if coef != 0:
                res.monomials.append(PauliStringMonomial(coef, list(unique_mono_atoms)))

        if inplace:
            self._monomials = res.monomials
        return res

    def round(self, max_digits: int = 4) -> PauliString:
        """Rounds the coefficients of the PauliString to a specified number of
        decimals.

        Args:
            max_digits: Number of decimal places to round the coefficients to.

        Returns:
            The Pauli string with coefficients rounded to the specified number of
            decimals.

        Example:
            >>> from mpqp.measures import I, X, Y, Z
            >>> ps = 0.6875 * I @ I + 0.1275 * I @ Z
            >>> ps.round(1)
            0.7*I@I + 0.1*I@Z

        """
        from sympy import Expr

        res = PauliString()
        res._initial_nb_qubits = self.nb_qubits
        for mono in self.monomials:
            coef: Coef = format_element(
                mono.coef
            )  # pyright: ignore[reportAssignmentType]
            if isinstance(coef, Expr):
                res.monomials.append(PauliStringMonomial(mono.coef, mono.atoms))
            else:
                coef = float(np.round(float(coef), max_digits))
                if coef != 0:
                    res.monomials.append(PauliStringMonomial(coef, mono.atoms))
        return res

    def sorted_monomials(self) -> PauliString:
        """Creates a new Pauli string with the same monomials but sorted in
        monomial alphabetical ascending order (and the coefficients are not
        taken into account).

        Returns:
            The Pauli string with its monomials sorted.

        Example:
            >>> from mpqp.measures import I, X, Y, Z
            >>> (2*I@Z + .5*I@X + X@Y).sorted_monomials()
            0.5*I@X + 2*I@Z + X@Y
        """
        return PauliString(
            sorted(self.monomials, key=lambda m: tuple(str(atom) for atom in m.atoms))
        )

    def to_matrix(self) -> Matrix:
        """Converts the PauliString to a matrix representation.

        Returns:
            Matrix representation of the Pauli string.

        Example:
            >>> from mpqp.measures import I, X, Y, Z
            >>> pprint((I + Z).to_matrix())
            [[2, 0],
             [0, 0]]

        """
        self.simplify(inplace=True)
        size = 2**self.nb_qubits
        return sum(
            map(lambda m: m.to_matrix(), self.monomials),
            start=np.zeros((size, size), dtype=np.complex64),
        )

    def commutes_with(self, other: PauliString) -> bool:
        """
        3M-TODO: Determine EFFICIENTLY if this pauli string is commuting with the one in parameter.
        Args:
            other:

        Returns:

        Examples:
            >>> from mpqp.measures import I, X, Y, Z

        """
        raise NotImplementedError()

    @staticmethod
    def from_matrix(
        matrix: Matrix, method: Literal["ptdr", "trace"] = "ptdr"
    ) -> PauliString:
        """Constructs a PauliString from a matrix.

        Args:
            matrix: Matrix from which the PauliString is generated.
            method: String indicating which Pauli decomposition method is used.
                "ptdr" refers to the tree-based decomposition algorithm (see
                :func:`~mpqp.tools.obs_decomposition.decompose_hermitian_matrix_ptdr`.),
                and "trace" to the one computing the trace of the observable
                with each possible monomial (naive method).

        Returns:
            Pauli string corresponding to the Pauli decomposition of the matrix
            in parameter.

        Raises:
            ValueError: If the input matrix is not square or its dimensions are
                not a power of 2.

        Example:
            >>> PauliString.from_matrix(np.array([[0, 1], [1, 2]]))
            I + X - Z

        """

        if method == "ptdr":
            from mpqp.tools.obs_decomposition import decompose_hermitian_matrix_ptdr

            return decompose_hermitian_matrix_ptdr(matrix)

        elif method == "trace":
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square.")

            if not is_power_of_two(matrix.shape[0]):
                raise ValueError("Matrix dimensions must be a power of 2.")
            num_qubits = int(np.log2(matrix.shape[0]))

            # Return the ordered Pauli basis for the n-qubit Pauli basis.
            pauli_1q = [PauliStringMonomial(1, [atom]) for atom in [I, X, Y, Z]]
            basis = pauli_1q
            for _ in range(num_qubits - 1):
                basis = [p1 @ p2 for p1 in basis for p2 in pauli_1q]

            pauli_list = PauliString()
            for i, mat in enumerate(basis):
                coeff = (np.trace(mat.to_matrix().dot(matrix)) / (2**num_qubits)).real
                if not np.isclose(coeff, 0, atol=atol, rtol=rtol):
                    mono = basis[i] * coeff
                    pauli_list += mono

            if len(pauli_list.monomials) == 0:
                pauli_list.monomials.append(
                    PauliStringMonomial(0, [I for _ in range(num_qubits)])
                )
            return pauli_list
        else:
            raise ValueError(
                f"Unexpected observable matrix decomposition method name {method}."
            )

    @staticmethod
    def from_diagonal_elements(
        diagonal_elements: list[float] | npt.NDArray[np.float64],
        method: Literal["walsh", "ptdr"] = "walsh",
    ) -> PauliString:
        """Create a PauliString from the diagonal elements of a diagonal
        observable, by using decomposition algorithms in the Pauli basis.

        Currently, two different methods are available: "walsh" and "ptdr".
        The first one is based on the computation of a walsh-hadamard matrix to
        retrieve the coefficients in front of the monomials that are
        combinations of ``I`` and ``Z``. The second is an adaptation of the PTDR
        algorithm (Pauli Tree Decomposition Routine), see
        :func:`~mpqp.tools.obs_decomposition.decompose_diagonal_observable_ptdr`.

        Args:
            diagonal_elements: List of Real coefficients
            method: String used to specify the decomposition method. "walsh"  or
                "ptdr".

        Returns:
            The pauli decomposition of the given diagonal observable elements.

        Example:
            >>> PauliString.from_diagonal_elements([1, -1, 4, 2])
            1.5*I@I + I@Z - 1.5*Z@I
        """

        if not is_power_of_two(len(diagonal_elements)):
            raise ValueError(
                f"Size of diagonal elements must be a power of 2, but got {len(diagonal_elements)}."
            )

        if method == "walsh":
            from mpqp.tools.obs_decomposition import (
                decompose_diagonal_observable_walsh_hadamard,
            )

            return decompose_diagonal_observable_walsh_hadamard(diagonal_elements)

        elif method == "ptdr":
            from mpqp.tools.obs_decomposition import decompose_diagonal_observable_ptdr

            return decompose_diagonal_observable_ptdr(diagonal_elements)
        else:
            raise ValueError(
                f"Unexpected observable matrix decomposition method name {method}."
            )

    @staticmethod
    def _get_dimension_cirq_pauli(
        pauli: Union[CirqPauliSum, CirqPauliString, CirqGateOperation],
    ):
        from cirq.ops.gate_operation import GateOperation as CirqGateOperation
        from cirq.ops.linear_combinations import PauliSum as CirqPauliSum
        from cirq.ops.pauli_string import PauliString as CirqPauliString

        dimension = 0
        if isinstance(pauli, CirqPauliSum):
            for term in pauli:
                for qubit, _ in term.items():
                    nb_qubits = (
                        int(qubit.x)
                        if hasattr(qubit, "x")
                        else int(qubit.name.split("_")[1])
                    )
                    dimension = max(dimension, nb_qubits + 1)
        elif isinstance(pauli, CirqPauliString):
            for qubit in pauli.qubits:
                nb_qubits = (
                    int(qubit.x)
                    if hasattr(qubit, "x")
                    else int(qubit.name.split("_")[1])
                )
                dimension = max(dimension, nb_qubits + 1)
        elif isinstance(pauli, CirqGateOperation):
            for line_qubit in pauli._qubits:
                nb_qubits = int(line_qubit.x)
                dimension = max(dimension, nb_qubits + 1)
        return dimension

    @staticmethod
    def _from_cirq(
        pauli: Union[CirqPauliSum, CirqPauliString, CirqGateOperation],
        min_dimension: int = 1,
    ) -> PauliString:
        from cirq.ops.gate_operation import GateOperation as CirqGateOperation
        from cirq.ops.identity import I as Cirq_I
        from cirq.ops.linear_combinations import PauliSum as CirqPauliSum
        from cirq.ops.pauli_gates import X as Cirq_X
        from cirq.ops.pauli_gates import Y as Cirq_Y
        from cirq.ops.pauli_gates import Z as Cirq_Z
        from cirq.ops.pauli_string import PauliString as CirqPauliString

        ps_mapping = {Cirq_X: X, Cirq_Y: Y, Cirq_Z: Z, Cirq_I: I}

        num_qubits = max(PauliString._get_dimension_cirq_pauli(pauli), min_dimension)
        pauli_string = PauliString()

        def process_term(term: CirqPauliString, pauli_string: PauliString):
            coef = term.coefficient.real
            monomial = [I] * num_qubits
            for qubit, op in term.items():
                index = (
                    int(qubit.x)
                    if hasattr(qubit, "x")
                    else int(qubit.name.split("_")[1])
                )
                monomial[index] = ps_mapping[op]
            pauli_string += PauliStringMonomial(coef, monomial)

        if isinstance(pauli, CirqPauliSum):
            for term in pauli:
                process_term(term, pauli_string)
        elif isinstance(pauli, CirqPauliString):
            process_term(pauli, pauli_string)
        elif isinstance(pauli, CirqGateOperation):
            monomial = [I] * num_qubits
            for line_qubit in pauli._qubits:
                index = int(line_qubit.x)
                monomial[index] = ps_mapping[pauli._gate]
            pauli_string += PauliStringMonomial(1, monomial)

        return pauli_string

    @staticmethod
    def _from_qiskit(
        pauli: SparsePauliOp,
    ) -> PauliString:
        pauli_string = PauliString()
        for pauli_str, coef in pauli.to_list():
            monomial = PauliStringMonomial()
            for atom in pauli_str:
                monomial = _pauli_atom_dict[atom] @ monomial
            monomial *= coef.real
            pauli_string += monomial
        return pauli_string

    @staticmethod
    def _from_braket(
        pauli: BraketObservable,
    ) -> PauliString:
        from braket.circuits.observables import I as Braket_I
        from braket.circuits.observables import Sum as BraketSum
        from braket.circuits.observables import TensorProduct
        from braket.circuits.observables import X as Braket_X
        from braket.circuits.observables import Y as Braket_Y
        from braket.circuits.observables import Z as Braket_Z

        def tensor_product_to_pauli_sting(pauli: TensorProduct):
            monomial = PauliStringMonomial()
            for atom in pauli.factors:
                monomial @= _pauli_atom_dict[atom.ascii_symbols[0]]
            monomial *= pauli.coefficient
            return monomial

        if isinstance(pauli, BraketSum):
            pauli_str = PauliString()
            for tensor_product in pauli.summands:
                if isinstance(tensor_product, TensorProduct):
                    pauli_str += tensor_product_to_pauli_sting(tensor_product)
                elif isinstance(
                    tensor_product, (Braket_I, Braket_X, Braket_Y, Braket_Z)
                ):
                    pauli_str += _pauli_atom_dict[tensor_product.ascii_symbols[0]]
                else:
                    raise NotImplementedError(
                        f"Unsupported input type: {type(tensor_product)}."
                    )
            return pauli_str
        elif isinstance(pauli, TensorProduct):
            return tensor_product_to_pauli_sting(pauli)
        elif isinstance(pauli, (Braket_I, Braket_X, Braket_Y, Braket_Z)):
            return _pauli_atom_dict[pauli.ascii_symbols[0]]
        else:
            raise NotImplementedError(f"Unsupported input type: {type(pauli)}.")

    @staticmethod
    def _from_my_qml(
        pauli: Term,
        min_dimension: int = 0,
    ) -> PauliStringMonomial:
        min_dimension = (
            max(min_dimension, max(pauli.qbits) + 1)
            if len(pauli.qbits) > 0
            else min_dimension
        )
        monomial = [I] * min_dimension
        for index, atom in enumerate(pauli.op):
            monomial[pauli.qbits[index]] = _pauli_atom_dict[atom]
        return PauliStringMonomial(pauli.coeff, monomial)

    @staticmethod
    def from_other_language(
        pauli: Union[
            SparsePauliOp,
            BraketObservable,
            TensorProduct,
            list[Term],
            Term,
            CirqPauliSum,
            CirqPauliString,
            list[CirqPauliString],
            CirqGateOperation,
        ],
        min_dimension: int = 1,
    ) -> PauliString | list[PauliString]:
        """Convert pauli objects from other quantum SDKs to :class:`PauliString`.

        args:
            pauli: The pauli object(s) to be converted.
            min_dimension: Minimal dimension of the resulting Pauli string.

        Returns:
            The converted :class:`PauliString`. If the input is a list, the
            output will be a list of :class:`PauliString`.

        Examples:
            >>> from cirq import LineQubit, PauliSum, X as Cirq_X, Y as Cirq_Y, Z as Cirq_Z
            >>> a, b, c = LineQubit.range(3)
            >>> cirq_ps = 0.5 * Cirq_Z(a) * 0.5 * Cirq_Y(b) + 2 * Cirq_X(c)
            >>> PauliString.from_other_language(cirq_ps)
            0.25*Z@Y@I + 2.0*I@I@X

            >>> from braket.circuits.observables import (
            ...     Sum as BraketSum,
            ...     I as Braket_I,
            ...     X as Braket_X,
            ...     Y as Braket_Y,
            ...     Z as Braket_Z,
            ... )
            >>> braket_ps = 0.25 * Braket_Z() @ Braket_Y() @ Braket_I() + 2 * Braket_I() @ Braket_I() @ Braket_X()
            >>> PauliString.from_other_language(braket_ps)
            0.25*Z@Y@I + 2*I@I@X

            >>> from qiskit.quantum_info import SparsePauliOp
            >>> qiskit_ps = SparsePauliOp(["IYZ", "XII"], coeffs=[0.25 + 0.0j, 2.0 + 0.0j])
            >>> PauliString.from_other_language(qiskit_ps)
            0.25*Z@Y@I + 2.0*I@I@X

            >>> from qat.core.wrappers.observable import Term
            >>> my_qml_ps = [Term(0.25, "ZY", [0, 1]), Term(2, "X", [2])]
            >>> PauliString.from_other_language(my_qml_ps)
            0.25*Z@Y@I + 2*I@I@X

        """
        if isinstance(pauli, list) and any(
            not isinstance(p, type(pauli[0])) for p in pauli
        ):
            raise ValueError(
                "Cannot parse non-homogeneous types when `pauli` is a `list`."
            )

        from braket.circuits.observables import Observable as BraketObservable
        from cirq.ops.gate_operation import GateOperation as CirqGateOperation
        from cirq.ops.linear_combinations import PauliSum as CirqPauliSum
        from cirq.ops.pauli_string import PauliString as CirqPauliString
        from qat.core.wrappers.observable import Term
        from qiskit.quantum_info import SparsePauliOp

        if isinstance(pauli, SparsePauliOp):
            return PauliString._from_qiskit(pauli)
        elif isinstance(pauli, BraketObservable):
            return PauliString._from_braket(pauli)
        elif isinstance(pauli, Term):
            return PauliString._from_my_qml(pauli)
        elif isinstance(pauli, list) and isinstance(pauli[0], Term):
            for term in pauli:
                min_dimension = (
                    max(max(term.qbits) + 1, min_dimension)
                    if len(term.qbits) > 0
                    else min_dimension
                )
            pauli_string = PauliString()
            for term in pauli:
                pauli_string += PauliString._from_my_qml(term, min_dimension)
            return pauli_string
        elif isinstance(pauli, (CirqPauliSum, CirqPauliString, CirqGateOperation)):
            return PauliString._from_cirq(pauli, min_dimension)
        elif isinstance(pauli, list) and isinstance(pauli[0], CirqPauliString):
            min_dimension = max(
                max(map(PauliString._get_dimension_cirq_pauli, pauli)), min_dimension
            )
            return [
                PauliString._from_cirq(pauli_mono, min_dimension)
                for pauli_mono in pauli
            ]

        raise NotImplementedError(f"Unsupported input type: {type(pauli)}.")

    def to_other_language(
        self, language: Language, circuit: Optional[CirqCircuit] = None
    ) -> Union[
        SparsePauliOp,
        BraketSum,
        list[Term],
        Term,
        CirqPauliSum,
        CirqPauliString,
        list[CirqPauliString],
    ]:
        """Converts the pauli string to pauli string of another quantum
        programming language.

        Args:
            language: The target programming language.
            circuit: The Cirq circuit associated with the pauli string (required
                for ``cirq``).

        Returns:
            Depends on the target language.

        Example:
            >>> from mpqp.measures import I, X, Y, Z
            >>> ps = X @ X @ I + I @ Y @ I + I @ I @ Z
            >>> print(ps.to_other_language(Language.CIRQ))
            1.000*X(q(0))*X(q(1))+1.000*Y(q(1))+1.000*Z(q(2))
            >>> for term in ps.to_other_language(Language.MY_QLM):
            ...     print(term.op, term.qbits)
            XX [0, 1]
            Y [1]
            Z [2]
            >>> print(ps.to_other_language(Language.QISKIT))
            SparsePauliOp(['IXX', 'IYI', 'ZII'],
                          coeffs=[1.+0.j, 1.+0.j, 1.+0.j])
            >>> for tensor in ps.to_other_language(Language.BRAKET).summands:
            ...     print(tensor.coefficient, "".join(a.name for a in tensor.factors))
            1 XXI
            1 IYI
            1 IIZ

        """

        if language == Language.QISKIT:
            from qiskit.quantum_info import SparsePauliOp

            pauli_string = []
            pauli_string_coef = []
            for mono in self.monomials:
                pauli_string.append(
                    "".join(atom.label for atom in reversed(mono.atoms))
                )
                pauli_string_coef.append(mono.coef)
            return SparsePauliOp(pauli_string, np.array(pauli_string_coef))
        elif language == Language.MY_QLM:
            return [mono.to_other_language(language) for mono in self.monomials]
        elif language == Language.BRAKET:
            pauli_string = None
            for mono in self.monomials:
                braket_mono = mono.to_other_language(Language.BRAKET)
                pauli_string = (
                    pauli_string + braket_mono
                    if pauli_string is not None
                    else braket_mono
                )
            return pauli_string
        elif language == Language.CIRQ:
            cirq_pauli_string = None
            for monomial in self.monomials:
                cirq_monomial = monomial.to_other_language(language, circuit)
                cirq_pauli_string = (
                    cirq_monomial
                    if cirq_pauli_string is None
                    else cirq_pauli_string + cirq_monomial
                )

            return cirq_pauli_string
        else:
            raise NotImplementedError(f"Unsupported language: {language}")

    def to_dict(self) -> dict[str, str]:
        """Converts the Pauli string to a dictionary representation with the
        keys being the Pauli monomials and the values the corresponding
        coefficients.

        Returns:
            Mapping representing the Pauli string.

        Example:
            >>> from mpqp.measures import I, X, Y, Z
            >>> (1 * I @ Z + 2 * I @ I).to_dict()
            {'II': '2', 'IZ': '1'}

        """
        from sympy import simplify

        self = self.simplify()
        result_dict = {}
        for mono in self.monomials:
            atom_str = "".join(str(atom) for atom in mono.atoms)
            if atom_str not in result_dict:
                result_dict[atom_str] = mono.coef
            else:
                result_dict[atom_str] += mono.coef
        return {k: str(simplify(result_dict[k])) for k in sorted(result_dict)}

    def __hash__(self):
        monomials_as_tuples = tuple(
            tuple((atom.label for atom in mono.atoms) for mono in self.monomials)
        )
        return hash(monomials_as_tuples)

    def is_diagonal(self) -> bool:
        """Checks whether this pauli string has a diagonal representation, by checking if only ``I`` and ``Z`` Pauli
        operators appears in the monomials of the string.

        Returns:
            True if the observable represented by pauli string is diagonal.

        Examples:
            >>> from mpqp.measures import I, X, Y, Z
            >>> (I @ X + Z @ Y - Y @ X).is_diagonal()
            False
            >>> (I @ Z @ I - 2* Z @ Z @ Z + I @ I @ I).is_diagonal()
            True

        """

        return all([all([a == I or a == Z for a in m.atoms]) for m in self.monomials])


@typechecked
class PauliStringMonomial(PauliString):
    """Represents a monomial in a Pauli string, consisting of a coefficient and
    a list of PauliStringAtom objects.

    Args:
        coef: The coefficient of the monomial.
        atoms: The list of PauliStringAtom objects forming the monomial.
    """

    def __init__(self, coef: Coef = 1, atoms: Optional[list["PauliStringAtom"]] = None):
        self.coef = coef
        """Coefficient of the monomial."""
        self._atoms = [] if atoms is None else atoms

    @property
    def atoms(self) -> list["PauliStringAtom"]:
        """The list of atoms in the monomial."""
        return self._atoms

    @property
    def nb_qubits(self) -> int:
        return len(self.atoms)

    @property
    def monomials(self) -> list["PauliStringMonomial"]:
        return [PauliStringMonomial(self.coef, self.atoms)]

    @property
    def positive_eigen_values(self) -> npt.NDArray[np.bool_]:
        """Return the eigen values associated with this Pauli monomial,
        stored as booleans for efficiency. True refers to +1 and False to -1.

        Example:
            >>> from mpqp.measures import I, X, Y
            >>> (X @ Y @ I).positive_eigen_values  # doctest: +NORMALIZE_WHITESPACE
            array([ True, True, False, False, False, False, True, True])
        """
        eigvals = reduce(np.kron, [a.eigen_values for a in self.atoms])
        return eigvals > 0

    def __str__(self):
        coef = format_element(self.coef)
        if isinstance(coef, Expr):
            coef = f'({str(self.coef)})*'
        else:
            coef = f"{self.coef}*" if coef != 1 else ""
        return f"{coef}{'@'.join(map(str,self.atoms))}"

    def __repr__(self):
        return str(self)

    def to_matrix(self) -> Matrix:
        return (
            reduce(
                np.kron,
                map(lambda a: a.to_matrix(), self.atoms),
                np.eye(1, dtype=np.complex64).tolist(),
            )
            * self.coef
        )

    def __iadd__(self, other: "PauliString"):
        for mono in other.monomials:
            if (
                len(self.monomials) != 0
                and mono.nb_qubits != self.monomials[0].nb_qubits
            ):
                raise ValueError(
                    f"Non homogeneous sizes for given PauliStrings: {(self, other)}"
                )
        res = PauliString([self])
        res.monomials.extend(deepcopy(other.monomials))
        return res

    def __add__(self, other: "PauliString") -> PauliString:
        res = deepcopy(self)
        res += other
        return res

    def __imul__(self, other: Coef) -> PauliStringMonomial:
        self.coef *= other  # pyright: ignore[reportOperatorIssue]
        return self

    def __mul__(self, other: Coef) -> PauliStringMonomial:
        res = deepcopy(self)
        res *= other
        return res

    def __itruediv__(self, other: Coef) -> PauliStringMonomial:
        self.coef /= other  # pyright: ignore[reportOperatorIssue]
        return self

    def __truediv__(self, other: Coef) -> PauliStringMonomial:
        res = deepcopy(self)
        res /= other
        return res

    def __imatmul__(self, other: PauliString) -> PauliString:
        if isinstance(other, PauliStringAtom):
            self.atoms.append(other)
            return self
        elif isinstance(other, PauliStringMonomial):
            self.coef *= other.coef  # pyright: ignore[reportOperatorIssue]
            self.atoms.extend(other.atoms)
            return self
        else:
            res = deepcopy(other)
            res._monomials = [  # pyright: ignore[reportAttributeAccessIssue]
                self @ mono for mono in other.monomials
            ]
            return res

    def __matmul__(self, other: PauliString):
        res = deepcopy(self)
        res @= other
        return res

    def simplify(self, inplace: bool = False):
        return deepcopy(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PauliStringMonomial):
            for a1, a2 in zip(self.atoms, other.atoms):
                if a1 != a2:
                    return False
            return bool(self.coef == other.coef)
        return super().__eq__(other)

    def __hash__(self):
        atoms_as_tuples = tuple((atom.label for atom in self.atoms))
        return hash(atoms_as_tuples)

    def commutes_with(self, other: PauliString) -> bool:
        """Checks wether this Pauli monomial commutes (full commutativity) with the Pauli monomial in parameter. This
        is done by checking that the number of anti-commuting atoms is even.

        Args:
            other: The Pauli monomial for which we want to check commutativity with this monomial.

        Returns:
            True if this Pauli monomial commutes with the one in parameter.

        Examples:
            >>> from mpqp.measures import I, X, Y, Z
            >>> (I @ X @ Y).commutes_with(Z @ Y @ X)
            True
            >>> (X @ Z @ Z).commutes_with(Y @ Z @ I)
            False
            >>> (X @ Z @ Z).commutes_with(X @ Z @ Z)
            True

        """
        if not isinstance(other, PauliStringMonomial):
            raise NotImplementedError(
                f"Commutativity checking is only implemented for PauliStringMonomial in the current version."
            )

        if self.nb_qubits != other.nb_qubits:
            raise NumberQubitsError(
                f"Mismatch between {self.nb_qubits=} and {other.nb_qubits=}."
            )

        return (
            sum(1 for a, b in zip(self.atoms, other.atoms) if not a.commutes_with(b))
            % 2
            == 0
        )

    def subs(
        self, values: dict[Expr | str, Real], remove_symbolic: bool = True
    ) -> PauliStringMonomial:
        r"""Substitutes the parameters of the instruction with complex values.
        Optionally, also removes all symbolic variables such as `\pi` (needed for
        circuit execution, for example).

        Since we use ``sympy`` for gate parameters, ``values`` can in fact be
        anything the ``subs`` method from ``sympy`` would accept.

        Args:
            values: Mapping between the variables and the replacing values.
            remove_symbolic: Whether symbolic values should be replaced by their
                numeric counterparts.

        Returns:
            The circuit with the replaced parameters.

        Example:
            >>> from mpqp.measures import I, X, Y, Z
            >>> theta = symbols("θ")
            >>> print((theta * I @ X).subs({theta: np.pi}))
            3.141592653589793*I@X
        """

        new_monomial = deepcopy(self)
        caster = lambda v: format_element(v) if remove_symbolic else v
        if isinstance(new_monomial.coef, Expr):
            new_monomial.coef = caster(new_monomial.coef.subs(values))

        return new_monomial

    def to_other_language(
        self, language: Language, circuit: Optional[CirqCircuit] = None
    ):
        if language == Language.QISKIT:
            from qiskit.quantum_info import SparsePauliOp

            pauli_mono_str = "".join(atom.label for atom in reversed(self.atoms))
            return SparsePauliOp(pauli_mono_str, np.array(self.coef))
        elif language == Language.MY_QLM:
            from qat.core.wrappers.observable import Term

            pauli_mono_str = "".join(atom.label for atom in self.atoms)
            return Term(self.coef, pauli_mono_str, list(range(len(pauli_mono_str))))
        elif language == Language.BRAKET:
            braket_atoms: list[BraketObservable] = [
                atom.to_other_language(Language.BRAKET)
                for atom in self.atoms  # pyright: ignore[reportAssignmentType]
            ]

            return self.coef * reduce(matmul, braket_atoms)
        elif language == Language.CIRQ:
            from cirq.devices.line_qubit import LineQubit

            if TYPE_CHECKING:
                from cirq.ops.identity import IdentityGate as CirqI
                from cirq.ops.pauli_gates import Pauli as CirqPauli
                from cirq.ops.pauli_string import (
                    PauliString as CirqPauliString,  # pyright: ignore[reportUnusedImport]
                )

            all_qubits = (
                LineQubit.range(self.nb_qubits)
                if circuit is None
                else sorted(
                    set(
                        qubit
                        for moment in circuit
                        for op in moment.operations
                        for qubit in op.qubits
                    )
                )
            )

            cirq_atoms: list[Union[CirqPauli, CirqI, CirqPauliString]] = [
                atom.to_other_language(Language.CIRQ, target=all_qubits[index])
                for index, atom in enumerate(self.atoms)
            ]

            return reduce(mul, cirq_atoms) * self.coef
        else:
            raise NotImplementedError(f"Unsupported language: {language}")


@typechecked
class PauliStringAtom(PauliStringMonomial):
    """Represents a single Pauli operator acting on a qubit in a Pauli string.

    Args:
        label: The label representing the Pauli operator.
        matrix: The matrix representation of the Pauli operator.
        eig_values: The eigenvalues associated with the Pauli operator.
        eig_vectors: A list of eigenvectors associated with the Pauli operator.
        basis_change: Basis change of the measurement associated with the Pauli operator.

    Raises:
        RuntimeError: New atoms cannot be created, you should use the available
            ones.

    Note:
        All the atoms are already initialized. Available atoms are (``I``,
        ``X``, ``Y``, ``Z``).
    """

    __is_mutable = True

    def __init__(
        self,
        label: str,
        matrix: npt.NDArray[np.complex64],
        eig_values: list[int],
        eig_vectors: npt.NDArray[np.complex64],
        basis_change: list[type[NativeGate]],
    ):
        if _allow_atom_creation:
            self.label = label
            self.matrix = matrix
            self._eig_vals = np.array(eig_values)
            self._eig_vecs = np.array(eig_vectors)
            self._basis_change = basis_change
            self.__is_mutable = False
        else:
            raise RuntimeError(
                "New atoms cannot be created, just use the given `I`, `X`, `Y` "
                "and `Z`"
            )

    @property
    def nb_qubits(self) -> int:
        return 1

    @property
    def atoms(self) -> list["PauliStringAtom"]:
        """Atoms present. (needed for upward compatibility with
        :class:`PauliStringMonomial`)"""
        return [self]

    @property
    def coef(self):
        """Coefficient of the monomial."""
        return 1

    @property
    def monomials(self):
        return [PauliStringMonomial(self.coef, [a for a in self.atoms])]

    @property
    def eigen_values(self) -> npt.NDArray[np.bool_]:
        return self._eig_vals

    def __setattr__(self, name: str, value: Any):
        if not self.__is_mutable:
            raise AttributeError("This object is immutable")
        super().__setattr__(name, value)

    def __str__(self):
        return self.label

    def __repr__(self):
        return str(self)

    def __itruediv__(self, other: Coef) -> PauliStringMonomial:
        self = self / other
        return self

    def __truediv__(self, other: Coef) -> PauliStringMonomial:
        return PauliStringMonomial(
            1 / other,  # pyright: ignore[reportOperatorIssue]
            [self],
        )

    def __imul__(self, other: Coef) -> PauliStringMonomial:
        self = self * other
        return self

    def __mul__(self, other: Coef) -> PauliStringMonomial:
        return PauliStringMonomial(other, [self])

    def __rmul__(self, other: Coef) -> PauliStringMonomial:
        return PauliStringMonomial(other, [self])

    def __imatmul__(self, other: PauliString) -> PauliString:
        res = (
            PauliStringMonomial(1, [other])
            if isinstance(other, PauliStringAtom)
            else deepcopy(other)
        )
        if isinstance(res, PauliStringMonomial):
            res.atoms.insert(0, self)
        else:
            for i, mono in enumerate(res.monomials):
                res.monomials[i] = PauliStringMonomial(mono.coef, mono.atoms)
                res.monomials[i].atoms.insert(0, self)
        return res

    def __matmul__(self, other: PauliString):
        res = deepcopy(self)
        res @= other
        return res

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PauliStringAtom):
            return self.label == other.label
        else:
            return super().__eq__(other)

    def __hash__(self):
        return hash(self.label)

    def to_matrix(self) -> npt.NDArray[np.complex64]:
        return self.matrix

    def commutes_with(self, other: PauliString) -> bool:
        """Determines whether this single-qubit Pauli operator commutes with the one in parameter. In this case, this
        can be done by simply checking if one of the atoms are identity, or if they are the same.

        Args:
            other: The single-qubit Pauli operator with which want to check the commutativity

        Returns:
            True if the atoms commute, False otherwise.

        Examples:
            >>> from mpqp.measures import I, X, Y, Z
            >>> X.commutes_with(X)
            True
            >>> X.commutes_with(Y)
            False
            >>> X.commutes_with(I)
            True
            >>> Y.commutes_with(Z)
            False
            >>> I.commutes_with(Z)
            True
        """
        if not isinstance(other, PauliStringAtom):
            raise ValueError(
                f"Expected a PauliStringAtom in parameter but got {type(other).__name__}"
            )
        return other == I or self == I or self == other

    def to_other_language(
        self,
        language: Language,
        circuit: Optional[CirqCircuit] = None,
        target: Optional[Qid] = None,
    ):
        if language == Language.QISKIT:
            from qiskit.quantum_info import SparsePauliOp

            return SparsePauliOp(self.label)
        elif language == Language.MY_QLM:
            from qat.core.wrappers.observable import Term

            return Term(1.0, self.label, [0])
        elif language == Language.BRAKET:
            from braket.circuits.observables import I as Braket_I
            from braket.circuits.observables import X as Braket_X
            from braket.circuits.observables import Y as Braket_Y
            from braket.circuits.observables import Z as Braket_Z

            pauli_gate_map = {
                "I": Braket_I(),
                "X": Braket_X(),
                "Y": Braket_Y(),
                "Z": Braket_Z(),
            }
            return pauli_gate_map[self.label]
        elif language == Language.CIRQ:
            from cirq.devices.line_qubit import LineQubit
            from cirq.ops.identity import I as Cirq_I
            from cirq.ops.pauli_gates import X as Cirq_X
            from cirq.ops.pauli_gates import Y as Cirq_Y
            from cirq.ops.pauli_gates import Z as Cirq_Z

            pauli_gate_map = {
                "I": Cirq_I,
                "X": Cirq_X,
                "Y": Cirq_Y,
                "Z": Cirq_Z,
            }
            return pauli_gate_map[self.label](
                LineQubit(0) if target is None else target
            )
        else:
            raise NotImplementedError(f"Unsupported language: {language}")


_allow_atom_creation = True

I = PauliStringAtom(
    "I", np.eye(2, dtype=np.complex64), [1, 1], np.array([[1, 0], [0, 1]]), []
)
r"""Pauli-I atom representing the identity operator in a Pauli monomial or string.
Matrix representation:
`\begin{pmatrix}1&0\\0&1\end{pmatrix}`
"""
X = PauliStringAtom(
    "X",
    1 - np.eye(2, dtype=np.complex64),
    [1, -1],
    (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
    [H],
)
r"""Pauli-X atom representing the X operator in a Pauli monomial or string.
Matrix representation:
`\begin{pmatrix}0&1\\1&0\end{pmatrix}`

"""
Y = PauliStringAtom(
    "Y",
    np.fliplr(np.diag([1j, -1j])),
    [1, -1],
    (1 / np.sqrt(2)) * np.array([[1, 1j], [1, -1j]]),
    [S_dagger, H],
)
r"""Pauli-Y atom representing the Y operator in a Pauli monomial or string.
Matrix representation:
`\begin{pmatrix}0&-i\\i&0\end{pmatrix}`
"""
Z = PauliStringAtom("Z", np.diag([1, -1]), [1, -1], np.array([[1, 0], [0, 1]]), [])
r"""Pauli-Z atom representing the Z operator in a Pauli monomial or string.
Matrix representation:
`\begin{pmatrix}1&0\\0&-1\end{pmatrix}`
"""

_pauli_atom_dict = {"I": I, "X": X, "Y": Y, "Z": Z}
_allow_atom_creation = False
