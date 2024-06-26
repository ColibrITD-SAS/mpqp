"""Information about the state can be retrieved using the expectation value of
this state measured by an observable. This is done using the :class:`Observable`
class to define your observable, and a :class:`ExpectationMeasure` to perform
the measure."""

from __future__ import annotations

import copy
from numbers import Complex
from typing import TYPE_CHECKING, Optional
from warnings import warn

import numpy as np
from typeguard import typechecked

if TYPE_CHECKING:
    from sympy import Expr
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator
    from qat.core.wrappers.observable import Observable as QLMObservable
    from braket.circuits.observables import Hermitian
    from cirq.circuits.circuit import Circuit as Cirq_Circuit
    from cirq.ops.pauli_string import PauliString as CirqPauliString
    from cirq.ops.linear_combinations import PauliSum as CirqPauliSum

from mpqp.core.instruction.gates.native_gates import SWAP
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.core.languages import Language
from mpqp.tools.errors import NumberQubitsError
from mpqp.tools.generics import Matrix, one_lined_repr
from mpqp.tools.maths import is_hermitian


@typechecked
class Observable:
    """Class defining an observable, used for evaluating expectation values.

    An observable can be defined by using a Hermitian matrix, or using a
    combination of operators in a specific basis Pauli.

    Args:
        observable : can be either a Hermitian matrix representing the
            observable or PauliString representing the observable.

    Raises:
        ValueError: If the input matrix is not Hermitian or does not have a
            square shape.
        NumberQubitsError: If the number of qubits in the input observable does
            not match the number of target qubits.

    Example:
        >>> from mpqp.core.instruction.measurement.pauli_string import I, X, Y, Z
        >>> matrix = np.array([[1, 0], [0, -1]])
        >>> ps = 3 * I @ Z + 4 * X @ Y
        >>> obs = Observable(matrix)
        >>> obs2 = Observable(ps)

    """

    def __init__(self, observable: Matrix | PauliString):
        self._matrix = None
        self._pauli_string = None

        if isinstance(observable, PauliString):
            self.nb_qubits = observable.nb_qubits
            self._pauli_string = observable.simplify()
        else:
            self.nb_qubits = int(np.log2(len(observable)))
            """Number of qubits of this observable."""
            self._matrix = np.array(observable)

            basis_states = 2**self.nb_qubits
            if self.matrix.shape != (basis_states, basis_states):
                raise ValueError(
                    f"The size of the matrix {self.matrix.shape} doesn't neatly fit on a"
                    " quantum register. It should be a square matrix of size a power"
                    " of two."
                )

            if not is_hermitian(self.matrix):
                raise ValueError(
                    "The matrix in parameter is not hermitian (cannot define an observable)"
                )

    @property
    def matrix(self) -> Matrix:
        """The matrix representation of the observable."""
        if self._matrix is None:
            self._matrix = self.pauli_string.to_matrix()
        matrix = copy.deepcopy(self._matrix).astype(np.complex64)
        return matrix

    @property
    def pauli_string(self) -> PauliString:
        """The PauliString representation of the observable."""
        if self._pauli_string is None:
            self._pauli_string = PauliString.from_matrix(self.matrix)
        pauli_string = copy.deepcopy(self._pauli_string)
        return pauli_string

    @pauli_string.setter
    def pauli_string(self, pauli_string: PauliString):
        self._pauli_string = pauli_string
        self._matrix = None

    @matrix.setter
    def matrix(self, matrix: Matrix):
        self._matrix = matrix
        self._pauli_string = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({one_lined_repr(self.matrix)})"

    def __mult__(self, other: Expr | Complex) -> Observable:
        """3M-TODO"""
        ...

    def subs(
        self, values: dict[Expr | str, Complex], remove_symbolic: bool = False
    ) -> Observable:
        """3M-TODO"""
        ...

    def to_other_language(
        self, language: Language, circuit: Optional[Cirq_Circuit] = None
    ) -> Operator | QLMObservable | Hermitian | CirqPauliSum | CirqPauliString:
        """Converts the observable to the representation of another quantum
        programming language.

        Args:
            language: The target programming language.
            circuit: The Cirq circuit associated with the observable (required
                for ``cirq``).

        Returns:
            Depends on the target language.

        Example:
            >>> obs = Observable(np.diag([0.7, -1, 1, 1]))
            >>> obs_qiskit = obs.to_other_language(Language.QISKIT)
            >>> print(obs_qiskit)
            Operator([[ 0.69999999+0.j,  0.        +0.j,  0.        +0.j,
                        0.        +0.j],
                      [ 0.        +0.j, -1.        +0.j,  0.        +0.j,
                        0.        +0.j],
                      [ 0.        +0.j,  0.        +0.j,  1.        +0.j,
                        0.        +0.j],
                      [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                        1.        +0.j]],
                     input_dims=(2, 2), output_dims=(2, 2))

        """
        if language == Language.QISKIT:
            from qiskit.quantum_info import Operator

            return Operator(self.matrix)
        elif language == Language.MY_QLM:
            from qat.core.wrappers.observable import Observable as QLMObservable

            return QLMObservable(self.nb_qubits, matrix=self.matrix)
        elif language == Language.BRAKET:
            from braket.circuits.observables import Hermitian

            return Hermitian(self.matrix)
        elif language == Language.CIRQ:
            if circuit is None:
                raise ValueError("Circuit must be specified for cirq_observable.")

            from cirq.ops.identity import I as Cirq_I
            from cirq.ops.pauli_gates import X as Cirq_X
            from cirq.ops.pauli_gates import Y as Cirq_Y
            from cirq.ops.pauli_gates import Z as Cirq_Z

            all_qubits = sorted(
                set(
                    q
                    for moment in circuit
                    for op in moment.operations
                    for q in op.qubits
                )
            )

            pauli_gate_map = {"I": Cirq_I, "X": Cirq_X, "Y": Cirq_Y, "Z": Cirq_Z}

            cirq_pauli_string = None

            for monomial in self.pauli_string.monomials:
                cirq_monomial = None

                for index, atom in enumerate(monomial.atoms):
                    cirq_atom = pauli_gate_map[atom.label](all_qubits[index])
                    cirq_monomial = cirq_atom if cirq_monomial is None else cirq_monomial * cirq_atom  # type: ignore

                cirq_monomial *= monomial.coef  # type: ignore
                cirq_pauli_string = (
                    cirq_monomial
                    if cirq_pauli_string is None
                    else cirq_pauli_string + cirq_monomial
                )

            return cirq_pauli_string
        else:
            raise ValueError(f"Unsupported language: {language}")


@typechecked
class ExpectationMeasure(Measure):
    """This measure evaluates the expectation value of the output of the circuit
    measured by the observable given as input.

    If the ``targets`` are not sorted and contiguous, some additional swaps will
    be needed. This will affect the performance of your circuit if run on noisy
    hardware. The swaps added can be checked out in the ``pre_measure``
    attribute of the :class:`ExpectationMeasure`.

    Args:
        targets: List of indices referring to the qubits on which the measure
            will be applied.
        observable: Observable used for the measure.
        shots: Number of shots to be performed.
        label: Label used to identify the measure.

    Example:
        >>> obs = Observable(np.diag([0.7, -1, 1, 1]))
        >>> c = QCircuit([H(0), CNOT(0,1), ExpectationMeasure([0,1], observable=obs, shots=10000)])
        >>> run(c, ATOSDevice.MYQLM_PYLINALG).expectation_value # doctest: +SKIP
        0.85918

    Warns:
        UserWarning: If the ``targets`` are not sorted and contiguous, some
            additional swaps will be needed. This will change the performance of
            your circuit is run on noisy hardware.

    """

    def __init__(
        self,
        targets: list[int],
        observable: Observable,
        shots: int = 0,
        label: Optional[str] = None,
    ):
        from mpqp.core.circuit import QCircuit

        super().__init__(targets, shots, label)
        self.observable = observable
        """See parameter description."""
        # Raise an error if the number of target qubits does not match the size of the observable.
        if self.nb_qubits != observable.nb_qubits:
            raise NumberQubitsError(
                f"{self.nb_qubits}, the number of target qubit(s) doesn't match"
                f" {observable.nb_qubits}, the size of the observable"
            )

        self.pre_measure = QCircuit(max(targets) + 1)
        """Circuit added before the expectation measurement to correctly swap
        target qubits when their are note ordered or contiguous."""
        targets_is_ordered = all(
            [targets[i] > targets[i - 1] for i in range(1, len(targets))]
        )
        tweaked_tgt = copy.copy(targets)
        if (
            max(tweaked_tgt) - min(tweaked_tgt) + 1 != len(tweaked_tgt)
            or not targets_is_ordered
        ):
            warn(
                "Non contiguous or non sorted observable target will introduce "
                "additional CNOTs."
            )

            for t_index, target in enumerate(tweaked_tgt):  # sort the targets
                min_index = tweaked_tgt.index(min(tweaked_tgt[t_index:]))
                if t_index != min_index:
                    self.pre_measure.add(SWAP(target, tweaked_tgt[min_index]))
                    tweaked_tgt[t_index], tweaked_tgt[min_index] = (
                        tweaked_tgt[min_index],
                        target,
                    )
            for t_index, target in enumerate(tweaked_tgt):  # compact the targets
                if t_index == 0:
                    continue
                if target != tweaked_tgt[t_index - 1] + 1:
                    self.pre_measure.add(SWAP(target, tweaked_tgt[t_index - 1] + 1))
                    tweaked_tgt[t_index] = tweaked_tgt[t_index - 1] + 1
        self.rearranged_targets = tweaked_tgt
        """Adjusted list of target qubits when they are not initially sorted and
        contiguous."""

    def __repr__(self) -> str:
        return (
            f"ExpectationMeasure({self.targets}, {self.observable}, shots={self.shots})"
        )

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ) -> None:
        raise NotImplementedError(
            "This object should not be exported as is, because other SDKs have "
            "no equivalent. Instead, this object is used to store the "
            "appropriate data, and the data in later used in the needed "
            "locations."
        )
