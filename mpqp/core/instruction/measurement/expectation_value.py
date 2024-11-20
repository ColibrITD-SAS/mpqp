"""Information about the state can be retrieved using the expectation value of
this state measured by an observable. This is done using the :class:`Observable`
class to define your observable, and a :class:`ExpectationMeasure` to perform
the measure."""

from __future__ import annotations

import copy
from numbers import Complex
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn

import numpy as np
from typeguard import typechecked

if TYPE_CHECKING:
    from sympy import Expr
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qat.core.wrappers.observable import Observable as QLMObservable
    from braket.circuits.observables import Hermitian
    from cirq.circuits.circuit import Circuit as CirqCircuit
    from cirq.ops.pauli_string import PauliString as CirqPauliString
    from cirq.ops.linear_combinations import PauliSum as CirqPauliSum

from mpqp.core.instruction.gates.native_gates import SWAP
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.core.languages import Language
from mpqp.tools.display import one_lined_repr
from mpqp.tools.errors import NumberQubitsError
from mpqp.tools.generics import Matrix
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

    Examples:
        >>> Observable(np.array([[1, 0], [0, -1]]))
        Observable(array([[ 1.+0.j, 0.+0.j], [ 0.+0.j, -1.+0.j]], dtype=complex64))

        >>> from mpqp.measures import I, X, Y, Z
        >>> Observable(3 * I @ Z + 4 * X @ Y)  # doctest: +NORMALIZE_WHITESPACE
        Observable(array([[ 3.+0.j,  0.+0.j, 0.+0.j,  0.+4.j],
                [ 0.+0.j, -3.+0.j, 0.-4.j,  0.+0.j],
                [ 0.+0.j,  0.+4.j, 3.+0.j,  0.+0.j],
                [ 0.-4.j,  0.+0.j, 0.+0.j, -3.+0.j]],
            dtype=complex64))
        >>> Observable(3 * I @ Z + 4 * X @ Y).pauli_string.sort_monomials()
        3*I@Z + 4*X@Y

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

    @matrix.setter
    def matrix(self, matrix: Matrix):
        self._matrix = matrix
        self._pauli_string = None

    @pauli_string.setter
    def pauli_string(self, pauli_string: PauliString):
        self._pauli_string = pauli_string
        self._matrix = None

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
        self, language: Language, circuit: Optional[CirqCircuit] = None
    ) -> Union[SparsePauliOp, QLMObservable, Hermitian, CirqPauliSum, CirqPauliString]:
        """Converts the observable to the representation of another quantum
        programming language.

        Args:
            language: The target programming language.
            circuit: The Cirq circuit associated with the observable (required
                if ``language == Language.CIRQ``).

        Returns:
            Depends on the target language.

        Example:
            >>> obs = Observable(np.diag([0.7, -1, 1, 1]))
            >>> obs_qiskit = obs.to_other_language(Language.QISKIT)
            >>> obs_qiskit.to_list()  # doctest: +NORMALIZE_WHITESPACE
            [('II', (0.42499999701976776+0j)), ('IZ', (0.42499999701976776+0j)),
             ('ZI', (-0.5750000029802322+0j)), ('ZZ', (0.42499999701976776+0j))]

        """
        if language == Language.QISKIT:
            from qiskit.quantum_info import Operator, SparsePauliOp

            return SparsePauliOp.from_operator(Operator(self.matrix))
        elif language == Language.MY_QLM:
            from qat.core.wrappers.observable import Observable as QLMObservable

            return QLMObservable(self.nb_qubits, matrix=self.matrix)
        elif language == Language.BRAKET:
            from braket.circuits.observables import Hermitian

            return Hermitian(self.matrix)
        elif language == Language.CIRQ:
            return self.pauli_string.to_other_language(Language.CIRQ, circuit)
        else:
            raise ValueError(f"Unsupported language: {language}")


@typechecked
class ExpectationMeasure(Measure):
    """This measure evaluates the expectation value of the output of the circuit
    measured by the observable given as input.

    If the ``targets`` are not sorted and contiguous, some additional swaps will
    be needed. This will affect the performance of your circuit if run on noisy
    hardware. The swaps added can be checked out in the :attr:`pre_measure`
    attribute.

    Args:
        targets: List of indices referring to the qubits on which the measure
            will be applied.
        observable: Observable used for the measure.
        shots: Number of shots to be performed.
        label: Label used to identify the measure.

    Warns:
        UserWarning: If the ``targets`` are not sorted and contiguous, some
            additional swaps will be needed. This will change the performance of
            your circuit is run on noisy hardware.

    Example:
        >>> obs = Observable(np.diag([0.7, -1, 1, 1]))
        >>> c = QCircuit([H(0), CNOT(0,1), ExpectationMeasure(obs, shots=10000)])
        >>> run(c, ATOSDevice.MYQLM_PYLINALG).expectation_value # doctest: +SKIP
        0.85918

    """

    # TODO: problem here with the doc generation, the arguments are messed up

    def __init__(
        self,
        observable: Observable,
        targets: Optional[list[int]] = None,
        shots: int = 0,
        label: Optional[str] = None,
    ):

        super().__init__(targets, shots, label)
        self.observable = observable
        """See parameter description."""
        self._check_targets_order()

    def _check_targets_order(self):
        """Ensures target qubits are ordered and contiguous, rearranging them if necessary (private)."""
        from mpqp.core.circuit import QCircuit

        if len(self.targets) == 0:
            self.pre_measure = QCircuit(0)
            return

        if self.nb_qubits != self.observable.nb_qubits:
            raise NumberQubitsError(
                f"Target size {self.nb_qubits} doesn't match observable size "
                f"{self.observable.nb_qubits}."
            )

        self.pre_measure = QCircuit(max(self.targets) + 1)
        """Circuit added before the expectation measurement to correctly swap
        target qubits when their are note ordered or contiguous."""
        targets_is_ordered = all(
            [self.targets[i] > self.targets[i - 1] for i in range(1, len(self.targets))]
        )
        tweaked_tgt = copy.copy(self.targets)
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
                    tweaked_tgt[t_index] = tweaked_tgt[min_index]
                    tweaked_tgt[min_index] = target
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
        targets = (
            f", {self.targets}"
            if (not self._dynamic and len(self.targets)) != 0
            else ""
        )
        shots = "" if self.shots == 0 else f", shots={self.shots}"
        label = "" if self.label is None else f", label={self.label}"
        return f"ExpectationMeasure({self.observable}{targets}{shots}{label})"

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ) -> None | str:
        raise NotImplementedError(
            "This object should not be exported as is, because other SDKs have "
            "no equivalent. Instead, this object is used to store the "
            "appropriate data, and the data in later used in the needed "
            "locations."
        )
