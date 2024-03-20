"""Information about the state can be retrieved using the expectation value of
this state measured by an observable. This is done using the :class:`Observable`
class to define your observable, and a :class:`ExpectationMeasure` to perform
the measure."""

from __future__ import annotations

import copy
from numbers import Complex
from typing import Optional
from warnings import warn

import numpy as np
from qiskit.circuit import Parameter
from sympy import Expr
from typeguard import typechecked

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

    An observable can be defined by using a hermitian matrix, or using a combination of operators in a specific
    basis (Kraus, Pauli, ...).

    For the moment, on can only define the observable using a matrix.

    Example:
        >>> matrix = np.array([[1, 0], [0, -1]])
        >>> obs = Observable(matrix)

    Args:
        matrix: Hermitian matrix representing the observable.
    """

    def __init__(self, observable: Matrix | PauliString):
        self.observable = observable
        self._matrix = None
        self._pauli_string = None

        if isinstance(observable, PauliString):
            self.nb_qubits = observable.nb_qubits
            # simplify pauli string
        else:
            self.nb_qubits = int(np.log2(len(observable)))
            """Number of qubits of this observable."""

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
    def matrix(self):
        if isinstance(self.observable, PauliString):
            return self.observable.to_matrix() if self._matrix is None else self._matrix
        else:
            return self.observable

    @property
    def pauli_string(self):
        if isinstance(self.observable, PauliString):
            return self.observable
        else:
            if self._pauli_string is not None:
                return self.paulistring
            else:
                # TODO transform to paulistring
                pauliString = PauliString()
                self.paulistring = pauliString
                return self.paulistring

    def __repr__(self) -> str:
        return f"{type(self).__name__}({one_lined_repr(self.observable)})"

    def __mult__(self, other: Expr | Complex) -> Observable:
        """3M-TODO"""
        ...

    def subs(
        self, values: dict[Expr | str, Complex], remove_symbolic: bool = False
    ) -> Observable:
        """3M-TODO"""
        ...

    def to_matrix(self) -> Matrix:
        return self.matrix

    def to_qiskit_observable(self):
        pass

    def to_myqlm_observable(self):
        pass

    def to_cirq_observable(self):
        pass

    def to_other_language(self, language: Language):
        if language == Language.QISKIT:
            return self.to_qiskit_observable
        elif language == Language.CIRQ:
            return self.to_cirq_observable
        elif language == Language.MY_QLM:
            return self.to_myqlm_observable


@typechecked
class ExpectationMeasure(Measure):
    """This measure evaluates the expectation value of the output of the circuit
    measured by the observable given as input.

    If the ``targets`` are not sorted and contiguous, some additional swaps will
    be needed. This will affect the performance of your circuit if run on noisy
    hardware. The swaps added can be checked out in the ``pre_measure``
    attribute of the :class:`ExpectationMeasure`.

    Example:
        >>> obs = Observable(np.diag([0.7, -1, 1, 1]))
        >>> c = QCircuit([H(0), CNOT(0,1), ExpectationMeasure([0,1], observable=obs, shots=10000)])
        >>> run(c, ATOSDevice.MYQLM_PYLINALG).expectation_value
        0.85918

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

    Note:
        In a future version, we would will also allow you to provide a
        ``PauliDecomposition``, a decomposition of the observable in the
        generalized Pauli basis.
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
        # TODO Check
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
        qiskit_parameters: Optional[set[Parameter]] = None,
    ) -> None:
        if qiskit_parameters is None:
            qiskit_parameters = set()
        if language == Language.QISKIT:
            raise NotImplementedError(
                "Qiskit does not implement these kind of measures"
            )
        else:
            raise NotImplementedError(
                "Only Qiskit supported for language export for now"
            )
