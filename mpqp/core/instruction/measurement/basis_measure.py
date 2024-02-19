"""The :class:`BasisMeasure` is used to project the state on a statevector of a
given :class:`Basis` and returns the corresponding eigenvalue."""

from __future__ import annotations
from typing import Optional

import qiskit.circuit
from qiskit.circuit import Parameter
from typeguard import typechecked

from .measure import Measure
from .basis import Basis, ComputationalBasis
from mpqp.core.languages import Language


@typechecked
class BasisMeasure(Measure):
    """Class representing a measure of one or several qubits in a specific
    basis.

    By default, the computational basis will be used. The user can also precise
    a specific basis using the class Basis.

    The number of shots indicates the number of time the measure is repeated.
    When shots is equal to 0 (by default), the simulator is used to produce
    exact value of the amplitudes/probabilities.

    Args:
        targets: List of indices referring to the qubits on which the measure
            will be applied.
        c_targets: List of indices referring to the classical bits on which the
            measure will be applied.
        shots: Number of shots to be performed basis: basis in which the qubits
            should be measured.
        basis: Basis in which the measure is performed. Defaults to
            :class:`ComputationalBasis()<mpqp.core.instruction.measurement.basis.ComputationalBasis>`
        label: Label used to identify the measure.

    Examples:
        >>> c1 = QCircuit([H(0), H(1), CNOT(0,1)])
        >>> c1.add(BasisMeasure([0, 1, 2], shots=1024))
        >>> c2 = QCircuit([H(0), H(1), CNOT(0,1)])
        >>> c2.add(BasisMeasure([0, 1, 2], shots=1024, basis=HadamardBasis()))
    """

    def __init__(
        self,
        targets: list[int],
        c_targets: Optional[list[int]] = None,
        shots: int = 0,
        basis: Optional[Basis] = None,
        label: Optional[str] = None,
    ):
        if basis is None:
            basis = ComputationalBasis()
        # 6M-TODO: implement basis thing
        if c_targets is not None:
            if len(set(c_targets)) != len(c_targets):
                raise ValueError(f"Duplicate registers in targets: {c_targets}")
        super().__init__(targets, shots, label)
        self.user_set_c_targets = c_targets is not None
        self.c_targets = c_targets
        """See parameter description."""
        self.basis = basis
        """See parameter description."""

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        if qiskit_parameters is None:
            qiskit_parameters = set()
        if language == Language.QISKIT:
            if isinstance(self.basis, ComputationalBasis):
                return qiskit.circuit.Measure()
            else:
                raise NotImplementedError(f"{type(self.basis)} not handled")
        else:
            raise NotImplementedError(f"{language} is not supported")

    def __repr__(self) -> str:
        options = ""
        if self.shots != 0:
            options += f", shots={self.shots}"
        if not isinstance(self.basis, ComputationalBasis):
            options += f", basis={self.basis}"
        if self.label is not None:
            options += f", label={self.label}"
        return f"BasisMeasure({self.targets}{options})"
