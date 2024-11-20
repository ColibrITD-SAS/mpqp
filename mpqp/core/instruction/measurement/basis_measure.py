"""The :class:`BasisMeasure` is used to project the state on a statevector of a
given :class:`Basis` and returns the corresponding eigenvalue."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit.circuit import Parameter

from mpqp.core.languages import Language

from .basis import Basis, ComputationalBasis, VariableSizeBasis
from .measure import Measure


@typechecked
class BasisMeasure(Measure):
    """Class representing a measure of one or several qubits in a specific
    basis.

    By default, the computational basis will be used. The user can also precise
    a specific basis using the class Basis.

    The number of shots indicates the number of time the measure is repeated.
    When shots is equal to 0 (by default), the simulator is used to produce
    exact value of the amplitudes/probabilities. If you don't specify a target,
    the operation will apply to all qubits.

    Args:
        targets: List of indices referring to the qubits on which the measure
            will be applied. Defaults to the entire circuit for
            :class:`~mpqp.core.instruction.measurement.basis.VariableSizeBasis`
            and the first qubits matching the size of the basis for other basis.
        c_targets: List of indices referring to the classical bits on which the
            measure will be applied.
        shots: Number of shots to be performed basis: basis in which the qubits
            should be measured.
        basis: Basis in which the measure is performed. Defaults to
            :class:`~mpqp.core.instruction.measurement.basis.ComputationalBasis`
        label: Label used to identify the measure.

    Examples:
        >>> c1 = QCircuit([H(0), H(1), CNOT(0,1), BasisMeasure()])
        >>> c2 = QCircuit([
        ...     H(0),
        ...     H(2),
        ...     CNOT(0,1),
        ...     BasisMeasure([0, 1], shots=512, basis=HadamardBasis())
        ... ])

    """

    def __init__(
        self,
        targets: Optional[list[int]] = None,
        c_targets: Optional[list[int]] = None,
        shots: int = 1024,
        basis: Optional[Basis] = None,
        label: Optional[str] = None,
    ):

        if c_targets is not None:
            if len(set(c_targets)) != len(c_targets):
                raise ValueError(f"Duplicate registers in targets: {c_targets}")

        super().__init__(targets, shots, label)

        if basis is None:
            basis = ComputationalBasis()

        if not isinstance(basis, VariableSizeBasis):
            self._dynamic = False
            self.targets = list(range(basis.nb_qubits))

        self.user_set_c_targets = c_targets is not None
        self.c_targets = c_targets
        """See parameter description."""
        self.basis = basis
        """See parameter description."""

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if qiskit_parameters is None:
            qiskit_parameters = set()
        if language == Language.QISKIT:
            from qiskit.circuit import Measure

            return Measure()
        if language == Language.QASM2:
            if self.c_targets is None:
                return "\n".join(
                    f"measure q[{target}] -> c[{i}];"
                    for i, target in enumerate(self.targets)
                )
            else:
                return "\n".join(
                    f"measure q[{target}] -> c[{c_target}];"
                    for target, c_target in zip(self.targets, self.c_targets)
                )

        else:
            raise NotImplementedError(f"{language} is not supported")

    @property
    def pre_measure(self):
        return self.basis.to_computational()

    def __repr__(self) -> str:
        targets = (
            f"{self.targets}" if (not self._dynamic and len(self.targets)) != 0 else ""
        )
        options = ""
        if self.shots != 1024:
            options += f"shots={self.shots}"
        if not isinstance(self.basis, ComputationalBasis):
            options += (
                f", basis={self.basis}"
                if len(options) != 0 or len(targets) != 0
                else f"basis={self.basis}"
            )
        if self.label is not None:
            options += (
                f", label={self.label}"
                if len(options) != 0 or len(targets) != 0
                else f"label={self.label}"
            )
        separator = ", " if len(options) != 0 and len(targets) != 0 else ""
        return f"BasisMeasure({targets}{separator}{options})"
