"""The :class:`BasisMeasure` is used to project the state on a statevector of a
given :class:`Basis` and returns the corresponding eigenvalue."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit.circuit import Parameter
    from mpqp import QCircuit

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
            if targets is None:
                raise ValueError(f"Missing targets for c_targets: {c_targets}")
            elif len(c_targets) != len(targets):
                raise ValueError(
                    f"Different number of targets and c_targets: targets={len(targets)}, c_targets={len(c_targets)}"
                )
            self._user_set_c_targets = True
        else:
            self._user_set_c_targets = False

        super().__init__(targets, shots, label)

        if basis is None:
            basis = ComputationalBasis()
        if (
            isinstance(basis, VariableSizeBasis)
            and basis._dynamic  # pyright: ignore[reportPrivateUsage]
        ):
            if targets is not None:
                basis.set_size(max(targets) + 1)
        else:
            self._dynamic = False
            if (
                len(self.targets) != 0
                and max(self.targets) - min(self.targets) + 1 != basis.nb_qubits
            ):
                raise ValueError(
                    f"Size mismatch between target and basis: target size is "
                    f"{max(self.targets)-min(self.targets) + 1} but basis size is {basis.nb_qubits}"
                )
            self.targets = list(range(basis.nb_qubits))

        self._user_set_c_targets = c_targets is not None
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
        elif language == Language.CIRQ:
            from cirq.ops.measurement_gate import MeasurementGate

            return MeasurementGate(num_qubits=self.nb_qubits)
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
    def pre_measure(self) -> QCircuit:
        return self.basis.to_computational()

    def __repr__(self) -> str:
        components = []
        if not self._dynamic and len(self.targets) != 0:
            components.append(str(self.targets))
        if not self._dynamic and self._user_set_c_targets:
            components.append(f"c_targets={self.c_targets}")
        if self.shots != 1024:
            components.append(f"shots={self.shots}")
        if self.label is not None:
            components.append(f"label='{self.label}'")
        if (
            not isinstance(self.basis, ComputationalBasis)
            or not self.basis._dynamic  # pyright: ignore[reportPrivateUsage]
        ):
            components.append(f"basis={self.basis}")

        return f"BasisMeasure({', '.join(components)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasisMeasure):
            return False
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        # TODO: can this be a bit more automatic ?
        return {
            "targets": self.targets,
            "c_targets": self.c_targets,
            "shots": self.shots,
            "basis": self.basis,
            "label": self.label,
            "_dynamic": self._dynamic,
        }
