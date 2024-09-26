"""In some cases, we need to manipulate unitary operations that are not defined using native gates (by the
corresponding unitary matrix for instance). We define a
:class:`CustomGate<mpqp.core.instruction.gates.custom_gate.CustomGate>` class allowing the user to add his custom
unitary operation to the circuit, that will be decomposed and executed transparently."""


from typing import TYPE_CHECKING, Optional

from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit.circuit import Parameter

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.languages import Language


@typechecked
class CustomGate(Gate):
    """Custom gates allow you to define your own unitary gates.

    Args:
        definition: The UnitaryMatrix (only way supported for now) description of the gate.
        targets: The qubits on which the gate operates.
        label: The label of the gate. Defaults to None.

    Raises:
        ValueError: the target qubits must be contiguous and in order, and must match the size of the UnitaryMatrix
    """

    def __init__(
        self, definition: UnitaryMatrix, targets: list[int], label: Optional[str] = None
    ):
        self.matrix = definition.matrix
        """See parameter description."""

        if definition.nb_qubits != len(targets):
            raise ValueError(f"Size of the targets ({len(targets)}) must match the number of qubits of the "
                             f"UnitaryMatrix ({definition.nb_qubits})")
        if not all([targets[i]+1==targets[i+1] for i in range(len(targets)-1)]):
            raise ValueError("Target qubits must be ordered and contiguous for a CustomGate.")

        # 3M-TODO: add later the possibility to give non-contiguous and/or non-ordered target qubits for CustomGate

        super().__init__(targets, label)

    def to_matrix(self):
        return self.matrix

    def to_canonical_matrix(self):
        return self.matrix

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        from qiskit.quantum_info.operators import Operator as QiskitOperator

        if qiskit_parameters is None:
            qiskit_parameters = set()
        return QiskitOperator(self.matrix)

    def __repr__(self):
        # TODO: fill the __repr__
        return ""

    def decompose(self):
        """Returns the circuit made of native gates equivalent to this gate.

        3M-TODO refine this doc and implement
        """
        raise NotImplementedError()
