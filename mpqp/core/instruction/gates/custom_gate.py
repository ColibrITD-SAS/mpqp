from typing import TYPE_CHECKING, Optional

from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit.circuit import Parameter

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.languages import Language


@typechecked
class CustomGate(Gate):
    """Custom gates allow you to define your own gates.

    Args:
        definition: The matrix (this is the only way supported to now) semantics of the gate.
        targets: The qubits on which the gate operates.
        label: The label of the gate. Defaults to None.
    """

    def __init__(
        self, definition: UnitaryMatrix, targets: list[int], label: Optional[str] = None
    ):
        self.matrix = definition.matrix
        """See parameter description."""
        super().__init__(targets, label)

    def to_matrix(self):
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

    def decompose(self):
        """Returns the circuit made of native gates equivalent to this gate.

        3M-TODO refine this doc and implement
        """
        from mpqp.core.circuit import QCircuit

        return QCircuit(self.nb_qubits)
