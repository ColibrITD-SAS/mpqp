from typing import Optional

from typeguard import typechecked

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.gate_definition import (
    UnitaryMatrix,
    KrausRepresentation,
    PauliDecomposition,
)
from mpqp.core.languages import Language
from qiskit.circuit import Parameter


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
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        if qiskit_parameters is None:
            qiskit_parameters = set()
        return super().to_other_language(language, qiskit_parameters)

    def decompose(self):
        """Returns the circuit made of native gates equivalent to this gate.

        6M-TODO refine this doc and implement
        """
        from mpqp.core.circuit import QCircuit

        return QCircuit(self.nb_qubits)


@typechecked
class KrausGate(CustomGate):
    """6M-TODO"""
    def __init__(
        self,
        definition: KrausRepresentation,
        targets: list[int],
        label: Optional[str] = None,
    ):
        self.kraus_representation = definition
        """See parameter description."""
        CustomGate.__init__(self, UnitaryMatrix(definition.to_matrix()), targets, label)


@typechecked
class PauliDecompositionGate(CustomGate):
    """6M-TODO"""
    def __init__(
        self,
        definition: PauliDecomposition,
        targets: list[int],
        label: Optional[str] = None,
    ):
        self.pauli_decomposition = definition
        """See parameter description."""
        CustomGate.__init__(self, UnitaryMatrix(definition.to_matrix()), targets, label)
