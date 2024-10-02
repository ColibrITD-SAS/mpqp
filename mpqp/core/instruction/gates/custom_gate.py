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

    def to_matrix(self, desired_gate_size: int = 0):
        return self.matrix

    def to_canonical_matrix(self):
        return self.matrix

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QISKIT:
            from qiskit.quantum_info.operators import Operator as QiskitOperator

            if qiskit_parameters is None:
                qiskit_parameters = set()
            return QiskitOperator(self.matrix)
        elif language == Language.QASM2:
            import collections.abc

            from qiskit.qasm2.export import (
                _define_custom_operation,  # pyright: ignore[reportPrivateUsage]
                _instruction_call_site,  # pyright: ignore[reportPrivateUsage]
            )
            from qiskit.quantum_info.operators import Operator as QiskitOperator
            from qiskit.circuit import Instruction as QiskitInstruction
            from mpqp.qasm.open_qasm_2_and_3 import remove_user_gates

            gates_to_define: collections.OrderedDict[
                str, tuple[QiskitInstruction, str]
            ] = collections.OrderedDict()

            op = (
                QiskitOperator(self.matrix)
                .to_instruction()
                ._qasm2_decomposition()  # pyright: ignore[reportPrivateUsage]
            )
            _define_custom_operation(op, gates_to_define)

            gate_definitions_qasm = "\n".join(
                f"{qasm}" for _, qasm in gates_to_define.values()
            )

            qubits = ",".join([f"q[{j}]" for j in self.targets])

            qasm_str = remove_user_gates(
                "\n"
                + gate_definitions_qasm
                + "\n"
                + _instruction_call_site(op)
                + " "
                + qubits
                + ";"
            )

            return "\n" + qasm_str
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def decompose(self):
        """Returns the circuit made of native gates equivalent to this gate.

        3M-TODO refine this doc and implement
        """
        from mpqp.core.circuit import QCircuit

        return QCircuit(self.nb_qubits)
