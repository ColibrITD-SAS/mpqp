"""In some cases, we need to manipulate unitary operations that are not defined
using native gates (by the corresponding unitary matrix for instance). For those
cases, you can use :class:`mpqp.core.instruction.gates.custom_gate.CustomGate`
to add your custom unitary operation to the circuit, which will be decomposed
and executed transparently."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from typeguard import typechecked

from mpqp.tools import Matrix

if TYPE_CHECKING:
    from qiskit.circuit import Parameter
    from mpqp.core.circuit import QCircuit

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.languages import Language


@typechecked
class CustomGate(Gate):
    """Custom gates allow you to define your own unitary gates.

    Args:
        matrix: The matrix describing the gate.
        targets: The qubits on which the gate operates.
        label: The label of the gate. Defaults to None.

    Raises:
        ValueError: the target qubits must be contiguous and in order, and must
            match the size of the matrix

        ValueError: Target qubits must be ordered and contiguous for a CustomGate.


    Example:
        >>> u = np.array([[0,-1],[1,0]])
        >>> cg = CustomGate(u, [0])
        >>> print(run(QCircuit([X(0), cg]), IBMDevice.AER_SIMULATOR))
        Result: IBMDevice, AER_SIMULATOR
          State vector: [-1, 0]
          Probabilities: [1, 0]
          Number of qubits: 1

    """

    def __init__(
        self,
        matrix: Matrix,
        targets: Union[list[int], int],
        label: Optional[str] = None,
    ):
        definition = UnitaryMatrix(matrix)
        if isinstance(targets, int):
            targets = [targets]
        self.definition = definition
        """See parameter description."""
        if definition.nb_qubits != len(targets):
            raise ValueError(
                f"Size of the targets ({len(targets)}) must match the number of qubits of the "
                f"UnitaryMatrix ({definition.nb_qubits})"
            )

        # 3M-TODO: add later the possibility to give non-contiguous and/or non-ordered target qubits for CustomGate,
        #  use the to_matrix() method inherited from Gate, maybe

        super().__init__(targets, label)

    @property
    def matrix(self) -> Matrix:
        # TODO: move this to `to_canonical_matrix` and check for the usages
        return self.definition.matrix

    def to_matrix(self, desired_gate_size: int = 0):
        return self.matrix

    def to_canonical_matrix(self):
        return self.matrix

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
        printing: bool = False,
    ):
        if language == Language.QISKIT:
            from qiskit.quantum_info.operators import Operator as QiskitOperator
            from sympy import Expr

            if qiskit_parameters is None:
                qiskit_parameters = set()
            gate_symbols = set().union(
                *(
                    elt.free_symbols
                    for elt in self.matrix.flatten()
                    if isinstance(elt, Expr)
                )
            )
            from mpqp.core.instruction.gates.native_gates import (
                _qiskit_parameter_adder,  # pyright: ignore[reportPrivateUsage]
            )

            for symbol in gate_symbols:
                if TYPE_CHECKING:
                    assert isinstance(symbol, Expr)
                _qiskit_parameter_adder(symbol, qiskit_parameters)

            if len(gate_symbols) > 0:
                if not printing:
                    raise ValueError(
                        "Custom gates defined with symbolic variables cannot be"
                        " exported to qiskit."
                    )
                from qiskit import QuantumCircuit

                dummy_circuit = QuantumCircuit(self.nb_qubits)
                for param in qiskit_parameters:
                    # Rx is just a random choice so to have the parameter in the
                    # list of inputs
                    dummy_circuit.rx(param, 0)
                return dummy_circuit.to_gate(label="CustomGate")
            return QiskitOperator(self.matrix)
        elif language == Language.QASM2:
            from qiskit import QuantumCircuit, qasm2

            from mpqp.tools.circuit import replace_custom_gate

            nb_qubits = max(self.targets) + 1

            qiskit_circ = QuantumCircuit(nb_qubits)
            instr = self.to_other_language(Language.QISKIT)
            if TYPE_CHECKING:
                from qiskit.quantum_info.operators import Operator as QiskitOperator

                assert isinstance(instr, QiskitOperator)
            qiskit_circ.unitary(
                instr,
                list(reversed(self.targets)),  # dang qiskit qubits order
                self.label,
            )

            circuit, gphase = replace_custom_gate(qiskit_circ.data[0], nb_qubits)

            qasm_str = qasm2.dumps(circuit)
            qasm_lines = qasm_str.splitlines()

            instructions_only = [
                line
                for line in qasm_lines
                if not (
                    line.startswith("qreg")
                    or line.startswith("include")
                    or line.startswith("creg")
                    or line.startswith("OPENQASM")
                )
            ]
            return "\n".join(instructions_only), gphase
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def __repr__(self) -> str:
        label = f", \"{self.label}\"" if self.label else ""
        return f"CustomGate({repr(self.matrix)}, {self.targets}{label})"

    def decompose(self) -> "QCircuit":
        """Returns the circuit made of native gates equivalent to this gate.

        The circuit follows the quantum Shannon decomposition which decomposes any unitary matrix into Ry,Rz and CNOT gates.

        Example:
            >>> U = np.array([[0,1], [1,0]])
            >>> gate = CustomGate(U, [0])
            >>> print(gate.decompose()) # doctest: +NORMALIZE_WHITESPACE
               ┌─────────┐┌───────┐┌──────────┐
            q: ┤ Rz(π/2) ├┤ Ry(π) ├┤ Rz(-π/2) ├
               └─────────┘└───────┘└──────────┘

        """
        from mpqp.tools.unitary_decomposition import quantum_shannon_decomposition

        return quantum_shannon_decomposition(self.matrix)
