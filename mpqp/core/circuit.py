from __future__ import annotations

from typing import Iterable, Optional, Sequence, Type, Union
from copy import deepcopy
from numbers import Complex

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from qiskit.circuit import QuantumCircuit, Operation
from qiskit.circuit.quantumcircuit import CircuitInstruction
from qat.core.wrappers.circuit import Circuit as myQLM_Circuit
from braket.circuits import Circuit as braket_Circuit
from sympy import Basic, Expr
from typeguard import typechecked, TypeCheckError

from mpqp.core.instruction import Instruction
from mpqp.core.instruction.barrier import Barrier
from mpqp.core.instruction.gates import (
    ControlledGate,
    Gate,
)
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
from mpqp.core.instruction.measurement import (
    ComputationalBasis,
    BasisMeasure,
    Measure,
)
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.qasm import qasm2_to_myqlm_Circuit
from mpqp.qasm.open_qasm_2_and_3 import open_qasm_2_to_3
from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit
from mpqp.tools.errors import NumberQubitsError
from mpqp.core.languages import Language
from mpqp.tools.maths import matrix_eq


@typechecked
class QCircuit:
    """This class models a quantum circuit.

    A circuit is composed of quantum and classical bits and it is defined as a
    list of instructions applied on specific quantum and/or classical bits.

    Example:
        >>> circuit = QCircuit(2)
        >>> circuit.pretty_print()
        QCircuit : Size (Qubits,Cbits) = (2, 0), Nb instructions = 0
        q_0:
        q_1:
        >>> circuit = QCircuit(5, nb_cbits=2, label="Circuit 1")
        >>> circuit.add(Rx(1.23, 3))
        >>> circuit.pretty_print()
        QCircuit Circuit 1 : Size (Qubits,Cbits) = (5, 2), Nb instructions = 1
        q_0: ────────────
        q_1: ────────────
        q_2: ────────────
             ┌──────────┐
        q_3: ┤ Rx(1.23) ├
             └──────────┘
        q_4: ────────────

    Args:
        data: Number of qubits or List of instructions to initiate the circuit with. If the number of qubits is passed,
            it should be a positive int.
        nb_qubits: Optional number of qubits, in case you input the sequence of instruction and want to hardcode the
            number of qubits.
        nb_cbits: Number of classical bits. It should be positive. Defaults to None.
        label: Name of the circuit. Defaults to None.
    """

    def __init__(
        self,
        data: int | Sequence[Instruction],
        *,
        nb_qubits: Optional[int] = None,
        nb_cbits: Optional[int] = None,
        label: Optional[str] = None,
    ):
        self.nb_cbits = nb_cbits
        """See parameter description."""
        self.label = label
        """See parameter description."""
        self.instructions: list[Instruction] = []
        """List of instructions of the circuit."""
        self.nb_qubits: int
        """Number of qubits of the circuit."""

        if isinstance(data, int):
            if data < 0:
                raise TypeCheckError(
                    f"The data passed to QCircuit is a negative int ({data}), "
                    "this does not make sense."
                )
            self.nb_qubits = data
        else:
            if nb_qubits is None:
                if len(data) == 0:
                    self.nb_qubits = 0
                else:
                    connections: set[int] = set.union(
                        *(inst.connections() for inst in data)
                    )
                    self.nb_qubits = max(connections) + 1
            else:
                self.nb_qubits = nb_qubits
            self.add(map(deepcopy, data))

    def add(self, instruction: Instruction | Iterable[Instruction]):
        """Adds one instruction or a list of instructions at the end of the
        circuit.

        Args:
            instruction : Instruction(s) to append at the end of the circuit.

        Example:
            >>> circuit = QCircuit(2)
            >>> circuit.add(X(0))
            >>> circuit.add([CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> circuit.pretty_print()
            QCircuit : Size (Qubits,Cbits) = (2, 2), Nb instructions = 3
                 ┌───┐     ┌─┐
            q_0: ┤ X ├──■──┤M├───
                 └───┘┌─┴─┐└╥┘┌─┐
            q_1: ─────┤ X ├─╫─┤M├
                      └───┘ ║ └╥┘
            c: 2/═══════════╩══╩═
                            0  1
        """
        if isinstance(instruction, Iterable):
            for inst in instruction:
                self.add(inst)
            return

        if any(conn > self.nb_qubits for conn in instruction.connections()):
            raise NumberQubitsError(
                f"Instruction {type(instruction)}'s connections "
                f"({instruction.connections()}) are not compatible with circuit"
                f" size ({self.nb_qubits})."
            )
        if any(qb >= self.nb_qubits for qb in instruction.targets):
            raise NumberQubitsError("Instruction targets qubit outside of circuit")
        if isinstance(instruction, ControlledGate):
            if any(qb >= self.nb_qubits for qb in instruction.controls):
                raise NumberQubitsError("Control targets qubit outside of circuit")

        if isinstance(instruction, BasisMeasure) and instruction.c_targets is None:
            if self.nb_cbits is None:
                self.nb_cbits = 0
            instruction.c_targets = [
                self.nb_cbits + i for i in range(len(instruction.targets))
            ]
            self.nb_cbits += len(instruction.c_targets)

        if isinstance(instruction, Barrier):
            instruction.size = self.nb_qubits

        self.instructions.append(instruction)

    def append(self, other: QCircuit, qubits_offset: int = 0) -> None:
        """Appends the circuit at the end (right side) of this circuit, inplace.

        If the size of the ``other`` is smaller than this circuit,
        the parameter ``qubits_offset`` can be used to indicate at which qubit
        the ``other`` circuit must be added.

        Example:
            >>> c1 = QCircuit([CNOT(0,1),CNOT(1,2)])
            >>> c2 = QCircuit([X(1),CNOT(1,2)])
            >>> c1.append(c2)
            >>> print(c1)
            q_0: ──■─────────────────
                 ┌─┴─┐     ┌───┐
            q_1: ┤ X ├──■──┤ X ├──■──
                 └───┘┌─┴─┐└───┘┌─┴─┐
            q_2: ─────┤ X ├─────┤ X ├
                      └───┘     └───┘

        Args:
            other: The circuit to append at the end of this circuit.
            qubits_offset: If the circuit in parameter is smaller, this
                parameter precise at which qubit (vertically) the circuit will
                be added.

        Raises:
            NumberQubitsError: if the circuit in parameter is larger than this
                circuit or if the ``qubits_offset`` is too big such that the
                ``other`` circuit would "stick out".
        """

        if self.nb_qubits < other.nb_qubits:
            raise NumberQubitsError(
                "Size of the circuit to be appended is greater than the size of"
                " this circuit"
            )
        if qubits_offset + other.nb_qubits > self.nb_qubits:
            raise NumberQubitsError(
                "Size of the circuit to be appended is too large given the"
                " index and the size of this circuit"
            )

        for inst in deepcopy(other.instructions):
            inst.targets = [qubit + qubits_offset for qubit in inst.targets]
            if isinstance(inst, ControlledGate):
                inst.controls = [qubit + qubits_offset for qubit in inst.controls]
            if isinstance(inst, BasisMeasure):
                if not inst.user_set_c_targets:
                    inst.c_targets = None

            self.add(inst)

    def __iadd__(self, other: QCircuit):
        self.append(other)
        return self

    def __add__(self, other: QCircuit) -> QCircuit:
        res = deepcopy(self)
        res += other
        return res

    def tensor(self, other: QCircuit) -> QCircuit:
        """Computes the tensor product of this circuit with the one in parameter.

        In the circuit notation, the upper part of the output circuit will correspond
        to the first circuit, while the bottom part correspond to the one in
        parameter.

        Example:
            >>> c1 = QCircuit([CNOT(0,1),CNOT(1,2)])
            >>> c2 = QCircuit([X(1),CNOT(1,2)])
            >>> print(c1.tensor(c2))
            q_0: ──■───────
                 ┌─┴─┐
            q_1: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_2: ─────┤ X ├
                 ┌───┐└───┘
            q_3: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_4: ─────┤ X ├
                      └───┘

        Args:
            other: QCircuit being the second operand of the tensor product with this circuit.

        Returns:
            The QCircuit resulting from the tensor product of this circuit with the one in parameter.
        """
        res = deepcopy(self)
        res.nb_qubits += other.nb_qubits
        res.append(other, qubits_offset=self.nb_qubits)
        return res

    def __matmul__(self, other: QCircuit) -> QCircuit:
        return self.tensor(other)

    def display(self, output: str = "mpl"):
        r"""Displays this circuit in the desired output format.

        For now, this uses the qiskit circuit drawer, so all formats supported
        by qiskit are supported.

        Example:
            >>> theta = symbols("θ")
            >>> circ = QCircuit([
            ...     P(theta, 0),
            ...     ExpectationMeasure([0], Observable(np.array([[0, 1], [1, 0]])), shots=1000)
            ... ])
            >>> circ.display("text")
               ┌──────┐
            q: ┤ P(θ) ├
               └──────┘
            >>> circ.display("latex")
            \documentclass[border=2px]{standalone}
            \usepackage[braket, qm]{qcircuit}
            \usepackage{graphicx}
            \begin{document}
            \scalebox{1.0}{
            \Qcircuit @C=1.0em @R=0.2em @!R { \\
                \nghost{{q} :  } & \lstick{{q} :  } & \gate{\mathrm{P}\,(\mathrm{{\ensuremath{\theta}}})} & \qw & \qw\\
            \\ }}
            \end{document}
            
        Args:
            output: Format of the output, see
                `docs.quantum.ibm.com/build/circuit-visualization <https://docs.quantum.ibm.com/build/circuit-visualization#alternative-renderers>`_
                for more information.
        """
        from qiskit.tools.visualization import circuit_drawer

        qc = self.to_other_language(language=Language.QISKIT)
        fig = circuit_drawer(qc, output=output, style={"backgroundcolor": "#EEEEEE"})
        if isinstance(fig, Figure):
            fig.show()
        return fig

    def size(self) -> tuple[int, int]:
        """Provides the size of the circuit, in terms of number of quantum and
        classical bits.

        Examples:
            >>> c1 = QCircuit([CNOT(0,1),CNOT(1,2)])
            >>> c1.size()
            (3, 0)
            >>> c2 = QCircuit(3,nb_cbits=2)
            >>> c2.size()
            (3, 2)
            >>> c3 = QCircuit([CNOT(0,1),CNOT(1,2), BasisMeasure([0,1,2], shots=200)])
            >>> c3.size()
            (3, 3)

        Returns:
            A couple ``(q, c)`` of integers, with ``q`` the number of qubits,
            and ``c`` the number of cbits of this circuit.
        """
        return self.nb_qubits, (self.nb_cbits or 0)

    def depth(self) -> int:
        """Computes the depth of the circuit.

        Example:
            >>> QCircuit([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), X(2)]).depth()
            3
            >>> QCircuit([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), Barrier(), X(2)]).depth()
            4

        Returns:
            Depth of the circuit.
        """
        if len(self) == 0:
            return 0

        nb_qubits = self.nb_qubits
        instructions = self.without_measurements().instructions
        layers = np.zeros((nb_qubits, self.count_gates()), dtype=bool)

        current_layer = 0
        last_barrier = 0
        for instr in instructions:
            if isinstance(instr, Barrier):
                last_barrier = current_layer
                current_layer += 1
                continue
            conns = list(instr.connections())
            if any(layers[conns, current_layer]):
                current_layer += 1
            fitting_layer_index = current_layer
            for index in range(current_layer, last_barrier - 1, -1):
                if any(layers[conns, index]):
                    fitting_layer_index = index + 1
                    break
            layers[conns, fitting_layer_index] = [True] * len(conns)

        return current_layer + 1

    def __len__(self) -> int:
        """Returns the number of instructions added to this circuit.

        Example:
            >>> c1 = QCircuit([CNOT(0,1), CNOT(1,2), X(1), CNOT(1,2)])
            >>> len(c1)
            4

        Returns:
            An integer representing the number of instructions in this circuit.
        """
        return len(self.instructions)

    def is_equivalent(self, circuit: QCircuit) -> bool:
        """Whether the circuit in parameter is equivalent to this circuit, in
        terms of gates, but not measurements.

        Depending on the definition of the gates of the circuit, several methods
        could be used to do it in an optimized way.

        Example:
            >>> c1 = QCircuit([H(0), H(0)])
            >>> c2 = QCircuit([Rx(0, 0)])
            >>> c1.is_equivalent(c2)
            True

        Args:
            circuit: The circuit for which we want to know if it is equivalent
                to this circuit.

        Returns:
            ``True`` if the circuit in parameter is equivalent to this circuit

        3M-TODO: will only work once the circuit.to_matrix is implemented
        """
        return matrix_eq(self.to_matrix(), circuit.to_matrix())

    def optimize(self, criteria: Optional[str | list[str]] = None) -> QCircuit:
        """Optimize the circuit to satisfy some criteria (depth, number of
        qubits, gate restriction) in parameter.

        Examples:
            >>>
            >>>
            >>>

        Args:
            criteria: String, or list of strings, regrouping the criteria of optimization of the circuit.

        Returns:
            the optimized QCircuit
        # 6M-TODO implement, example and test, can be an internship
        """
        # ideas: a circuit can be optimized
        # - to reduce the depth of the circuit (combine gates, simplify some sequences)
        # - according to a given topology or qubits connectivity map
        # - to avoid the use of some gates (imperfect or more noisy)
        # - to avoid multi-qubit gates
        ...

    def to_matrix(self) -> npt.NDArray[np.complex64]:
        """
        Compute the unitary matrix associated to this circuit.

        Examples:
            >>> c = QCircuit([H(0), CNOT(0,1)])
            >>> c.to_matrix()
            array([[ 0.70710678,  0.        ,  0.70710678,  0.        ],
                   [ 0.        ,  0.70710678,  0.        ,  0.70710678],
                   [ 0.        ,  0.70710678,  0.        , -0.70710678],
                   [ 0.70710678,  0.        , -0.70710678,  0.        ]])
            >>>

        Returns:
            a unitary matrix representing this circuit

        # 3M-TODO implement and double check examples and test:
        the idea is to compute the tensor product of the matrices associated
        with the gates of the circuit in a clever way (to minimize the number of
        multiplications) and then return the big matrix
        """
        ...

    def inverse(self) -> QCircuit:
        """Generate the inverse (dagger) of this circuit.

        Examples:
            >>> c1 = QCircuit([H(0), CNOT(0,1)])
            >>> print(c1)
                 ┌───┐
            q_0: ┤ H ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘
            >>> print(c1.inverse())
                      ┌───┐
            q_0: ──■──┤ H ├
                 ┌─┴─┐└───┘
            q_1: ┤ X ├─────
                 └───┘
            >>> c2 = QCircuit([S(0), CZ(0,1), H(1), Ry(4.56, 1)])
            >>> print(c2)
                 ┌───┐
            q_0: ┤ S ├─■──────────────────
                 └───┘ │ ┌───┐┌──────────┐
            q_1: ──────■─┤ H ├┤ Ry(4.56) ├
                         └───┘└──────────┘
            >>> print(c2.inverse())

        Returns:
            The inverse circuit.

        # 3M-TODO implement, test, fill second example
        The inverse could be computed in several ways, depending on the
        definition of the circuit. One can inverse each gate in the circuit, or
        take the global unitary of the gate and inverse it.
        """
        dagger = QCircuit(self.nb_qubits)
        return dagger

    def to_gate(self) -> Gate:
        """Generate a gate from this entire circuit.

        Examples:
            >>> c = QCircuit([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), CNOT(2, 3)])
            >>> c.to_gate().definition.matrix
            >>>

        Returns:
            A gate representing this circuit.

        # 3M-TODO check implementation, example and test, this will only work
        chen circuit.to_matrix will be implemented
        """
        gate_def = UnitaryMatrix(self.to_matrix())
        return CustomGate(gate_def, list(range(self.nb_qubits)), label=self.label)

    @classmethod
    def initializer(cls, state: npt.NDArray[np.complex64]) -> QCircuit:
        """Initialize this circuit at a given state, given in parameter.

        This will imply adding gates at the beginning of the circuit.

        Examples:
            >>> qc = QCircuit.initializer(np.array([1, 0, 0 ,1])/np.sqrt(2))
            >>> print(qc)
                 ┌───┐
            q_0: ┤ H ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘

        Args:
            state: StateVector modeling the state for initializing the circuit.

        Returns:
            A copy of the input circuit with additional instructions added
            before-hand to generate the right initial state.

        # 3M-TODO : to implement --> a first sort term way could be to reuse the
        # qiskit QuantumCircuit feature qc.initialize()
        """
        size = int(np.log2(len(state)))
        if 2**size != len(state):
            raise ValueError(f"Input state {state} should have a power of 2 size")
        res = cls(size)
        ...
        return res

    def count_gates(self, gate: Optional[Type[Gate]] = None) -> int:
        """Returns the number of gates contained in the circuit. If a specific
        gate is given in the ``gate`` arg, it returns the number of occurrences
        of this gate.

        Examples:
            >>> circuit = QCircuit(
            ...     [X(0), Y(1), Z(2), CNOT(0, 1), SWAP(0, 1), CZ(1, 2), X(2), X(1), X(0)]
            ... )
            >>> circuit.count_gates()
            9
            >>> circuit.count_gates(X)
            4
            >>> circuit.count_gates(Ry)
            0

        Args:
            gate: The gate for which we want to know its occurrence in this
                circuit.

        Returns:
            The number of gates (eventually of a specific type) contained in the
            circuit.
        """
        filter2 = Gate if gate is None else gate
        return len([inst for inst in self.instructions if isinstance(inst, filter2)])

    def get_measurements(self) -> list[Measure]:
        """Returns all the measurements present in this circuit.

        Example:
            >>> circuit = QCircuit([
            ...     BasisMeasure([0, 1], shots=1000),
            ...     ExpectationMeasure([1], Observable(np.identity(2)), shots=1000)
            ... ])
            >>> circuit.get_measurements()
            [BasisMeasure([0, 1], shots=1000),
             ExpectationMeasure([1], Observable(array([[1., 0.], [0., 1.]])), shots=1000)]

        Returns:
            The list of all measurements present in the circuit.
        """
        return [inst for inst in self.instructions if isinstance(inst, Measure)]

    def without_measurements(self) -> QCircuit:
        """Provides a copy of this circuit with all the measurements removed.

        Example:
            >>> circuit = QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit)
                 ┌───┐     ┌─┐
            q_0: ┤ X ├──■──┤M├───
                 └───┘┌─┴─┐└╥┘┌─┐
            q_1: ─────┤ X ├─╫─┤M├
                      └───┘ ║ └╥┘
            c: 2/═══════════╩══╩═
                            0  1
            >>> print(circuit.without_measurements())
                 ┌───┐
            q_0: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘

        Returns:
            A copy of this circuit with all the measurements removed.
        """
        new_circuit = QCircuit(self.nb_qubits)
        new_circuit.instructions = [
            inst for inst in self.instructions if not isinstance(inst, Measure)
        ]

        return new_circuit

    def to_other_language(
        self, language: Language = Language.QISKIT
    ) -> Union[QuantumCircuit, myQLM_Circuit, braket_Circuit]:
        """Transforms this circuit into the corresponding circuit in the language
        specified in the ``language`` arg.

        By default, the circuit is translated to the corresponding
        ``QuantumCircuit`` in Qiskit, since it is the interface we use to
        generate the OpenQASM code.

        In the future, we will generate the OpenQASM code on our own, and this
        method will be used only for complex objects that are not tractable by
        OpenQASM (like hybrid structures).

        Example:
            >>> circuit = QCircuit([X(0), CNOT(0, 1)])
            >>> qc = circuit.to_other_language()
            >>> type(qc)
            qiskit.circuit.quantumcircuit.QuantumCircuit

        Args:
            language: Enum representing the target language.

        Returns:
            The corresponding circuit in the target language.
        """

        if language == Language.QISKIT:
            # to avoid defining twice the same parameter, we keep trace of the
            # added parameters, and we use those instead of new ones when they
            # are used more than once
            qiskit_parameters = set()
            if self.nb_cbits is None:
                new_circ = QuantumCircuit(self.nb_qubits)
            else:
                new_circ = QuantumCircuit(self.nb_qubits, self.nb_cbits)

            for instruction in self.instructions:
                if isinstance(instruction, ExpectationMeasure):
                    # these measures have no equivalent in Qiskit
                    continue
                qiskit_inst = instruction.to_other_language(
                    Language.QISKIT, qiskit_parameters
                )
                assert isinstance(qiskit_inst, CircuitInstruction) or isinstance(
                    qiskit_inst, Operation
                )
                cargs = []

                if isinstance(instruction, ControlledGate):
                    qargs = instruction.controls + instruction.targets
                elif isinstance(instruction, Gate):
                    qargs = instruction.targets
                elif isinstance(instruction, BasisMeasure) and isinstance(
                    instruction.basis, ComputationalBasis
                ):
                    assert instruction.c_targets is not None
                    qargs = [instruction.targets]
                    cargs = [instruction.c_targets]
                elif isinstance(instruction, Barrier):
                    qargs = range(instruction.size)
                else:
                    raise ValueError(f"Instruction not handled: {instruction}")

                new_circ.append(
                    qiskit_inst,
                    qargs,
                    cargs,
                )
            return new_circ

        elif language == Language.MY_QLM:
            cleaned_circuit = self.without_measurements()
            myqlm_circuit = qasm2_to_myqlm_Circuit(cleaned_circuit.to_qasm2())
            return myqlm_circuit

        elif language == Language.BRAKET:
            circuit_qasm3 = self.to_qasm3()
            brkt_circuit = qasm3_to_braket_Circuit(circuit_qasm3)
            return brkt_circuit

        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def to_qasm2(self) -> str:
        """Converts this circuit to the corresponding OpenQASM 2 code.

        For now, we use an intermediate conversion to a Qiskit
        ``QuantumCircuit``.

        Example:
            >>> circuit = QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit.to_qasm2())
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            x q[0];
            cx q[0],q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];

        Returns:
            A string representing the OpenQASM2 code corresponding to this
            circuit.
        """
        qiskit_circ = self.subs({}, remove_symbolic=True).to_other_language(
            Language.QISKIT
        )
        assert isinstance(qiskit_circ, QuantumCircuit)
        qasm = qiskit_circ.qasm()
        assert qasm is not None
        return qasm

    def to_qasm3(self) -> str:
        """Converts this circuit to the corresponding OpenQASM 3 code.

        For now, we use an intermediate conversion to OpenQASM 2, and then a
        converter from 2 to 3.

        Example:
            >>> circuit = QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit.to_qasm3())
            OPENQASM 3.0;
            include 'stdgates.inc';
            qubit[2] q;
            bit[2] c;
            x q[0];
            cx q[0],q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];

        Returns:
            A string representing the OpenQASM3 code corresponding to this
            circuit.
        """
        qasm2_code = self.to_qasm2()
        qasm3_code = open_qasm_2_to_3(qasm2_code)
        return qasm3_code

    def subs(
        self, values: dict[Expr | str, Complex], remove_symbolic: bool = False
    ) -> QCircuit:
        r"""Substitute the parameters of the circuit with complex values.
        Optionally also remove all symbolic variables such as `\pi` (needed for
        example for circuit execution).

        Since we use ``sympy`` for gates' parameters, ``values`` can in fact be
        anything the ``subs`` method from ``sympy`` would accept.

        Example:
            >>> theta, k = symbols("θ k")
            >>> c = QCircuit(
            ...     [Rx(theta, 0), CNOT(1,0), CNOT(1,2), X(2), Rk(2,1), H(0), CRk(k, 0, 1),
            ...      BasisMeasure(list(range(3)), shots=1000)]
            ... )
            >>> print(c)
                 ┌───────┐┌───┐┌───┐                              ┌─┐
            q_0: ┤ Rx(θ) ├┤ X ├┤ H ├────────────■─────────────────┤M├───
                 └───────┘└─┬─┘└───┘┌─────────┐ │P(2**(1 - k)*pi) └╥┘┌─┐
            q_1: ───────────■────■──┤ P(pi/2) ├─■──────────────────╫─┤M├
                               ┌─┴─┐└──┬───┬──┘        ┌─┐         ║ └╥┘
            q_2: ──────────────┤ X ├───┤ X ├───────────┤M├─────────╫──╫─
                               └───┘   └───┘           └╥┘         ║  ║
            c: 3/═══════════════════════════════════════╩══════════╩══╩═
                                                        2          0  1
            >>> print(c.subs({theta: np.py, k: 1}))
                 ┌───────┐┌───┐┌───┐                 ┌─┐
            q_0: ┤ Rx(π) ├┤ X ├┤ H ├───────────■─────┤M├───
                 └───────┘└─┬─┘└───┘┌────────┐ │P(π) └╥┘┌─┐
            q_1: ───────────■────■──┤ P(π/2) ├─■──────╫─┤M├
                               ┌─┴─┐└─┬───┬──┘  ┌─┐   ║ └╥┘
            q_2: ──────────────┤ X ├──┤ X ├─────┤M├───╫──╫─
                               └───┘  └───┘     └╥┘   ║  ║
            c: 3/════════════════════════════════╩════╩══╩═
                                                 2    0  1

        Args:
            values: Mapping between the variables and the replacing values.
            remove_symbolic: If symbolic values should be replaced by their
                numeric counterpart.

        Returns:
            The circuit with the replaced parameters.
        """
        return QCircuit(
            data=[inst.subs(values, remove_symbolic) for inst in self.instructions],
            nb_qubits=self.nb_qubits,
            nb_cbits=self.nb_cbits,
            label=self.label,
        )

    def pretty_print(self):
        """Provides a pretty print of the QCircuit.

        Example:
            >>> c = QCircuit([H(0), CNOT(0,1)])
            >>> c.pretty_print()
            QCircuit : Size (Qubits,Cbits) = (2, None), Nb instructions = 2
                 ┌───┐
            q_0: ┤ H ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘
        """
        print(
            f"QCircuit {self.label or ''}: Size (Qubits,Cbits) = {self.size()},"
            f" Nb instructions = {len(self)}\n"
            f"{self.to_other_language(Language.QISKIT)}"
        )

    def __str__(self) -> str:
        return str(self.to_other_language(Language.QISKIT))

    def __repr__(self) -> str:
        return f"QCircuit({self.instructions})"

    def variables(self):
        """Returns all the parameters involved in this circuit.

        Example:
            >>> circ = QCircuit([
            ...     Rx(theta, 0), CNOT(1,0), CNOT(1,2), X(2), Rk(2,1),
            ...     H(0), CRk(k, 0, 1), ExpectationMeasure([1], obs)
            ... ])
            >>> circ.variables()
            {k, θ}

        Returns:
            All the parameters of the circuit.
        """
        params: set[Basic] = set()
        for inst in self.instructions:
            if isinstance(inst, ParametrizedGate):
                for param in inst.parameters:
                    if isinstance(param, Expr):
                        params.update(param.free_symbols)
        return params
