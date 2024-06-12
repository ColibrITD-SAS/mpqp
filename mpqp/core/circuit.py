"""MPQP is focused on gate based quantum computing. As such, the main element of
a script using MPQP is the quantum circuit, or :class:`QCircuit`. The
:class:`QCircuit` contains the data of all gates, measures, and noise models you
want to apply to your qubits. 

The qubits are only referred by their indices, so one could keep track of
specific registers using python features, for instance

.. code-block:: python

    >>> circ = QCircuit(6)
    >>> targets = range(3)
    >>> ancillas = range(3,6)
    >>> for i in range(3):
    ...     circ.add(CNOT(targets[i], ancillas[i]))
    
could be use to add CNOT gates to your circuit, using the two registers
``targets`` and ``ancillas``.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Complex
from pickle import dumps
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Type, Union

if TYPE_CHECKING:
    from qat.core.wrappers.circuit import Circuit as myQLM_Circuit
    from cirq.circuits.circuit import Circuit as cirq_Circuit
    from braket.circuits import Circuit as braket_Circuit
    from qiskit.circuit import QuantumCircuit
    from sympy import Basic, Expr

import numpy as np
import numpy.typing as npt
from typeguard import TypeCheckError, typechecked

from mpqp.core.instruction import Instruction
from mpqp.core.instruction.barrier import Barrier
from mpqp.core.instruction.gates import ControlledGate, CRk, Gate, Id
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
from mpqp.core.instruction.measurement import BasisMeasure, ComputationalBasis, Measure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.core.languages import Language
from mpqp.noise.noise_model import Depolarizing, NoiseModel
from mpqp.qasm import qasm2_to_myqlm_Circuit
from mpqp.qasm.open_qasm_2_and_3 import open_qasm_2_to_3
from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit
from mpqp.qasm.qasm_to_cirq import qasm2_to_cirq_Circuit
from mpqp.tools.errors import NumberQubitsError
from mpqp.tools.generics import OneOrMany
from mpqp.tools.maths import matrix_eq


@typechecked
class QCircuit:
    """This class models a quantum circuit.

    A circuit is composed of instructions and noise models applied on
    quantum and/or classical bits. These elements (instructions and noise
    models) will be called ``components`` hereafter.

    Args:
        data: Number of qubits or list of ``components`` to initialize the circuit
            with. If the number of qubits is passed, it should be a positive int.
        nb_qubits: Optional number of qubits, in case you input the sequence of
            instruction and want to hardcode the number of qubits.
        nb_cbits: Number of classical bits. It should be positive.
        label: Name of the circuit.

    Examples:
        >>> circuit = QCircuit(2)
        >>> circuit.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
        QCircuit : Size (Qubits,Cbits) = (2, 0), Nb instructions = 0
        q_0:
        q_1:

        >>> circuit = QCircuit(5, nb_cbits=2, label="Circuit 1")
        >>> circuit.add(Rx(1.23, 3))
        >>> circuit.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
        QCircuit Circuit 1: Size (Qubits,Cbits) = (5, 2), Nb instructions = 1
        q_0: ────────────
        q_1: ────────────
        q_2: ────────────
             ┌──────────┐
        q_3: ┤ Rx(1.23) ├
             └──────────┘
        q_4: ────────────
        c: 2/════════════

        >>> circuit = QCircuit(3, label="NoiseExample")
        >>> circuit.add([H(0), T(1), CNOT(0,1), S(2)])
        >>> circuit.add(BasisMeasure(list(range(3)), shots=2345))
        >>> circuit.add(Depolarizing(prob=0.50, targets=[0, 1]))
        >>> circuit.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
        QCircuit NoiseExample: Size (Qubits,Cbits) = (3, 3), Nb instructions = 5
        Depolarizing noise: probability 0.5 on qubits [0, 1]
             ┌───┐     ┌─┐
        q_0: ┤ H ├──■──┤M├───
             ├───┤┌─┴─┐└╥┘┌─┐
        q_1: ┤ T ├┤ X ├─╫─┤M├
             ├───┤└┬─┬┘ ║ └╥┘
        q_2: ┤ S ├─┤M├──╫──╫─
             └───┘ └╥┘  ║  ║
        c: 3/═══════╩═══╩══╩═
                    2   0  1

    """

    def __init__(
        self,
        data: int | Sequence[Union[Instruction, NoiseModel]],
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
        self.noises: list[NoiseModel] = []
        """List of noise models attached to the circuit."""
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
                        *(item.connections() for item in data)
                    )
                    self.nb_qubits = max(connections) + 1
            else:
                self.nb_qubits = nb_qubits
            self.add(list(map(deepcopy, data)))

    def __eq__(self, value: object) -> bool:
        return dumps(self) == dumps(value)

    def add(self, components: OneOrMany[Instruction | NoiseModel]):
        """Adds a ``component`` or a list of ``component`` at the end of the circuit.

        Args:
            components : Instruction(s) or NoiseModel(s) to append to the circuit.

        Examples:
            >>> circuit = QCircuit(2)
            >>> circuit.add(X(0))
            >>> circuit.add([CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> circuit.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
            QCircuit : Size (Qubits,Cbits) = (2, 2), Nb instructions = 3
                 ┌───┐     ┌─┐
            q_0: ┤ X ├──■──┤M├───
                 └───┘┌─┴─┐└╥┘┌─┐
            q_1: ─────┤ X ├─╫─┤M├
                      └───┘ ║ └╥┘
            c: 2/═══════════╩══╩═
                            0  1

            >>> circuit.add(Depolarizing(0.3, [0,1], dimension=2, gates=[CNOT]))
            >>> circuit.add([Depolarizing(0.02, [0])])
            >>> circuit.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
            QCircuit : Size (Qubits,Cbits) = (2, 2), Nb instructions = 3
            Depolarizing noise: probability 0.3 for gates [CNOT]
            Depolarizing noise: probability 0.02 on qubits [0]
                 ┌───┐     ┌─┐
            q_0: ┤ X ├──■──┤M├───
                 └───┘┌─┴─┐└╥┘┌─┐
            q_1: ─────┤ X ├─╫─┤M├
                      └───┘ ║ └╥┘
            c: 2/═══════════╩══╩═
                            0  1

        """

        if isinstance(components, Iterable):
            for comp in components:
                self.add(comp)
            return

        if any(conn >= self.nb_qubits for conn in components.connections()):
            component_type = (
                "Instruction" if isinstance(components, Instruction) else "Noise model"
            )
            raise NumberQubitsError(
                f"{component_type} {type(components)}'s connections "
                f"({components.connections()}) are not compatible with circuit"
                f" size ({self.nb_qubits})."
            )

        if isinstance(components, BasisMeasure):
            if self.noises and len(components.targets) != self.nb_qubits:
                raise ValueError(
                    "In noisy circuits, BasisMeasure must span all qubits in the circuit."
                )
            # has to be done in two steps, because Pycharm's type checker is
            # unable to understand chained type inference
            if components.c_targets is None:
                if self.nb_cbits is None:
                    self.nb_cbits = 0
                components.c_targets = [
                    self.nb_cbits + i for i in range(len(components.targets))
                ]
                self.nb_cbits += len(components.c_targets)

        if isinstance(components, Barrier):
            components.size = self.nb_qubits

        if isinstance(components, NoiseModel):
            if len(components.targets) == 0:
                components.targets = [target for target in range(self.nb_qubits)]
                
            basisMs = [
                instr for instr in self.instructions if isinstance(instr, BasisMeasure)
            ]
            if basisMs and all([len(bm.targets) != self.nb_qubits for bm in basisMs]):
                raise ValueError(
                    "In noisy circuits, BasisMeasure must span all qubits in the circuit."
                )

            self.noises.append(components)
        else:
            self.instructions.append(components)

    def append(self, other: QCircuit, qubits_offset: int = 0) -> None:
        """Appends the circuit at the end (right side) of this circuit, inplace.

        If the size of the ``other`` is smaller than this circuit,
        the parameter ``qubits_offset`` can be used to indicate at which qubit
        the ``other`` circuit must be added.

        Args:
            other: The circuit to append at the end of this circuit.
            qubits_offset: If the circuit in parameter is smaller, this
                parameter precise at which qubit (vertically) the circuit will
                be added.

        Raises:
            NumberQubitsError: if the circuit in parameter is larger than this
                circuit or if the ``qubits_offset`` is too big such that the
                ``other`` circuit would "stick out".

        Example:
            >>> c1 = QCircuit([CNOT(0,1),CNOT(1,2)])
            >>> c2 = QCircuit([X(1),CNOT(1,2)])
            >>> c1.append(c2)
            >>> print(c1)  # doctest: +NORMALIZE_WHITESPACE
            q_0: ──■─────────────────
                 ┌─┴─┐     ┌───┐
            q_1: ┤ X ├──■──┤ X ├──■──
                 └───┘┌─┴─┐└───┘┌─┴─┐
            q_2: ─────┤ X ├─────┤ X ├
                      └───┘     └───┘

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

        In the circuit notation, the upper part of the output circuit will
        correspond to the first circuit, while the bottom part correspond to the
        one in parameter.

        Args:
            other: QCircuit being the second operand of the tensor product with
                this circuit.

        Returns:
            The QCircuit resulting from the tensor product of this circuit with
            the one in parameter.

        Args:
            other: QCircuit being the second operand of the tensor product with this circuit.

        Returns:
            The QCircuit resulting from the tensor product of this circuit with the one in parameter.

        Example:
            >>> c1 = QCircuit([CNOT(0,1),CNOT(1,2)])
            >>> c2 = QCircuit([X(1),CNOT(1,2)])
            >>> print(c1.tensor(c2))  # doctest: +NORMALIZE_WHITESPACE
            q_0: ──■───────
                 ┌─┴─┐
            q_1: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_2: ─────┤ X ├
                      └───┘
            q_3: ──────────
                 ┌───┐
            q_4: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_5: ─────┤ X ├
                      └───┘

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
            
        Args:
            output: Format of the output, see
                `docs.quantum.ibm.com/build/circuit-visualization <https://docs.quantum.ibm.com/build/circuit-visualization#alternative-renderers>`_
                for more information.
        
        Examples:
            >>> theta = symbols("θ")
            >>> circ = QCircuit([
            ...     P(theta, 0),
            ...     ExpectationMeasure([0], Observable(np.array([[0, 1], [1, 0]])), shots=1000)
            ... ])
            >>> circ.display("text")
               ┌──────┐
            q: ┤ P(θ) ├
               └──────┘
            >>> print(circ.display("latex_source"))  # doctest: +NORMALIZE_WHITESPACE
            \documentclass[border=2px]{standalone}
            \usepackage[braket, qm]{qcircuit}
            \usepackage{graphicx}
            \begin{document}
            \scalebox{1.0}{
            \Qcircuit @C=1.0em @R=0.2em @!R { \\
                \nghost{{q} :  } & \lstick{{q} :  } & \gate{\mathrm{P}\,(\mathrm{{\ensuremath{\theta}}})} & \qw & \qw\\
            \\ }}
            \end{document}

        """
        from matplotlib.figure import Figure
        from qiskit.tools.visualization import circuit_drawer

        qc = self.to_other_language(language=Language.QISKIT)
        fig = circuit_drawer(qc, output=output, style={"backgroundcolor": "#EEEEEE"})

        if isinstance(fig, Figure):
            fig.show()
        return fig

    def size(self) -> tuple[int, int]:
        """Provides the size of the circuit, in terms of number of quantum and
        classical bits.

        Returns:
            A couple ``(q, c)`` of integers, with ``q`` the number of qubits,
            and ``c`` the number of cbits of this circuit.

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

        """
        return self.nb_qubits, (self.nb_cbits or 0)

    def depth(self) -> int:
        """Computes the depth of the circuit.

        Returns:
            Depth of the circuit.

        Examples:
            >>> QCircuit([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), X(2)]).depth()
            3
            >>> QCircuit([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), Barrier(), X(2)]).depth()
            4

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

        Returns:
            An integer representing the number of instructions in this circuit.

        Example:
            >>> c1 = QCircuit([CNOT(0,1), CNOT(1,2), X(1), CNOT(1,2)])
            >>> len(c1)
            4

        """
        return len(self.instructions)

    def is_equivalent(self, circuit: QCircuit) -> bool:
        """Whether the circuit in parameter is equivalent to this circuit, in
        terms of gates, but not measurements.

        Depending on the definition of the gates of the circuit, several methods
        could be used to do it in an optimized way.

        Args:
            circuit: The circuit for which we want to know if it is equivalent
                to this circuit.

        Returns:
            ``True`` if the circuit in parameter is equivalent to this circuit

        Example:
            >>> c1 = QCircuit([H(0), H(0)])
            >>> c2 = QCircuit([Rx(0, 0)])
            >>> c1.is_equivalent(c2)
            True

        3M-TODO: will only work once the circuit.to_matrix is implemented
         Also take into account Noise in the equivalence verification
        """
        return matrix_eq(self.to_matrix(), circuit.to_matrix())

    def optimize(self, criteria: Optional[OneOrMany[str]] = None) -> QCircuit:
        """Optimize the circuit to satisfy some criteria (depth, number of
        qubits, gate restriction) in parameter.

        Args:
            criteria: String, or list of strings, regrouping the criteria of optimization of the circuit.

        Returns:
            the optimized QCircuit

        Examples:
            >>>
            >>>
            >>>

        # 6M-TODO implement, example and test
        """
        # ideas: a circuit can be optimized
        # - to reduce the depth of the circuit (combine gates, simplify some sequences)
        # - according to a given topology or qubits connectivity map
        # - to avoid the use of some gates (imperfect or more noisy)
        # - to avoid multi-qubit gates
        ...

    def to_matrix(self) -> npt.NDArray[np.complex64]:
        """Compute the unitary matrix associated to this circuit.

        Returns:
            a unitary matrix representing this circuit

        Examples:
            >>> c = QCircuit([H(0), CNOT(0,1)])
            >>> c.to_matrix()
            array([[ 0.70710678,  0.        ,  0.70710678,  0.        ],
                   [ 0.        ,  0.70710678,  0.        ,  0.70710678],
                   [ 0.        ,  0.70710678,  0.        , -0.70710678],
                   [ 0.70710678,  0.        , -0.70710678,  0.        ]])

        # 3M-TODO implement and double check examples and test:
        the idea is to compute the tensor product of the matrices associated
        with the gates of the circuit in a clever way (to minimize the number of
        multiplications) and then return the big matrix
        """
        ...

    def inverse(self) -> QCircuit:
        """Generate the inverse (dagger) of this circuit.

        Returns:
            The inverse circuit.

        Examples:
            >>> c1 = QCircuit([H(0), CNOT(0,1)])
            >>> print(c1)  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐
            q_0: ┤ H ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘
            >>> print(c1.inverse())  # doctest: +NORMALIZE_WHITESPACE
                      ┌───┐
            q_0: ──■──┤ H ├
                 ┌─┴─┐└───┘
            q_1: ┤ X ├─────
                 └───┘
            >>> c2 = QCircuit([S(0), CZ(0,1), H(1), Ry(4.56, 1)])
            >>> print(c2)  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐
            q_0: ┤ S ├─■──────────────────
                 └───┘ │ ┌───┐┌──────────┐
            q_1: ──────■─┤ H ├┤ Ry(4.56) ├
                         └───┘└──────────┘
            >>> print(c2.inverse())  # doctest: +NORMALIZE_WHITESPACE
                                     ┌───┐
            q_0: ──────────────────■─┤ S ├
                 ┌──────────┐┌───┐ │ └───┘
            q_1: ┤ Ry(4.56) ├┤ H ├─■──────
                 └──────────┘└───┘

        # TODO implement, test, fill second example
        The inverse could be computed in several ways, depending on the
        definition of the circuit. One can inverse each gate in the circuit, or
        take the global unitary of the gate and inverse it.
        """
        dagger = QCircuit(self.nb_qubits)
        for instr in reversed(self.instructions):
            dagger.add(instr)
        return dagger

    def to_gate(self) -> Gate:
        """Generate a gate from this entire circuit.

        Returns:
            A gate representing this circuit.

        Examples:
            >>> c = QCircuit([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), CNOT(2, 3)])
            >>> c.to_gate().definition.matrix

        # 3M-TODO check implementation, example and test, this will only work
           when circuit.to_matrix() will be implemented
        """
        gate_def = UnitaryMatrix(self.to_matrix())
        return CustomGate(gate_def, list(range(self.nb_qubits)), label=self.label)

    @classmethod
    def initializer(cls, state: npt.NDArray[np.complex64]) -> QCircuit:
        """Initialize this circuit at a given state, given in parameter.
        This will imply adding gates at the beginning of the circuit.

        Args:
            state: StateVector modeling the state for initializing the circuit.

        Returns:
            A copy of the input circuit with additional instructions added
            before-hand to generate the right initial state.

        Examples:
            >>> qc = QCircuit.initializer(np.array([1, 0, 0 ,1])/np.sqrt(2))
            >>> print(qc)  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐
            q_0: ┤ H ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘

        # 3M-TODO : to implement --> a first short term way could be to reuse
        # the qiskit QuantumCircuit feature qc.initialize()
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

        Args:
            gate: The gate for which we want to know its occurrence in this
                circuit.

        Returns:
            The number of gates (eventually of a specific type) contained in the
            circuit.

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

        """
        filter2 = Gate if gate is None else gate
        return len([inst for inst in self.instructions if isinstance(inst, filter2)])

    def get_measurements(self) -> list[Measure]:
        """Returns all the measurements present in this circuit.

        Returns:
            The list of all measurements present in the circuit.

        Example:
            >>> circuit = QCircuit([
            ...     BasisMeasure([0, 1], shots=1000),
            ...     ExpectationMeasure([1], Observable(np.identity(2)), shots=1000)
            ... ])
            >>> circuit.get_measurements()  # doctest: +NORMALIZE_WHITESPACE
            [BasisMeasure([0, 1], shots=1000),
            ExpectationMeasure([1], Observable(array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]], dtype=complex64)), shots=1000)]

        """
        return [inst for inst in self.instructions if isinstance(inst, Measure)]

    def without_measurements(self) -> QCircuit:
        """Provides a copy of this circuit with all the measurements removed.

        Returns:
            A copy of this circuit with all the measurements removed.

        Example:
            >>> circuit = QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit)  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐     ┌─┐
            q_0: ┤ X ├──■──┤M├───
                 └───┘┌─┴─┐└╥┘┌─┐
            q_1: ─────┤ X ├─╫─┤M├
                      └───┘ ║ └╥┘
            c: 2/═══════════╩══╩═
                            0  1
            >>> print(circuit.without_measurements())  # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐
            q_0: ┤ X ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘

        """
        new_circuit = QCircuit(self.nb_qubits)
        new_circuit.instructions = [
            inst for inst in self.instructions if not isinstance(inst, Measure)
        ]

        return new_circuit

    def without_noises(self) -> QCircuit:
        """Provides a copy of this circuit with all the noise models removed.

        Returns:
            A copy of this circuit with all the noise models removed.

        Example:
            >>> circuit = QCircuit(2)
            >>> circuit.add([CNOT(0, 1), Depolarizing(prob=0.4, targets=[0, 1]), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit)  # doctest: +NORMALIZE_WHITESPACE
                      ┌─┐
            q_0: ──■──┤M├───
                 ┌─┴─┐└╥┘┌─┐
            q_1: ┤ X ├─╫─┤M├
                 └───┘ ║ └╥┘
            c: 2/══════╩══╩═
                       0  1
            NoiseModel: Depolarizing(0.4, [0, 1], 1)
            >>> print(circuit.without_noises())  # doctest: +NORMALIZE_WHITESPACE
                      ┌─┐
            q_0: ──■──┤M├───
                 ┌─┴─┐└╥┘┌─┐
            q_1: ┤ X ├─╫─┤M├
                 └───┘ ║ └╥┘
            c: 2/══════╩══╩═
                       0  1

        """
        new_circuit = deepcopy(self)
        new_circuit.noises = []
        return new_circuit

    def to_other_language(
        self, language: Language = Language.QISKIT, cirq_proc_id: Optional[str] = None
    ) -> Union[
        QuantumCircuit,
        myQLM_Circuit,
        braket_Circuit,
        cirq_Circuit,
    ]:
        """Transforms this circuit into the corresponding circuit in the language
        specified in the ``language`` arg.

        By default, the circuit is translated to the corresponding
        ``QuantumCircuit`` in Qiskit, since it is the interface we use to
        generate the OpenQASM code.

        In the future, we will generate the OpenQASM code on our own, and this
        method will be used only for complex objects that are not tractable by
        OpenQASM (like hybrid structures).

        Note:
            Most providers take noise into account at the job level. A notable
            exception is Braket, where the noise is contained in the circuit
            object. For this reason you find the noise included in the Braket
            circuits.

        Args:
            language: Enum representing the target language.
            cirq_proc_id : Identifier of the processor for cirq.

        Returns:
            The corresponding circuit in the target language.

        Examples:
            >>> circuit = QCircuit([X(0), CNOT(0, 1)])
            >>> qc = circuit.to_other_language()
            >>> type(qc)
            <class 'qiskit.circuit.quantumcircuit.QuantumCircuit'>
            >>> circuit2 = QCircuit([H(0), CZ(0,1), Depolarizing(0.6, [0])])
            >>> braket_circuit = circuit2.to_other_language(Language.BRAKET)
            >>> print(braket_circuit)  # doctest: +NORMALIZE_WHITESPACE
            T  : │         0         │         1         │
                  ┌───┐ ┌───────────┐       ┌───────────┐
            q0 : ─┤ H ├─┤ DEPO(0.6) ├───●───┤ DEPO(0.6) ├─
                  └───┘ └───────────┘   │   └───────────┘
                                      ┌─┴─┐
            q1 : ─────────────────────┤ Z ├───────────────
                                      └───┘
            T  : │         0         │         1         │

        """

        if language == Language.QISKIT:
            from qiskit.circuit import Operation, QuantumCircuit
            from qiskit.circuit.quantumcircuit import CircuitInstruction
            from qiskit.quantum_info import Operator

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
                if TYPE_CHECKING:
                    assert (
                        isinstance(qiskit_inst, CircuitInstruction)
                        or isinstance(qiskit_inst, Operation)
                        or isinstance(qiskit_inst, Operator)
                    )
                cargs = []

                if isinstance(instruction, CustomGate):
                    new_circ.unitary(  # pyright: ignore[reportAttributeAccessIssue]
                        instruction.to_other_language(),
                        instruction.targets,
                        instruction.label,
                    )
                    # FIXME: minus sign appearing when it should not, seems
                    # there a phase added somewhere, check u gate in OpenQASM
                    # translation.
                    continue
                elif isinstance(instruction, ControlledGate):
                    qargs = instruction.controls + instruction.targets
                elif isinstance(instruction, Gate):
                    qargs = instruction.targets
                elif isinstance(instruction, BasisMeasure) and isinstance(
                    instruction.basis, ComputationalBasis
                ):
                    # TODO muhammad/henri, for custom basis, check if something
                    # should be changed here, otherwise remove the condition to
                    # have only computational basis
                    assert instruction.c_targets is not None
                    qargs = [instruction.targets]
                    cargs = [instruction.c_targets]
                elif isinstance(instruction, Barrier):
                    qargs = range(instruction.size)
                else:
                    raise ValueError(f"Instruction not handled: {instruction}")

                if TYPE_CHECKING:
                    assert not isinstance(qiskit_inst, Operator)
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
            # filling the circuit with identity gates when some qubits don't have any instruction
            used_qubits = set().union(
                *(
                    inst.connections()
                    for inst in self.instructions
                    if isinstance(inst, Gate)
                )
            )
            circuit = QCircuit(
                [
                    Id(qubit)
                    for qubit in range(self.nb_qubits)
                    if qubit not in used_qubits
                ],
                nb_qubits=self.nb_qubits,
            ) + deepcopy(self)

            from mpqp.execution.providers.aws import apply_noise_to_braket_circuit

            if len(self.noises) != 0:
                if any(isinstance(instr, CRk) for instr in self.instructions):
                    raise NotImplementedError(
                        "Cannot simulate noisy circuit with CRk gate due to "
                        "an error on AWS Braket side."
                    )

            return apply_noise_to_braket_circuit(
                qasm3_to_braket_Circuit(circuit.to_qasm3()), self.noises, self.nb_qubits
            )
        elif language == Language.CIRQ:
            cirq_circuit = qasm2_to_cirq_Circuit(self.to_qasm2())
            if cirq_proc_id:
                from cirq.transformers.optimize_for_target_gateset import (
                    optimize_for_target_gateset,
                )
                from cirq.transformers.routing.route_circuit_cqc import RouteCQC
                from cirq.transformers.target_gatesets.sqrt_iswap_gateset import (
                    SqrtIswapTargetGateset,
                )
                from cirq_google.engine.virtual_engine_factory import (
                    create_device_from_processor_id,
                )

                device = create_device_from_processor_id(cirq_proc_id)
                if device.metadata is None:
                    raise ValueError(
                        f"Device {device} does not have metadata for processor {cirq_proc_id}"
                    )

                router = RouteCQC(device.metadata.nx_graph)
                rcirc, initial_map, swap_map = router.route_circuit(cirq_circuit)  # type: ignore[reportUnusedVariable]
                cirq_circuit = optimize_for_target_gateset(
                    rcirc, gateset=SqrtIswapTargetGateset()
                )

                device.validate_circuit(cirq_circuit)
            return cirq_circuit

        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def to_qasm2(self) -> str:
        """Converts this circuit to the corresponding OpenQASM 2 code.

        For now, we use an intermediate conversion to a Qiskit
        ``QuantumCircuit``.

        Returns:
            A string representing the OpenQASM2 code corresponding to this
            circuit.

        Example:
            >>> circuit = QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit.to_qasm2())  # doctest: +NORMALIZE_WHITESPACE
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            x q[0];
            cx q[0],q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];

        """
        qiskit_circ = self.subs({}, remove_symbolic=True).to_other_language(
            Language.QISKIT
        )
        if TYPE_CHECKING:
            assert isinstance(qiskit_circ, QuantumCircuit)
        qasm = qiskit_circ.qasm()
        assert qasm is not None
        return qasm

    def to_qasm3(self) -> str:
        """Converts this circuit to the corresponding OpenQASM 3 code.

        For now, we use an intermediate conversion to OpenQASM 2, and then a
        converter from 2 to 3.

        Returns:
            A string representing the OpenQASM3 code corresponding to this
            circuit.

        Example:
            >>> circuit = QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
            >>> print(circuit.to_qasm3())  # doctest: +NORMALIZE_WHITESPACE
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            bit[2] c;
            x q[0];
            cx q[0],q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];

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

        Args:
            values: Mapping between the variables and the replacing values.
            remove_symbolic: If symbolic values should be replaced by their
                numeric counterpart.

        Returns:
            The circuit with the replaced parameters.

        Examples:
            >>> theta, k = symbols("θ k")
            >>> c = QCircuit(
            ...     [Rx(theta, 0), CNOT(1,0), CNOT(1,2), X(2), Rk(2,1), H(0), CRk(k, 0, 1),
            ...      BasisMeasure(list(range(3)), shots=1000)]
            ... )
            >>> print(c)  # doctest: +NORMALIZE_WHITESPACE
                 ┌───────┐┌───┐┌───┐                              ┌─┐
            q_0: ┤ Rx(θ) ├┤ X ├┤ H ├────────────■─────────────────┤M├───
                 └───────┘└─┬─┘└───┘┌─────────┐ │P(2**(1 - k)*pi) └╥┘┌─┐
            q_1: ───────────■────■──┤ P(pi/2) ├─■──────────────────╫─┤M├
                               ┌─┴─┐└──┬───┬──┘        ┌─┐         ║ └╥┘
            q_2: ──────────────┤ X ├───┤ X ├───────────┤M├─────────╫──╫─
                               └───┘   └───┘           └╥┘         ║  ║
            c: 3/═══════════════════════════════════════╩══════════╩══╩═
                                                        2          0  1
            >>> print(c.subs({theta: np.pi, k: 1}))  # doctest: +NORMALIZE_WHITESPACE
                 ┌───────┐┌───┐┌───┐                 ┌─┐
            q_0: ┤ Rx(π) ├┤ X ├┤ H ├───────────■─────┤M├───
                 └───────┘└─┬─┘└───┘┌────────┐ │P(π) └╥┘┌─┐
            q_1: ───────────■────■──┤ P(π/2) ├─■──────╫─┤M├
                               ┌─┴─┐└─┬───┬──┘  ┌─┐   ║ └╥┘
            q_2: ──────────────┤ X ├──┤ X ├─────┤M├───╫──╫─
                               └───┘  └───┘     └╥┘   ║  ║
            c: 3/════════════════════════════════╩════╩══╩═
                                                 2    0  1

        """
        return QCircuit(
            data=[inst.subs(values, remove_symbolic) for inst in self.instructions]
            + self.noises,  # 3M-TODO: modify this line when noise will be
            # parameterized, to substitute, like we do for inst
            nb_qubits=self.nb_qubits,
            nb_cbits=self.nb_cbits,
            label=self.label,
        )

    def pretty_print(self):
        """Provides a pretty print of the QCircuit.

        Examples:
            >>> c = QCircuit([H(0), CNOT(0,1)])
            >>> c.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
            QCircuit : Size (Qubits,Cbits) = (2, 0), Nb instructions = 2
                 ┌───┐
            q_0: ┤ H ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘

        """
        print(
            f"QCircuit {self.label or ''}: Size (Qubits,Cbits) = {self.size()},"
            f" Nb instructions = {len(self)}"
        )

        qubits = set(range(self.size()[0]))
        if self.noises:
            for noise in self.noises:
                if not isinstance(noise, Depolarizing):
                    raise NotImplementedError(
                        "For now, only depolarizing noise is supported."
                    )
                targets = set(noise.targets)
                noise_info = f"{type(noise).__name__} noise: probability {noise.proba}"
                if targets != qubits:
                    noise_info += f" on qubits {noise.targets}"
                if noise.gates:
                    noise_info += f" for gates {noise.gates}"
                print(noise_info)

        print(f"{self.to_other_language(Language.QISKIT)}")

    def __str__(self) -> str:
        qiskit_circ = self.to_other_language(Language.QISKIT)
        if TYPE_CHECKING:
            from qiskit import QuantumCircuit

            assert isinstance(qiskit_circ, QuantumCircuit)
        output = str(qiskit_circ.draw(output="text", fold=0))
        if TYPE_CHECKING:
            assert isinstance(output, str)
        if len(self.noises) != 0:
            output += "\nNoiseModel:\n    " + "\n    ".join(
                str(noise) for noise in self.noises
            )
        return output

    def __repr__(self) -> str:
        instructions_repr = ", ".join(repr(instr) for instr in self.instructions)
        instructions_repr = instructions_repr.replace("[", "").replace("]", "")

        if self.noises:
            noise_repr = ", ".join(map(repr, self.noises))
            return f'QCircuit([{instructions_repr}, {noise_repr}], nb_qubits={self.nb_qubits}, nb_cbits={self.nb_cbits}, label="{self.label}")'
        else:
            return f'QCircuit([{instructions_repr}], nb_qubits={self.nb_qubits}, nb_cbits={self.nb_cbits}, label="{self.label}")'

    def variables(self) -> set[Basic]:
        """Returns all the parameters involved in this circuit.

        Returns:
            All the parameters of the circuit.

        Example:
            >>> circ = QCircuit([
            ...     Rx(theta, 0), CNOT(1,0), CNOT(1,2), X(2), Rk(2,1),
            ...     H(0), CRk(k, 0, 1), ExpectationMeasure([1], obs)
            ... ])
            >>> circ.variables()  # doctest: +SKIP
            {θ, k}

        """
        from sympy import Expr

        params: set[Basic] = set()
        for inst in self.instructions:
            if isinstance(inst, ParametrizedGate):
                for param in inst.parameters:
                    if isinstance(param, Expr):
                        params.update(param.free_symbols)
        return params
