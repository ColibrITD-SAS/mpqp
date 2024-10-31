import random
from typing import TYPE_CHECKING, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import CircuitInstruction

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.gate import SingleQubitGate, Gate
from mpqp.core.instruction.gates.native_gates import (
    ControlledGate,
    NATIVE_GATES,
    TOF,
    CRk,
    NativeGate,
    P,
    Rk,
    RotationGate,
    Rx,
    Ry,
    Rz,
    U,
)
from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
from mpqp.tools.maths import closest_unitary


def random_circuit(
    gate_classes: Optional[list[type]] = None,
    nb_qubits: int = 5,
    nb_gates: int = np.random.randint(5, 10),
) -> QCircuit:
    """This function creates a QCircuit with a specified number of qubits and gates.
    The gates are chosen randomly from the provided list of native gate classes.

    args:
        nb_qubits : Number of qubits in the circuit.
        gate_classes : List of native gate classes to use in the circuit.
        nb_gates : Number of gates to add to the circuit. Defaults to a random
         integer between 5 and 10.

    Returns:
        A quantum circuit with the specified number of qubits and randomly chosen gates.

    Raises:
        ValueError: If the number of qubits is too low for the specified gates.

    Examples:
        >>> random_circuit([U, TOF], 3) # doctest: +SKIP
                            ┌──────────────────────────┐┌───┐
        q_0: ──■────■────■──┤ U(2.4433,0.72405,4.7053) ├┤ X ├───────────────────────────
             ┌─┴─┐  │    │  └┬────────────────────────┬┘└─┬─┘
        q_1: ┤ X ├──■────■───┤ U(4.603,5.6604,1.4087) ├───■─────────────────────────────
             └─┬─┘┌─┴─┐┌─┴─┐┌┴────────────────────────┤   │  ┌─────────────────────────┐
        q_2: ──■──┤ X ├┤ X ├┤ U(4.2873,4.4459,2.6049) ├───■──┤ U(3.6991,3.7342,2.4204) ├
                  └───┘└───┘└─────────────────────────┘      └─────────────────────────┘
        >>> from mpqp.core.instruction.gates import native_gates
        >>> random_circuit(nb_qubits=4, nb_gates=10) # doctest: +SKIP
                           ┌───┐           ┌────────────┐           ┌───┐
        q_0: ──────■───────┤ Z ├─X─────────┤ Rz(5.0279) ├───────────┤ S ├──────────
                   │       └───┘ │         └────────────┘           └───┘
        q_1: ──────■─────────────┼─────────────────────────────────────────────────
             ┌────────────┐      │ ┌───────────────────────────┐┌────────────┐┌───┐
        q_2: ┤ Rz(2.7008) ├──────X─┤ U(1.8753,2.3799,0.012721) ├┤ Rx(3.5982) ├┤ H ├
             └───┬───┬────┘        └───────────────────────────┘└────────────┘└───┘
        q_3: ────┤ Y ├─────────────────────────────────────────────────────────────
                 └───┘
    """
    qcircuit = QCircuit(nb_qubits)
    for _ in range(nb_gates):
        qcircuit.add(random_instruction(gate_classes, nb_qubits))
    return qcircuit


def random_instruction(
    gate_classes: Optional[list[type]] = None,
    nb_qubits: int = 5,
) -> Gate:
    """This function creates a instruction with a specified number of qubits.
    The gates are chosen randomly from the provided list of native gate classes.

    args:
        nb_qubits : Number of qubits in the circuit.
        gate_classes : List of native gate classes to use in the circuit.

    Returns:
        A quantum circuit with the specified number of qubits and randomly chosen gates.

    Raises:
        ValueError: If the number of qubits is too low for the specified gates.

    Examples:
        >>> random_instruction([U, TOF], 3) # doctest: +SKIP
        U(2.067365317109373, 0.18652872274018245, 0.443968374745352, 0)
        >>> random_instruction(nb_qubits=4) # doctest: +SKIP
        SWAP(3, 1)

    """

    if gate_classes is None:
        gate_classes = []
        for gate in NATIVE_GATES:
            if TYPE_CHECKING:
                assert isinstance(gate.nb_qubits, int)
            if gate.nb_qubits <= nb_qubits:
                gate_classes.append(gate)
    elif any(not issubclass(gate, Gate) for gate in gate_classes):
        raise ValueError("gate_classes must be an instance of Gate")

    qubits = list(range(nb_qubits))

    if any(
        not issubclass(gate, SingleQubitGate)
        and ((gate == TOF and nb_qubits <= 2) or nb_qubits <= 1)
        for gate in gate_classes
    ):
        raise ValueError("number of qubits too low for this gates")

    gate_class: type[Gate] = random.choice(gate_classes)
    target = random.choice(qubits)

    if issubclass(gate_class, SingleQubitGate):
        if issubclass(gate_class, ParametrizedGate):
            if issubclass(gate_class, U):
                return U(
                    random.uniform(0, 2 * np.pi),
                    random.uniform(0, 2 * np.pi),
                    random.uniform(0, 2 * np.pi),
                    target,
                )
            elif issubclass(gate_class, Rk):
                return Rk(random.randint(0, 10), target)
            elif issubclass(gate_class, RotationGate):
                if TYPE_CHECKING:
                    assert issubclass(gate_class, (Rx, Ry, Rz, P))
                return gate_class(random.uniform(0, 2 * np.pi), target)
            else:
                raise ValueError
        else:
            return gate_class(target)
    else:
        control = random.choice(list(set(qubits) - {target}))
        if issubclass(gate_class, ParametrizedGate):
            if TYPE_CHECKING:
                assert issubclass(gate_class, CRk)
            return gate_class(
                random.randint(0, 10),
                control,
                target,
            )
        elif issubclass(gate_class, TOF):
            control2 = random.choice(list(set(qubits) - {target, control}))
            return TOF([control, control2], target)
        else:
            return gate_class(control, target)


def compute_expected_matrix(qcircuit: QCircuit):
    """
    Computes the expected matrix resulting from applying single-qubit gates
    in reverse order on a quantum circuit.

    args:
        qcircuit : The quantum circuit object containing instructions.

    returns:
        Expected matrix resulting from applying the gates.

    raises:
        ValueError: If any gate in the circuit is not a SingleQubitGate.
    """
    from sympy import N

    from mpqp.core.instruction.gates.gate import Gate, SingleQubitGate

    gates = [
        instruction
        for instruction in qcircuit.instructions
        if isinstance(instruction, Gate)
    ]
    nb_qubits = qcircuit.nb_qubits

    result_matrix = np.eye(2**nb_qubits, dtype=complex)

    for gate in reversed(gates):
        if not isinstance(gate, SingleQubitGate):
            raise ValueError(
                f"Unsupported gate: {type(gate)} only SingleQubitGate can be computed for now"
            )
        matrix = np.eye(2**nb_qubits, dtype=complex)
        gate_matrix = gate.to_matrix()
        index = gate.targets[0]
        matrix = np.kron(
            np.eye(2**index, dtype=complex),
            np.kron(gate_matrix, np.eye(2 ** (nb_qubits - index - 1), dtype=complex)),
        )

        result_matrix = np.dot(result_matrix, matrix)

    return np.vectorize(N)(result_matrix).astype(complex)


def replace_custom_gate(
    custom_unitary: CircuitInstruction, nb_qubits: int
) -> tuple[QuantumCircuit, float]:
    """Decompose and replace the (custom) qiskit unitary given in parameter by a
    qiskit `QuantumCircuit` composed of ``U`` and ``CX`` gates.

    Note:
        When using Qiskit, a global phase is introduced (related to usage of
        ``u`` in OpenQASM2). This may be problematic in some cases, so this
        function also returns the global phase introduced so it can be corrected
        later on.

    Args:
        custom_unitary: instruction containing the custom unitary operator.
        nb_qubits: Number of qubits of the circuit from which the unitary
            instruction was taken.

    Returns:
        A circuit containing the decomposition of the unitary in terms
        of gates ``U`` and ``CX``, and the global phase used to
        correct the statevector if need be.
    """
    from qiskit.exceptions import QiskitError

    transpilation_circuit = QuantumCircuit(nb_qubits)
    transpilation_circuit.append(custom_unitary)
    try:
        transpiled = transpile(transpilation_circuit, basis_gates=['u', 'cx'])
    except QiskitError as e:
        # if the error is arising from TwoQubitWeylDecomposition, we replace the
        # matrix by the closest unitary
        if "TwoQubitWeylDecomposition" in str(e):
            custom_unitary.operation.params[0] = closest_unitary(
                custom_unitary.operation.params[0]
            )
            transpiled = transpile(transpilation_circuit, basis_gates=['u', 'cx'])
        else:
            raise e
    return transpiled, transpiled.global_phase
