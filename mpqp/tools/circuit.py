import random
from typing import TYPE_CHECKING, Optional

import numpy as np

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.gate import SingleQubitGate
from mpqp.core.instruction.gates.native_gates import (
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


def random_circuit(
    gate_classes: Optional[list[type]] = None,
    nb_qubits: int = 5,
    nb_gates: int = np.random.randint(5, 10),
):
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
        >>> random_circuit(native_gates.NATIVE_GATES, 4, 10) # doctest: +SKIP
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

    if gate_classes is None:
        gate_classes = []
        for gate in NATIVE_GATES:
            if TYPE_CHECKING:
                assert isinstance(gate.nb_qubits, int)
            if gate.nb_qubits <= nb_qubits:
                gate_classes.append(gate)

    qubits = list(range(nb_qubits))
    qcircuit = QCircuit(nb_qubits)
    if any(
        not issubclass(gate, SingleQubitGate)
        and ((gate == TOF and nb_qubits <= 2) or nb_qubits <= 1)
        for gate in gate_classes
    ):
        raise ValueError("number of qubits too low for this gates")

    for _ in range(nb_gates):
        gate_class: type[NativeGate] = random.choice(gate_classes)
        target = random.choice(qubits)
        if issubclass(gate_class, SingleQubitGate):
            if issubclass(gate_class, ParametrizedGate):
                if issubclass(gate_class, U):
                    qcircuit.add(
                        gate_class(
                            random.uniform(0, 2 * np.pi),
                            random.uniform(0, 2 * np.pi),
                            random.uniform(0, 2 * np.pi),
                            target,
                        )
                    )
                elif issubclass(gate_class, Rk):
                    qcircuit.add(Rk(random.randint(0, 10), target))
                elif issubclass(gate_class, RotationGate):
                    if TYPE_CHECKING:
                        assert issubclass(gate_class, (Rx, Ry, Rz, P))
                    qcircuit.add(gate_class(random.uniform(0, 2 * np.pi), target))
                else:
                    raise ValueError
            else:
                qcircuit.add(gate_class(target))
        else:
            control = random.choice(list(set(qubits) - {target}))
            if issubclass(gate_class, ParametrizedGate):
                if TYPE_CHECKING:
                    assert issubclass(gate_class, CRk)
                qcircuit.add(
                    gate_class(
                        random.randint(0, 10),
                        control,
                        target,
                    )
                )
            elif issubclass(gate_class, TOF):
                control2 = random.choice(list(set(qubits) - {target, control}))
                qcircuit.add(TOF([control, control2], target))
            else:
                qcircuit.add(gate_class(control, target))
    return qcircuit


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
