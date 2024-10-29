from typing import TYPE_CHECKING, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.quantumcircuitdata import CircuitInstruction

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.gate import SingleQubitGate
from mpqp.core.instruction.gates.native_gates import (
    NATIVE_GATES,
    TOF,
    CRk,
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
    nb_gates: Optional[int] = None,
    seed: Optional[int] = None,
):
    """This function creates a QCircuit with a specified number of qubits and gates.
    The gates are chosen randomly from the provided list of native gate classes.

    args:
        nb_qubits : Number of qubits in the circuit.
        gate_classes : List of native gate classes to use in the circuit.
        nb_gates : Number of gates to add to the circuit.
        seed: Seed used to initialize the random number generation.

    Returns:
        A quantum circuit with the specified number of qubits and randomly chosen gates.

    Raises:
        ValueError: If the number of qubits is too low for the specified gates.

    Examples:
        >>> print(random_circuit([U, TOF], 3)) # doctest: +NORMALIZE_WHITESPACE
             ┌───┐┌───┐     ┌───┐                            ┌───┐┌───┐
        q_0: ┤ X ├┤ X ├──■──┤ X ├────────────────────────────┤ X ├┤ X ├
             └─┬─┘└─┬─┘┌─┴─┐└─┬─┘                            └─┬─┘└─┬─┘
        q_1: ──■────■──┤ X ├──■────────────────────────────────■────■──
               │    │  └─┬─┘  │  ┌──────────────────────────┐  │    │
        q_2: ──■────■────■────■──┤ U(0.31076,5.5951,6.2613) ├──■────■──
                                 └──────────────────────────┘

        >>> print(random_circuit([U, TOF], 3, seed=123)) # doctest: +NORMALIZE_WHITESPACE
                  ┌───┐                           ┌───┐
        q_0: ──■──┤ X ├───────────────────────────┤ X ├
             ┌─┴─┐└─┬─┘┌─────────────────────────┐└─┬─┘
        q_1: ┤ X ├──■──┤ U(5.1025,5.8015,1.7378) ├──■──
             └─┬─┘  │  ├─────────────────────────┤  │
        q_2: ──■────■──┤ U(5.5914,3.2231,1.5392) ├──■──
                       └─────────────────────────┘

    """
    rng = np.random.default_rng(seed)

    if gate_classes is None:
        gate_classes = []
        for gate in NATIVE_GATES:
            if TYPE_CHECKING:
                assert isinstance(gate.nb_qubits, int)
            if gate.nb_qubits <= nb_qubits:
                gate_classes.append(gate)

    if nb_gates is None:
        nb_gates = rng.integers(5, 10)

    qubits = list(range(nb_qubits))
    qcircuit = QCircuit(nb_qubits)

    if any(
        not issubclass(gate, SingleQubitGate)
        and ((gate == TOF and nb_qubits <= 2) or nb_qubits <= 1)
        for gate in gate_classes
    ):
        raise ValueError("number of qubits too low for this gates")

    for _ in range(nb_gates):
        gate_class = rng.choice(np.array(gate_classes))
        target = rng.choice(qubits).item()
        if issubclass(gate_class, SingleQubitGate):
            if issubclass(gate_class, ParametrizedGate):
                if issubclass(gate_class, U):
                    qcircuit.add(
                        gate_class(
                            rng.uniform(0, 2 * np.pi),
                            rng.uniform(0, 2 * np.pi),
                            rng.uniform(0, 2 * np.pi),
                            target,
                        )
                    )
                elif issubclass(gate_class, Rk):
                    qcircuit.add(Rk(rng.integers(1, 10), target))
                elif issubclass(gate_class, RotationGate):
                    if TYPE_CHECKING:
                        assert issubclass(gate_class, (Rx, Ry, Rz, P))
                    qcircuit.add(gate_class(rng.uniform(0, 2 * np.pi), target))
                else:
                    raise ValueError
            else:
                qcircuit.add(gate_class(target))
        else:
            control = rng.choice(list(set(qubits) - {target})).item()
            if issubclass(gate_class, ParametrizedGate):
                if TYPE_CHECKING:
                    assert issubclass(gate_class, CRk)
                qcircuit.add(gate_class(rng.integers(1, 10), control, target))
            elif issubclass(gate_class, TOF):
                control2 = rng.choice(list(set(qubits) - {target, control})).item()
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
