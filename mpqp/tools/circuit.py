from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.random import Generator
from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from typeguard import typechecked

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.gate import Gate, SingleQubitGate
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
from mpqp.noise.noise_model import (
    NOISE_MODELS,
    AmplitudeDamping,
    BitFlip,
    Dephasing,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
)
from mpqp.tools.maths import closest_unitary

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit.quantumcircuitdata import CircuitInstruction


# @typechecked
# FIXME: Resolve type-checking errors encountered during test execution.
def random_circuit(
    gate_classes: Optional[Sequence[type[Gate]]] = None,
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

    if nb_gates is None:
        nb_gates = rng.integers(5, 10)

    qcircuit = QCircuit(nb_qubits)
    for _ in range(nb_gates):
        qcircuit.add(random_gate(gate_classes, nb_qubits, rng))
    return qcircuit


@typechecked
def random_gate(
    gate_classes: Optional[Sequence[type[Gate]]] = None,
    nb_qubits: int = 5,
    seed: Optional[int | Generator] = None,
) -> Gate:
    """This function creates a gate with a specified number of qubits.
    The gate are chosen randomly from the provided list of native gate classes.

    args:
        nb_qubits : Number of qubits in the circuit.
        gate_classes : List of native gate classes to use in the circuit.

    Returns:
        A quantum circuit with the specified number of qubits and randomly chosen gates.

    Raises:
        ValueError: If the number of qubits is too low for the specified gates.

    Examples:
        >>> random_gate([U, TOF], 3) # doctest: +SKIP
        U(2.067365317109373, 0.18652872274018245, 0.443968374745352, 0)
        >>> random_gate(nb_qubits=4) # doctest: +SKIP
        SWAP(3, 1)

    """
    rng = np.random.default_rng(seed)

    if gate_classes is None:
        gate_classes = []
        for gate in NATIVE_GATES:
            if TYPE_CHECKING:
                assert isinstance(gate.nb_qubits, int)
            if gate.nb_qubits <= nb_qubits:
                gate_classes.append(gate)

    qubits = list(range(nb_qubits))

    if any(
        not issubclass(gate, SingleQubitGate)
        and ((gate == TOF and nb_qubits <= 2) or nb_qubits <= 1)
        for gate in gate_classes
    ):
        raise ValueError("number of qubits too low for this gates")

    gate_class = rng.choice(np.array(gate_classes))
    target = rng.choice(qubits).item()

    if issubclass(gate_class, SingleQubitGate):
        if issubclass(gate_class, ParametrizedGate):
            if issubclass(gate_class, U):
                return U(
                    rng.uniform(0, 2 * np.pi),
                    rng.uniform(0, 2 * np.pi),
                    rng.uniform(0, 2 * np.pi),
                    target,
                )
            elif issubclass(gate_class, Rk):
                return Rk(rng.integers(1, 10), target)
            elif issubclass(gate_class, RotationGate):
                if TYPE_CHECKING:
                    assert issubclass(gate_class, (Rx, Ry, Rz, P))
                return gate_class(rng.uniform(0, 2 * np.pi), target)
            else:
                raise ValueError
        else:
            return gate_class(target)
    else:
        control = rng.choice(list(set(qubits) - {target})).item()
        if issubclass(gate_class, ParametrizedGate):
            if TYPE_CHECKING:
                assert issubclass(gate_class, CRk)
            return gate_class(
                rng.integers(1, 10),
                control,
                target,
            )
        elif issubclass(gate_class, TOF):
            control2 = rng.choice(list(set(qubits) - {target, control})).item()
            return TOF([control, control2], target)
        else:
            return gate_class(control, target)


def random_noise(
    noise_model: Optional[Sequence[type[NoiseModel]]] = None,
    seed: Optional[int | Generator] = None,
) -> NoiseModel:
    """This function creates a noise model.
    The noise are chosen randomly from the provided list of noise model.

    args:
        noise_model : List of noise model.

    Returns:
        A quantum circuit with the specified number of qubits and randomly chosen gates.

    Raises:
        ValueError: If the number of qubits is too low for the specified gates.

    Examples:
        >>> random_noise() # doctest: +SKIP
        Depolarizing(0.37785041428875576)

    """
    rng = np.random.default_rng(seed)

    if noise_model is None:
        noise_model = NOISE_MODELS

    noise = rng.choice(np.array(noise_model))

    if issubclass(noise, AmplitudeDamping):
        prob = rng.uniform(0, 1)
        return AmplitudeDamping(prob)
    elif issubclass(noise, BitFlip):
        prob = rng.uniform(0, 0.5)
        return BitFlip(prob)
    elif issubclass(noise, Dephasing):
        prob = rng.uniform(0, 1)
        return Depolarizing(prob)
    elif issubclass(noise, Depolarizing):
        prob = rng.uniform(0, 0.75)
        return Depolarizing(prob)
    elif issubclass(noise, PhaseDamping):
        gamma = rng.uniform(0, 1)
        return PhaseDamping(gamma)
    else:
        raise NotImplementedError(f"{noise} model not implemented")


@typechecked
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


@typechecked
def replace_custom_gate(
    custom_unitary: "CircuitInstruction", nb_qubits: int
) -> tuple["QuantumCircuit", float]:
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
    from qiskit import QuantumCircuit, transpile
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
