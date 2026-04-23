from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.gate import Gate, SingleQubitGate
from mpqp.core.instruction.gates.native_gates import (
    NATIVE_GATES,
    PRX,
    TOF,
    CRk,
    P,
    ComposedGate,
    Rk,
    RotationGate,
    Rx,
    Ry,
    Rz,
    U,
)
from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
from mpqp.core.languages import Language
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
    from braket.circuits import Circuit as braket_Circuit
    from cirq.circuits.circuit import Circuit as cirq_Circuit
    from qiskit.circuit import QuantumCircuit
    from qiskit._accelerate.circuit import CircuitInstruction


def random_circuit(
    gate_classes: Optional[Sequence[type[Gate]]] = None,
    nb_qubits: int = 5,
    nb_gates: Optional[int] = None,
    use_all_qubits: bool = False,
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
        nb_gates = int(rng.integers(5, 10))

    qcircuit = QCircuit(nb_qubits)
    for _ in range(nb_gates):
        qcircuit.add(random_gate(gate_classes, nb_qubits, rng))
    if use_all_qubits:  # used in case we want to test braket
        from mpqp.gates import H

        for i in range(nb_qubits):
            qcircuit.add(H(i))
    return qcircuit


def statevector_from_random_circuit(
    nb_qubits: int = 5,
    seed: Optional[int] = None,
) -> npt.NDArray[np.complex128]:
    """
    This function creates a statevector with a specified number of qubits,
    generated from a random circuit executed on IBM AER Simulator.
    The QCircuit is generated randomly and his statevector is calculated.

    args:
        nb_qubits : Number of qubits in the circuit.
        seed: Seed used to initialize the random number generation.

    Returns:
        The statevector with the specified number of qubits

    Examples:
        >>> pprint(statevector_from_random_circuit(2, seed=123)) # doctest: +NORMALIZE_WHITESPACE
        [0.70711, 0, 0.26893-0.65397j, 0]
    """
    from mpqp.execution import IBMDevice, Result, run

    mpqp_circ = random_circuit(None, nb_qubits, None, seed=seed)
    res = run(mpqp_circ, IBMDevice.AER_SIMULATOR_STATEVECTOR)
    if TYPE_CHECKING:
        assert isinstance(res, Result)
    return res.state_vector.vector


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
                    np.round(rng.uniform(0, 2 * np.pi), 5),
                    np.round(rng.uniform(0, 2 * np.pi), 5),
                    np.round(rng.uniform(0, 2 * np.pi), 5),
                    target,
                )
            elif issubclass(gate_class, Rk):
                return Rk(int(rng.integers(1, 10)), target)
            elif issubclass(gate_class, PRX):
                return gate_class(
                    np.round(rng.uniform(0, 2 * np.pi), 5),
                    np.round(rng.uniform(0, 2 * np.pi), 5),
                    target,
                )
            elif issubclass(gate_class, RotationGate):
                if TYPE_CHECKING:
                    assert issubclass(gate_class, (Rx, Ry, Rz, P))
                return gate_class(np.round(rng.uniform(0, 2 * np.pi), 5), target)
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
                int(rng.integers(1, 10)),
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
    custom_unitary: "CircuitInstruction", nb_qubits: int, targets: list[int]
) -> "tuple[QuantumCircuit, float]":
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
    from qiskit.circuit.library import UnitaryGate

    transpilation_circuit = QuantumCircuit(nb_qubits)
    transpilation_circuit.append(custom_unitary)
    try:
        transpiled = transpile(
            transpilation_circuit, basis_gates=['u3', 'cx'], optimization_level=0
        )
    except QiskitError as e:
        # if the error is arising from TwoQubitWeylDecomposition, we replace the
        # matrix by the closest unitary
        if "TwoQubitWeylDecomposition" in str(e):
            custom_closest_unitary = UnitaryGate(closest_unitary(custom_unitary.matrix))
            transpilation_circuit = QuantumCircuit(nb_qubits)
            transpilation_circuit.unitary(
                custom_closest_unitary, list(reversed(targets))
            )
            transpiled = transpile(
                transpilation_circuit,
                basis_gates=['u1', 'u2', 'u3', 'cx'],
                optimization_level=0,
            )
        else:
            raise e
    return transpiled, transpiled.global_phase


def mpqp_to_qiskit(
    circuit: QCircuit,
    skip_pre_measure: bool = False,
    skip_measurements: bool = False,
    printing: bool = False,
    authorized_gates: set[type[Gate]] = set(),
) -> QuantumCircuit:
    from qiskit.circuit import Operation, QuantumCircuit
    from qiskit.circuit.quantumcircuit import CircuitInstruction
    from qiskit.quantum_info import Operator

    from mpqp.core.instruction import (
        Measure,
        Breakpoint,
        CustomGate,
        Barrier,
        ControlledGate,
        BasisMeasure,
        ExpectationMeasure,
    )

    # to avoid defining twice the same parameter, we keep trace of the
    # added parameters, and we use those instead of new ones when they
    # are used more than once
    qiskit_parameters = set()
    if circuit.nb_cbits == 0:
        new_circ = QuantumCircuit(circuit.nb_qubits)
    else:
        new_circ = QuantumCircuit(circuit.nb_qubits, circuit.nb_cbits)

    if circuit.label is not None:
        new_circ.name = circuit.label

    for instruction in circuit.instructions:
        if isinstance(instruction, (Measure, Breakpoint)):
            continue
        options = {"printing": printing} if isinstance(instruction, CustomGate) else {}
        instr = [instruction]
        if isinstance(instruction, ComposedGate):
            if len(authorized_gates) != 0:
                if type(instruction) not in authorized_gates:
                    instr = instruction.decompose()
                    if any(type(gate) not in authorized_gates for gate in instr):
                        raise ValueError(
                            f"The gate {type(instruction)} and it's decomposition f{[type(g) for g in instr]} are not in the set: f{authorized_gates}"
                        )
                else:
                    instr = [instruction]
            else:
                instr = instruction.decompose()
        for instruction in instr:
            qiskit_inst = instruction.to_other_language(
                Language.QISKIT, qiskit_parameters, **options
            )
            if isinstance(qiskit_inst, list):
                for inst in qiskit_inst:
                    if TYPE_CHECKING:
                        assert isinstance(
                            inst, (CircuitInstruction, Operation, Operator)
                        )
                    cargs = []
                    if isinstance(instruction, CustomGate):
                        if TYPE_CHECKING:
                            assert isinstance(inst, Operator)
                        if printing and len(instruction.free_symbols) > 0:
                            new_circ.append(inst, list(reversed(instruction.targets)))
                        else:
                            new_circ.unitary(
                                inst,
                                list(reversed(instruction.targets)),
                                instruction.label,
                            )
                    else:
                        if isinstance(instruction, ControlledGate):
                            qargs = list(reversed(instruction.controls)) + list(
                                reversed(instruction.targets)
                            )
                        elif isinstance(instruction, Gate):
                            qargs = list(reversed(instruction.targets))
                        elif isinstance(instruction, Barrier):
                            qargs = range(circuit.nb_qubits)
                        else:
                            raise ValueError(f"Instruction not handled: {instruction}")

                        if TYPE_CHECKING:
                            assert not isinstance(inst, Operator)
                        new_circ.append(
                            inst,
                            list(qargs),
                            cargs,
                        )
            else:
                if TYPE_CHECKING:
                    assert isinstance(
                        qiskit_inst, (CircuitInstruction, Operation, Operator)
                    )
                cargs = []

                if isinstance(instruction, CustomGate):
                    if TYPE_CHECKING:
                        assert isinstance(qiskit_inst, Operator)
                    if printing and len(instruction.free_symbols) > 0:
                        new_circ.append(
                            qiskit_inst, list(reversed(instruction.targets))
                        )
                    else:
                        new_circ.unitary(
                            qiskit_inst,
                            list(reversed(instruction.targets)),
                            instruction.label,
                        )
                else:
                    qargs = []
                    if isinstance(instruction, ControlledGate):
                        qargs = list(reversed(instruction.controls)) + list(
                            reversed(instruction.targets)
                        )
                    elif isinstance(instruction, Gate):
                        qargs = list(reversed(instruction.targets))
                    elif isinstance(instruction, Barrier):
                        qargs = range(circuit.nb_qubits)
                    else:
                        raise ValueError(f"Instruction not handled: {instruction}")

                    if TYPE_CHECKING:
                        assert not isinstance(qiskit_inst, Operator)
                    new_circ.append(
                        qiskit_inst,
                        list(qargs),
                        cargs,
                    )
    for measurement in circuit.measurements:
        if not skip_pre_measure:

            for pre_measure in measurement.pre_measure:
                cargs = []
                qiskit_pre_measure = pre_measure.to_other_language(
                    Language.QISKIT, qiskit_parameters
                )
                new_circ.append(
                    qiskit_pre_measure,
                    list(reversed(pre_measure.targets)),
                    cargs=cargs,
                )
        if not skip_measurements:
            if isinstance(measurement, ExpectationMeasure):
                continue
            qiskit_inst = measurement.to_other_language(
                Language.QISKIT, qiskit_parameters
            )
            if isinstance(measurement, BasisMeasure):
                if TYPE_CHECKING:
                    assert measurement.c_targets is not None
            else:
                raise ValueError(f"measurement not handled: {measurement}")

            if TYPE_CHECKING:
                assert not isinstance(qiskit_inst, Operator)
            new_circ.append(
                qiskit_inst,
                [measurement.targets],
                [measurement.c_targets],
            )

    new_circ.global_phase += (
        circuit.input_g_phase
        + circuit._generated_g_phase  # type: ignore[reporPrivateUsage]
    )
    return new_circ


def mpqp_to_braket(
    circuit: QCircuit,
    skip_pre_measure: bool = False,
    skip_measurements: bool = False,
    authorized_gates: set[type[Gate]] = set(),
) -> braket_Circuit:
    from mpqp.execution.providers.aws import apply_noise_to_braket_circuit
    from mpqp.core.instruction import (
        Measure,
        Breakpoint,
        Barrier,
        ControlledGate,
        BasisMeasure,
        ComposedGate,
    )

    if len(circuit.noises) != 0:
        if any(isinstance(instr, CRk) for instr in circuit.instructions):
            raise NotImplementedError(
                "Cannot simulate noisy circuit with CRk gate due to "
                "an error on AWS Braket side."
            )
    from braket.circuits import Circuit as BracketCircuit

    braket_circuit = BracketCircuit()
    for instruction in circuit.instructions:
        targets = [target for target in instruction.targets]
        if isinstance(instruction, (Barrier, Breakpoint)):
            continue
        if isinstance(instruction, Measure):
            if not skip_pre_measure:
                for pre_measure in instruction.pre_measure:
                    bracket_pre_measure = pre_measure.to_other_language(Language.BRAKET)
                    braket_circuit.add(bracket_pre_measure, targets)
            if not skip_measurements:
                if isinstance(instruction, BasisMeasure) and instruction.shots != 0:
                    braket_circuit.measure(targets)
            continue
        gates = [instruction]
        if isinstance(instruction, ComposedGate):
            if len(authorized_gates) != 0:
                if type(instruction) not in authorized_gates:
                    gates = instruction.decompose()
                    if any(type(gate) not in authorized_gates for gate in gates):
                        raise ValueError(
                            f"Gate: {type(instruction)} and its decomposition {[type(g) for g in gates]} is not included in the gate set {authorized_gates}."
                        )
                else:
                    gates = [instruction]
            else:
                gates = instruction.decompose()
        else:
            gates = [instruction]
        for instruction in gates:
            braket_instr = instruction.to_other_language(Language.BRAKET)
            try:
                targets = [target for target in instruction.targets]
                if isinstance(instruction, ControlledGate):
                    targets = [control for control in instruction.controls] + targets
                braket_circuit.add_instruction(braket_instr, target=targets)
            except Exception as e:
                raise ValueError(
                    f"{type(braket_instr)}{braket_instr} cannot be added to the braket circuit: {e}"
                )
    if len(circuit.noises) != 0:
        braket_circuit = apply_noise_to_braket_circuit(
            braket_circuit,
            circuit.noises,
            circuit.nb_qubits,
        )
    return braket_circuit


def mpqp_to_cirq(
    circuit: QCircuit,
    skip_pre_measure: bool = False,
    skip_measurements: bool = False,
    authorized_gates: set[type[Gate]] | None = None,
) -> cirq_Circuit:
    from cirq.circuits.circuit import Circuit as CirqCircuit
    from cirq.ops.identity import I
    from cirq.ops.named_qubit import NamedQubit
    from mpqp.core.instruction import (
        Measure,
        Breakpoint,
        CustomGate,
        Barrier,
        ControlledGate,
        CustomControlledGate,
        ExpectationMeasure,
        ComposedGate,
    )

    if authorized_gates is None:
        authorized_gates = set()
    cirq_qubits = [NamedQubit(f"q_{i}") for i in range(circuit.nb_qubits)]
    cirq_circuit = CirqCircuit()

    for qubit in cirq_qubits:
        cirq_circuit.append(I(qubit))

    for instruction in circuit.instructions:
        if not skip_pre_measure:
            if isinstance(instruction, Measure):
                for pre_measure in instruction.pre_measure:
                    if isinstance(pre_measure, (CustomGate, CustomControlledGate)):
                        qasm2_code, gphase = pre_measure.to_other_language(
                            Language.QASM2
                        )  # pyright: ignore[reportGeneralTypeIssues]
                        if TYPE_CHECKING:
                            assert isinstance(qasm2_code, str)
                        from mpqp.qasm.qasm_to_cirq import qasm2_to_cirq_Circuit

                        qasm2_code = (
                            "OPENQASM 2.0;"
                            + "\ninclude \"qelib1.inc\";"
                            + f"\nqreg q[{circuit.nb_qubits}];\n"
                            + qasm2_code
                        )
                        custom_cirq_circuit = qasm2_to_cirq_Circuit(qasm2_code)
                        cirq_circuit += custom_cirq_circuit
                        # TODO: handle gphase in the circuit
                        circuit._generated_g_phase += gphase  # type: ignore[reporPrivateUsage]
                    else:
                        cirq_pre_measure = pre_measure.to_other_language(Language.CIRQ)
                        targets = []
                        for target in pre_measure.targets:
                            targets.append(cirq_qubits[target])
                        cirq_circuit.append(cirq_pre_measure.on(*targets))
        if isinstance(instruction, ComposedGate):
            if len(authorized_gates) != 0:
                if type(instruction) not in authorized_gates:
                    gates = instruction.decompose()
                    if any(type(gate) not in authorized_gates for gate in gates):
                        raise ValueError("")
                else:
                    gates = [instruction]
            else:
                gates = instruction.decompose()
        else:
            gates = [instruction]
        for gate in gates:
            if isinstance(gate, (ExpectationMeasure, Barrier, Breakpoint)):
                continue
            elif isinstance(gate, ControlledGate):
                targets = []
                for target in gate.targets:
                    targets.append(cirq_qubits[target])
                controls = []
                for control in gate.controls:
                    controls.append(cirq_qubits[control])
                cirq_instruction = gate.to_other_language(Language.CIRQ)
                cirq_circuit.append(cirq_instruction.on(*controls, *targets))
            else:
                if skip_measurements and isinstance(gate, Measure):
                    continue
                targets = []
                for target in gate.targets:
                    targets.append(cirq_qubits[target])
                cirq_instruction = gate.to_other_language(Language.CIRQ)
                if TYPE_CHECKING:
                    assert cirq_instruction
                cirq_circuit.append(cirq_instruction.on(*targets))

    if circuit.noises:
        from mpqp.execution.providers.google import apply_noise_to_cirq_circuit

        return apply_noise_to_cirq_circuit(
            cirq_circuit,
            circuit.noises,
        )

    return cirq_circuit
