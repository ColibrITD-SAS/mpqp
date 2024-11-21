from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from typeguard import typechecked

if TYPE_CHECKING:
    from mpqp.core.circuit import QCircuit

from mpqp.core.instruction import Instruction
from mpqp.core.instruction.gates import *
from mpqp.core.instruction.gates.gate import SingleQubitGate
from mpqp.core.languages import Language
from mpqp.core.instruction.breakpoint import Breakpoint
from mpqp.core.instruction.measurement import ExpectationMeasure, BasisMeasure


@typechecked
def float_to_qasm_str(f: float) -> str:
    if f.is_integer():
        return str(int(f))
    elif round(f % np.pi, 5) == 0:
        return f"{int(f/np.pi)}*pi" if f != np.pi else "pi"
    else:
        return f"pi/{int(1 / f * np.pi)}" if (np.pi * (1 / f)).is_integer() else str(f)


@typechecked
def _simplify_instruction_to_qasm(
    instruction: SingleQubitGate | BasisMeasure,
    targets: dict[int, int],
    c_targets: dict[int, int],
) -> str:
    instruction_str = instruction.to_other_language(Language.QASM2)
    if TYPE_CHECKING:
        assert isinstance(instruction_str, str)
    instruction_str = "\n" + instruction_str.split(" ")[0]

    final_str = ""

    while any(target != 0 for target in targets.values()):
        final_str += instruction_str + " "
        if all(target != 0 for target in targets.values()):
            final_str += "q"
            targets = {key: value - 1 for key, value in targets.items()}
            if isinstance(instruction, BasisMeasure):
                final_str += f" -> c"
                c_targets = {key: value - 1 for key, value in c_targets.items()}
            final_str += ";"
        else:
            for key, target in targets.items():
                if target != 0:
                    final_str += f"q[{key}],"
                    targets[key] -= 1
            final_str = final_str[:-1]
            if isinstance(instruction, BasisMeasure):
                final_str += " -> "
                for key, c_target in c_targets.items():
                    if c_target != 0:
                        final_str += f"c[{key}],"
                        c_targets[key] -= 1
                final_str = final_str[:-1]
            final_str += ";"

    return final_str


def _instruction_to_qasm2(instruction: Instruction) -> tuple[str, float]:
    if isinstance(instruction, (Breakpoint, ExpectationMeasure)):
        return "", 0
    elif isinstance(instruction, CustomGate):
        qasm_str_gphase = instruction.to_other_language(Language.QASM2)
        if TYPE_CHECKING:
            assert isinstance(qasm_str_gphase, tuple)
        return "\n" + qasm_str_gphase[0], qasm_str_gphase[1]
    else:
        instruction = instruction.to_other_language(Language.QASM2)
        if TYPE_CHECKING:
            assert isinstance(instruction, str)
        return "\n" + instruction, 0


@typechecked
def mpqp_to_qasm2(qcircuit: QCircuit, simplify: bool = False) -> tuple[str, float]:
    """Converts a :class:`~mpqp.core.circuit.QCircuit` object into a string in
    QASM 2.0 format. It handles various quantum instructions like gates,
    measurements, and barriers and can optionally simplify the circuit by
    merging consecutive single-qubit gates of the same type.

    Args:
        circuit: The circuit to be converted.
        simplify: If `True`, the function will attempt to simplify the circuit
            by merging consecutive single-qubit gates of the same type.

    Returns:
        A tuple containing, QASM 2.0 string representation of the provided circuit, and
        A global phase value associated with custom gates.

    Raises:
        ValueError: If an unknown gate or instruction type is encountered during
            the conversion process.

    Example:
        >>> circuit = QCircuit([H(0), H(1), CNOT(0, 1), BasisMeasure()])
        >>> qasm_code, gphase = mpqp_to_qasm2(circuit, simplify=True)
        >>> print(qasm_code)
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q;
        cx q[0],q[1];
        measure q -> c;
    """
    if qcircuit.noises:
        logging.warning(
            "Instructions such as noise are not supported by QASM2 hence have "
            "been ignored."
        )

    qasm_str = (
        "OPENQASM 2.0;"
        + "\ninclude \"qelib1.inc\";"
        + f"\nqreg q[{qcircuit.nb_qubits}];"
    )
    qasm_measure = ""
    if qcircuit.nb_cbits != None and qcircuit.nb_cbits != 0:
        qasm_str += f"\ncreg c[{qcircuit.nb_cbits}];"

    previous = None
    targets = {i: 0 for i in range(qcircuit.nb_qubits)}
    c_targets = {i: 0 for i in range(qcircuit.nb_qubits)}
    gphase = 0

    for instruction in qcircuit.instructions:
        if simplify:
            if isinstance(instruction, (SingleQubitGate, BasisMeasure)):
                if previous is None:
                    previous = instruction
                elif type(instruction) != type(previous) or (
                    isinstance(instruction, ParametrizedGate)
                    and instruction.parameters
                    != previous.parameters  # pyright: ignore[reportAttributeAccessIssue]
                ):
                    if isinstance(previous, BasisMeasure):
                        qasm_measure += _simplify_instruction_to_qasm(
                            previous, targets, c_targets
                        )
                    else:
                        qasm_str += _simplify_instruction_to_qasm(
                            previous, targets, c_targets
                        )
                    targets = {i: 0 for i in range(qcircuit.nb_qubits)}
                    c_targets = {i: 0 for i in range(qcircuit.nb_qubits)}
                    previous = instruction

                for target in instruction.targets:
                    targets[target] += 1
                if isinstance(instruction, BasisMeasure):
                    if instruction.c_targets is not None:
                        for c_target in instruction.c_targets:
                            c_targets[c_target] += 1
                    else:
                        for i in range(len(instruction.targets)):
                            c_targets[i] += 1
            else:
                if previous:
                    if isinstance(previous, BasisMeasure):
                        qasm_measure += _simplify_instruction_to_qasm(
                            previous, targets, c_targets
                        )
                    else:
                        qasm_str += _simplify_instruction_to_qasm(
                            previous, targets, c_targets
                        )
                    previous = None
                    targets = {i: 0 for i in range(qcircuit.nb_qubits)}
                    c_targets = {i: 0 for i in range(qcircuit.nb_qubits)}
                qasm, phase = _instruction_to_qasm2(instruction)
                if isinstance(instruction, BasisMeasure):
                    qasm_measure += qasm
                else:
                    qasm_str += qasm
                gphase += phase
        else:
            qasm, phase = _instruction_to_qasm2(instruction)
            if isinstance(instruction, BasisMeasure):
                qasm_measure += qasm
            else:
                qasm_str += qasm
            gphase += phase

    if previous:
        qasm_str += _simplify_instruction_to_qasm(previous, targets, c_targets)

    qasm_str += qasm_measure

    return qasm_str, gphase
