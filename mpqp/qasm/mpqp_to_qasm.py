from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mpqp.core.instruction import Instruction
    from mpqp.core.circuit import QCircuit

from mpqp.core.instruction.barrier import Barrier
from mpqp.core.instruction.breakpoint import Breakpoint
from mpqp.core.instruction.gates import *
from mpqp.core.instruction.gates.gate import SingleQubitGate
from mpqp.core.instruction.measurement import BasisMeasure, ExpectationMeasure
from mpqp.qasm.open_qasm_2_and_3 import remove_user_gates


def _handle_measurement(instruction: BasisMeasure) -> str:
    if instruction.c_targets is None:
        return "\n".join(
            f"measure q[{target}] -> c[{i}];"
            for i, target in enumerate(instruction.targets)
        )
    else:
        return "\n".join(
            f"measure q[{target}] -> c[{c_target}];"
            for target, c_target in zip(instruction.targets, instruction.c_targets)
        )


def _handle_connection(instruction: Gate | Barrier) -> str:
    control = ""
    if isinstance(instruction, ControlledGate):
        control = ",".join([f"q[{j}]" for j in instruction.controls]) + ","
    return control + ",".join([f"q[{j}]" for j in instruction.targets])


def _float_to_str(f: float) -> str:
    if f.is_integer():
        return str(int(f))
    elif f % np.pi == 0:
        return f"{int(f/np.pi)}*pi" if f != np.pi else "pi"
    else:
        return f"pi/{int(1 / f * np.pi)}" if (np.pi * (1 / f)).is_integer() else str(f)


def _handle_rotation(gate: ParametrizedGate) -> str:
    if gate.label in {"Rk", "CRk"}:
        return "(" + _float_to_str(2 * np.pi / (2 ** float(gate.parameters[0]))) + ")"
    return (
        "(" + ",".join(_float_to_str(float(param)) for param in gate.parameters) + ")"
    )


def instruction_to_qasm(instruction: Instruction, simplify: bool = False) -> str:
    if isinstance(instruction, CustomGate):
        import collections.abc

        from qiskit.qasm2.export import (
            _define_custom_operation,  # pyright: ignore[reportPrivateUsage]
        )
        from qiskit.qasm2.export import (
            _instruction_call_site,  # pyright: ignore[reportPrivateUsage]
        )
        from qiskit.quantum_info.operators import Operator as QiskitOperator

        gates_to_define: collections.OrderedDict[str, tuple[Instruction, str]] = (
            collections.OrderedDict()
        )

        op = (
            QiskitOperator(instruction.matrix)
            .to_instruction()
            ._qasm2_decomposition()  # pyright: ignore[reportPrivateUsage]
        )
        _define_custom_operation(op, gates_to_define)

        gate_definitions_qasm = "\n".join(
            f"{qasm}" for _, qasm in gates_to_define.values()
        )
        qasm_str = remove_user_gates(
            "\n"
            + gate_definitions_qasm
            + "\n"
            + _instruction_call_site(op)
            + " "
            + _handle_connection(instruction)
            + ";"
        )

        return "\n" + qasm_str

    elif isinstance(instruction, Gate):
        instruction_str = (
            instruction.qasm2_gate  # pyright: ignore[reportAttributeAccessIssue]
        )
        if instruction_str is None:
            raise ValueError(f"Unknown gate: {type(instruction)}")
        if isinstance(instruction, ParametrizedGate):
            instruction_str += _handle_rotation(instruction)
        return "\n" + instruction_str + " " + _handle_connection(instruction) + ";"
    elif isinstance(instruction, BasisMeasure):
        return "\n" + _handle_measurement(instruction)
    elif isinstance(instruction, Barrier):
        if simplify:
            return "\nbarrier q;"
        return "\nbarrier " + _handle_connection(instruction) + ";"
    elif isinstance(instruction, Breakpoint):
        return ""
    elif isinstance(instruction, ExpectationMeasure):
        return ""
    else:
        raise ValueError(f"Unknown instruction: {repr(instruction)}")


def _simplify_instruction(instruction: SingleQubitGate, targets: dict[int, int]):
    instruction_str = (
        instruction.qasm2_gate  # pyright: ignore[reportAttributeAccessIssue]
    )
    if instruction_str is None:
        raise ValueError(f"Unknown gate: {type(instruction)}")
    if isinstance(instruction, ParametrizedGate):
        instruction_str += _handle_rotation(instruction)
    final_str = ""

    while any(target != 0 for target in targets.values()):
        final_str += "\n" + instruction_str + " "
        if all(target != 0 for target in targets.values()):
            final_str += "q;"
            targets = {key: value - 1 for key, value in targets.items()}
        else:
            for key, target in targets.items():
                if target != 0:
                    final_str += f"q[{key}],"
                    targets[key] -= 1
            final_str = final_str[:-1] + ";"

    return final_str


def mpqp_to_qasm2(circuit: QCircuit, simplify: bool = False) -> str:
    """Converts an :class:`~mpqp.core.circuit.QCircuit` object into a string in
    QASM 2.0 format. It handles various quantum instructions like gates,
    measurements, and barriers and can optionally simplify the circuit by
    merging consecutive single-qubit gates of the same type.

    Args:
        circuit: The circuit to be converted.
        simplify: If `True`, the function will attempt to simplify the circuit
            by merging consecutive single-qubit gates of the same type.

    Returns:
        A string containing the QASM 2.0 representation of the provided circuit.

    Raises:
        ValueError: If an unknown gate or instruction type is encountered during
            the conversion process.

    Example:
        >>> circuit = QCircuit([H(0), CNOT(0, 1), BasisMeasure()])
        >>> qasm_code = mpqp_to_qasm2(circuit, simplify=True)
        >>> print(qasm_code)
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    """
    circuit_copy = circuit.hard_copy()
    if circuit_copy.noises:
        logging.warning(
            "Instructions such as noise are not supported by QASM2 hence have "
            "been ignored."
        )

    qasm_str = (
        "OPENQASM 2.0;"
        + "\ninclude \"qelib1.inc\";"
        + f"\nqreg q[{circuit_copy.nb_qubits}];"
    )

    if circuit_copy.nb_cbits != None:
        qasm_str += f"\ncreg c[{circuit_copy.nb_cbits}];"

    previous = None
    targets = {i: 0 for i in range(circuit_copy.nb_qubits)}

    for instruction in circuit_copy.instructions:
        if simplify:
            if isinstance(instruction, SingleQubitGate):
                if previous is None or not isinstance(instruction, type(previous)):
                    if previous:
                        qasm_str += _simplify_instruction(previous, targets)
                        targets = {i: 0 for i in range(circuit_copy.nb_qubits)}
                    previous = instruction
                else:
                    if (
                        isinstance(instruction, ParametrizedGate)
                        and instruction.parameters
                        != previous.parameters  # pyright: ignore[reportAttributeAccessIssue]
                    ):
                        qasm_str += _simplify_instruction(previous, targets)
                        targets = {i: 0 for i in range(circuit_copy.nb_qubits)}
                        previous = instruction
                for target in instruction.targets:
                    targets[target] += 1
            else:
                if previous:
                    qasm_str += _simplify_instruction(previous, targets)
                    previous = None
                    targets = {i: 0 for i in range(circuit_copy.nb_qubits)}
                qasm_str += instruction_to_qasm(instruction, simplify)
        else:
            qasm_str += instruction_to_qasm(instruction, simplify)

    if previous:
        qasm_str += _simplify_instruction(previous, targets)
    return qasm_str
