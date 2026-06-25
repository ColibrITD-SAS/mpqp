from __future__ import annotations

import re
from typing import TYPE_CHECKING
from venv import logger
from warnings import warn

import numpy as np
from ply.lex import lex

if TYPE_CHECKING:
    from mpqp.core.circuit import QCircuit

from mpqp.core.languages import Language
from mpqp.core.instruction import Barrier
from mpqp.gates import *
from mpqp.measures import *
from mpqp.qasm.lexer_utils import *
from mpqp.qasm.open_qasm_2_and_3 import (
    remove_include_and_comment,
    remove_user_gates,
)

# TODO:
# if: not handle
# barrier: handled for all qubits ("q"), not for multiple qubits ("q[0],q[1]")
# no ID name handle for qreg or creg

lexer = None
openqasm3_variables = None


def lex_openqasm(input_string: str) -> list[LexToken]:
    global lexer
    if lexer is None:
        lexer = lex()
    lexer.input(input_string)
    tokens = []
    tok = lexer.token()
    while tok:
        tokens.append(tok)
        tok = lexer.token()
    return tokens


def qasm2_parse(input_string: str) -> QCircuit:
    """
    Parses an OpenQASM 2.0 formatted string and returns a MPQP circuit.

    Args:
        input_string: The OpenQASM 2.0 source code to be parsed.

    Returns:
        QCircuit object representing the parsed QASM input.

    Raises:
        SyntaxError: If the input string does not conform to OpenQASM 2.0 format or contains syntactical issues.

    Example:
        >>> qasm_code = '''
        ... OPENQASM 2.0;
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q -> c;
        ... '''
        >>> print(qasm2_parse(qasm_code)) # doctest: +NORMALIZE_WHITESPACE
             ┌───┐     ┌─┐
        q_0: ┤ H ├──■──┤M├───
             └───┘┌─┴─┐└╥┘┌─┐
        q_1: ─────┤ X ├─╫─┤M├
                  └───┘ ║ └╥┘
        c: 2/═══════════╩══╩═
                        0  1

    """
    from mpqp.core.circuit import QCircuit

    input_string = remove_user_gates(input_string, skip_qelib1=True)
    input_string, gphase = remove_include_and_comment(input_string)
    tokens = lex_openqasm(input_string)

    if (
        tokens[0].type != 'OPENQASM'
        and tokens[1].type != 'REALN'
        and tokens[1].value != '2.0'
        and tokens[2].type != 'SEMICOLON'
    ):
        raise SyntaxError('Invalid OpenQASM, must start with OPENQASM 2.0;')

    idx = 3
    circuit = QCircuit()
    circuit.input_g_phase = gphase
    i_max = len(tokens)
    while idx < i_max:
        logger.debug(circuit)
        logger.debug('================================')
        logger.debug('new line:', tokens[idx].value, idx)
        idx = _TokenSwitch(circuit, tokens, idx)

    return circuit


def convert_qasm3_instructions(string: str) -> tuple[str, list[str]]:
    """This function iterates through the program line by line and removes qasm3 specific instructions before parsing.
    Here are handled the following features:
        - Merge measurement instructions for better MPQP translation
        - Compile and returns all of the variables in the program
        - Handle braket custom gates
    """
    vars = []
    warning_string = ""
    for line in string.split("\n"):
        line = line.lstrip()
        if line.startswith("input"):
            vars.append(line.split(' ')[2][:-1])
            string = string.replace(line, "")
        elif line.startswith("qubit") or line.startswith("bit"):
            string = string.replace(line, "")
        elif "measure" in line:
            splitted = line.split(" ")
            c = splitted[0]
            q = splitted[-1][:-1]
            string = string.replace(line, f"measure {q} -> {c};")
        elif line.startswith("#pragma"):
            from mpqp.qasm.qasm_to_braket import braket_custom_gates_to_mpqp

            custom_gate = braket_custom_gates_to_mpqp(line)
            string = string.replace(
                line, "#pragma mpqp" + repr(custom_gate).replace('\n', ' ') + "\n"
            )
    if warning_string != "":
        warn(warning_string)
    return (string, vars)


def transpile_qasm3_circuit(input: str, language: Language) -> str:
    from mpqp.qasm.open_qasm_2_and_3 import (
        qasm_code,
        Instr,
        _replace_header,  # pyright: ignore[reportPrivateUsage]
    )

    if re.search(r"if\s*\(.*?\)\s*{[^}]*}", input, flags=re.DOTALL):
        raise ValueError("\"If\" instructions aren't handled")
    if language not in {Language.QISKIT, Language.BRAKET}:
        return input
    if language == Language.QISKIT:

        lines = input.split(";")
        lines.insert(1, "\n" + qasm_code(Instr.QISKIT_CUSTOM_INCLUDE))
        input = ";".join(lines)
    elif language == Language.BRAKET:
        from mpqp.qasm.open_qasm_2_and_3 import qasm_code

        lines = input.split(";")
        lines.insert(1, "\n" + qasm_code(Instr.BRAKET_INVERSE_CUSTOM_INCLUDE))
        input = ";".join(lines)

    input = _replace_header(input)
    input = remove_user_gates(input, language=language)
    return input


def qasm3_parse(input_string: str, language: Language = Language.QASM3) -> QCircuit:
    from mpqp.core.circuit import QCircuit

    input_string = transpile_qasm3_circuit(input_string, language=language)
    # input_string = remove_user_gates(input_string, language=language)
    input_string, gphase = remove_include_and_comment(input_string)
    input_string, var = convert_qasm3_instructions(input_string)
    tokens = lex_openqasm(input_string)
    if (
        tokens[0].type != 'OPENQASM'
        and tokens[1].type != 'REALN'
        and tokens[1].value != '3.0'
        and tokens[2].type != 'SEMICOLON'
    ):
        raise SyntaxError('Invalid OpenQASM, must start with OPENQASM 3.0;')

    idx = 3
    circuit = QCircuit()
    circuit.input_g_phase = gphase
    i_max = len(tokens)
    while idx < i_max:
        logger.debug(circuit)
        logger.debug('================================')
        logger.debug('new line:', tokens[idx].value, idx)
        idx = _TokenSwitch(circuit, tokens, idx, var)

    return circuit


def _TokenSwitch(
    circuit: QCircuit, tokens: list[LexToken], idx: int, var: list[str] = []
) -> int:
    token = tokens[idx]
    if token.type == 'QREG':
        return _TokenQREG(circuit, tokens, idx)
    elif token.type == 'CREG':
        return _TokenCREG(circuit, tokens, idx)
    elif token.type == 'MEASURE':
        return _TokenMeasure(circuit, tokens, idx)
    elif token.type == 'BARRIER':
        return _TokenBarrier(circuit, tokens, idx)
    elif token.type == 'ID':
        return _TokenGate(circuit, tokens, idx, var)
    elif token.type == 'PRAGMA_MPQP':
        return _TokenCustom(circuit, tokens, idx)
    else:
        raise SyntaxError(f"Invalid token: {idx} {token.type}")


def _TokenQREG(circuit: QCircuit, tokens: list[LexToken], idx: int) -> int:
    if idx < len(tokens) and tokens[idx].type == 'QREG':
        if check_Id(tokens, idx + 1) and tokens[idx + 5].type != 'SEMICOLON':
            raise SyntaxError(
                'must  have a qreg with the number of qubit such as "qreg ID[INTN];": '
                + f'{" ".join(str(token.value) for token in tokens[idx : idx + 5])}'
            )
        circuit.nb_qubits += tokens[idx + 3].value
    return idx + 6


def _TokenCREG(circuit: QCircuit, tokens: list[LexToken], idx: int) -> int:
    idx += 1
    if check_Id(tokens, idx) or tokens[idx + 4].type != 'SEMICOLON':
        raise SyntaxError(' '.join(str(token.value) for token in tokens[idx : idx + 4]))
    circuit.nb_cbits = tokens[idx + 2].value
    return idx + 5


def _TokenMeasure(circuit: QCircuit, tokens: list[LexToken], idx: int) -> int:
    targets = []
    c_targets = []
    while tokens[idx].type == 'MEASURE':
        if targets is None:
            break
        idx += 1
        while tokens[idx].type != 'SEMICOLON' and tokens[idx].type != 'ARROW':
            if tokens[idx].type == 'ID' and tokens[idx + 1].type == 'ARROW':
                targets = None
                idx += 1
                break
            if check_Id(tokens, idx):
                raise SyntaxError(
                    ' '.join(str(token.value) for token in tokens[idx : idx + 4])
                )
            targets.append(tokens[idx + 2].value)
            idx += 4
            if tokens[idx].type == 'COMMA':
                idx += 1

        if tokens[idx].type == 'ARROW':
            idx += 1
            while tokens[idx].type != 'SEMICOLON':
                if tokens[idx].type == 'ID' and tokens[idx + 1].type == 'SEMICOLON':
                    c_targets = []
                    idx += 1
                    break
                if check_Id(tokens, idx):
                    raise SyntaxError(
                        ' '.join(str(token.value) for token in tokens[idx : idx + 4])
                    )
                c_targets.append(tokens[idx + 2].value)
                idx += 4
                if tokens[idx].type == 'COMMA':
                    idx += 1
        if idx + 1 == len(tokens):
            break
        idx += 1
    if targets == []:
        if c_targets == []:
            raise ValueError(
                "Cannot have a dynamic sized quantum register and a set sized classical register."
            )
        circuit.add(BasisMeasure())
    else:
        if c_targets == []:
            circuit.add(BasisMeasure(targets))
        else:
            circuit.add(BasisMeasure(targets, c_targets))
    return idx + 1


def _TokenBarrier(circuit: QCircuit, tokens: list[LexToken], idx: int) -> int:
    idx += 2
    while (
        tokens[idx].type != 'SEMICOLON'
    ):  # 3M-TODO: to be removed if we handle multi target
        idx += 1
    if tokens[idx].type != 'SEMICOLON':
        raise SyntaxError(f"Barrier: {idx} {tokens[idx]}")
    circuit.add(Barrier())
    return idx + 1


def _TokenGate(
    circuit: QCircuit, tokens: list[LexToken], idx: int, var: list[str] = []
) -> int:
    token = tokens[idx]
    idx += 1
    token_value = token.value.lower()
    if token_value in single_qubits_gate_qasm:
        return _Gate_single_qubits(circuit, token_value, tokens, idx)
    elif token_value in two_qubits_gate_qasm:
        return _Gate_two_qubits(circuit, token_value, tokens, idx)
    elif token_value in one_parametrized_gate_qasm:
        return _Gate_one_parametrized(circuit, token_value, tokens, idx, var)
    elif token_value in u_gate_qasm:
        return _Gate_U(circuit, token_value, tokens, idx, var)
    elif token_value in two_qubits_parametrized_gate_qasm:
        return _Gate_two_qubits_parametrized(circuit, token_value, tokens, idx, var)
    elif token_value == "ccx":
        return _Gate_tof(circuit, tokens, idx)
    else:
        raise ValueError(
            f"Gate is not defined/handled at the time of usage: {token_value}"
        )


def _Gate_single_qubits(
    circuit: QCircuit, gate_str: str, tokens: list[LexToken], idx: int
) -> int:
    if tokens[idx].type == 'ID' and tokens[idx + 1].type == 'SEMICOLON':
        for i in range(circuit.nb_qubits):
            circuit.add(single_qubits_gate_qasm[gate_str](i))
        return idx + 2

    multi = False
    while tokens[idx].type != 'SEMICOLON':
        if multi:
            if tokens[idx].type != 'COMMA':
                raise SyntaxError(f"Gate_single_qubits: {idx} {tokens[idx]}")
            idx += 1
        if check_Id(tokens, idx):
            raise SyntaxError(
                f'Gate_single_qubits: {" ".join(str(token.value) for token in tokens[idx : idx + 4])}'
            )
        circuit.add(single_qubits_gate_qasm[gate_str](tokens[idx + 2].value))
        idx += 4
        multi = True
    return idx + 1


def _Gate_two_qubits_parametrized(
    circuit: QCircuit,
    gate_str: str,
    tokens: list[LexToken],
    idx: int,
    var: list[str] = [],
) -> int:
    if tokens[idx].type != 'LPAREN':
        raise SyntaxError(f"Gate_one_parametrized: {idx} {tokens[idx]}")
    idx += 1
    parameter, idx = _eval_expr(tokens, idx, var)
    if (
        check_Id(tokens, idx)
        or tokens[idx + 4].type != 'COMMA'
        or check_Id(tokens, idx + 5)
    ):
        raise SyntaxError(
            f'Gate_two_qubits: {" ".join(str(token.value) for token in tokens[idx : idx + 10])}'
        )

    control = tokens[idx + 2].value
    target = tokens[idx + 7].value
    circuit.add(two_qubits_parametrized_gate_qasm[gate_str](parameter, control, target))
    return idx + 10


def _Gate_two_qubits(
    circuit: QCircuit, gate_str: str, tokens: list[LexToken], idx: int
) -> int:
    if (
        check_Id(tokens, idx)
        or tokens[idx + 4].type != 'COMMA'
        or check_Id(tokens, idx + 5)
    ):
        raise SyntaxError(
            f'Gate_two_qubits: {" ".join(str(token.value) for token in tokens[idx : idx + 10])}'
        )

    control = tokens[idx + 2].value
    target = tokens[idx + 7].value
    circuit.add(two_qubits_gate_qasm[gate_str](control, target))
    return idx + 10


def _Gate_tof(circuit: QCircuit, tokens: list[LexToken], idx: int) -> int:

    if (
        check_Id(tokens, idx)
        or tokens[idx + 4].type != 'COMMA'
        or check_Id(tokens, idx + 5)
    ):
        raise SyntaxError(
            f'Gate_tof: {" ".join(str(token.value) for token in tokens[idx : idx + 10])}'
        )

    qubits = []
    multi = False
    while tokens[idx].type != 'SEMICOLON':
        if multi:
            if tokens[idx].type != 'COMMA':
                raise SyntaxError(f"Gate_single_qubits: {idx} {tokens[idx]}")
            idx += 1
        if check_Id(tokens, idx):
            raise SyntaxError(
                f'_Gate_tof: {" ".join(str(token.value) for token in tokens[idx : idx + 4])}'
            )
        qubits.append(tokens[idx + 2].value)
        idx += 4
        multi = True

    if len(qubits) < 2:
        raise SyntaxError("TOF: missing control or target qubit")

    control = qubits[:-1]
    target = qubits[-1]
    circuit.add(TOF(control, target))
    return idx + 1


def _eval_expr(
    tokens: list[LexToken], idx: int, var: list[str] = []
) -> tuple[Any, int]:
    import numpy as np  # pyright: ignore[reportUnusedImport]

    expr = ""
    open_paren = 0
    while tokens[idx].type != 'COMMA' and (
        tokens[idx].type != 'RPAREN' or open_paren > 0
    ):
        if tokens[idx].type == 'LPAREN':
            open_paren += 1
            expr += "("
        elif tokens[idx].type == 'RPAREN':
            open_paren -= 1
            expr += ")"
        elif tokens[idx].type == 'ID':
            if tokens[idx].value == 'e':
                expr += 'e'
                idx += 1

                if tokens[idx].type in ('PLUS', 'MINUS'):
                    expr += tokens[idx].value
                    idx += 1
                if tokens[idx].type not in ('INTN', 'REALN'):
                    raise SyntaxError("Invalid scientific notation")

                expr += str(tokens[idx].value)
            elif tokens[idx].value in var:
                # makes sure sympy is imported when eval is called
                from sympy import Symbol  # pyright: ignore[reportUnusedImport]

                expr += f"Symbol('{tokens[idx].value}')"
            else:
                raise ValueError(f"Variable: {tokens[idx].value} not found.")

        elif check_num_expr(tokens[idx].type):
            raise SyntaxError(f"not a nb or expr: {idx}, {tokens[idx]}")
        elif tokens[idx].type == 'PI':
            expr += "np.pi"
        else:
            expr += str(tokens[idx].value)
        idx += 1
    try:
        result = eval(expr)
    except:
        raise ValueError(
            f"Expression: {expr} couldn't be evaluated, either a variable is not declared or it is not a correct expression."
        )
    return result, idx + 1


def _Gate_one_parametrized(
    circuit: QCircuit,
    gate_str: str,
    tokens: list[LexToken],
    idx: int,
    var: list[str] = [],
) -> int:
    if tokens[idx].type != 'LPAREN':
        raise SyntaxError(f"Gate_one_parametrized: {idx} {tokens[idx]}")
    idx += 1
    parameter, idx = _eval_expr(tokens, idx, var)

    if check_Id(tokens, idx):
        raise SyntaxError(
            f'Gate_two_qubits: {" ".join(token.value for token in tokens[idx : idx + 3])}'
        )
    target = tokens[idx + 2].value
    circuit.add(one_parametrized_gate_qasm[gate_str](parameter, target))
    return idx + 5


def _Gate_U(
    circuit: QCircuit,
    gate_str: str,
    tokens: list[LexToken],
    idx: int,
    var: list[str] = [],
) -> int:
    if tokens[idx].type != 'LPAREN':
        raise SyntaxError(f"Gate_U: {idx} {tokens[idx]}")
    idx += 1

    theta, phi, lbda = 0, 0, 0
    if gate_str == 'u1':
        theta, idx = _eval_expr(tokens, idx, var)
    elif gate_str == 'u2':
        theta, idx = _eval_expr(tokens, idx, var)
        phi, idx = _eval_expr(tokens, idx, var)
    elif gate_str == 'u3' or gate_str == 'u' or gate_str == 'U':
        theta, idx = _eval_expr(tokens, idx, var)
        phi, idx = _eval_expr(tokens, idx, var)
        lbda, idx = _eval_expr(tokens, idx, var)

    if check_Id(tokens, idx):
        raise SyntaxError(
            f'GateU:  {" ".join(str(token.value) for token in tokens[idx : idx + 4])}'
        )
    target = tokens[idx + 2].value
    circuit.add(U(theta, phi, lbda, target))
    return idx + 5


def _TokenCustom(circuit: QCircuit, tokens: list[LexToken], idx: int) -> int:
    raw = tokens[idx].value.strip()

    if not raw.startswith("#pragma mpqp"):
        raise SyntaxError(f"Unknown pragma: {raw}")

    expr = raw[len("#pragma mpqp") :].strip()

    safe_globals = {
        "__builtins__": {},
    }

    safe_locals = {
        "CustomGate": CustomGate,
        "array": np.array,
        "np": np,
    }

    try:
        gate = eval(expr, safe_globals, safe_locals)
    except Exception as e:
        raise SyntaxError(f"Custom gate eval failed: {expr}") from e

    circuit.add(gate)
    return idx + 1


def parse_qasm2_gates(code: str) -> tuple[str, float]:
    from mpqp.qasm.open_qasm_2_and_3 import (
        qasm_code,
        remove_user_gates,
        Instr,
        parse_gphase_instruction,
        remove_include_and_comment,
    )
    import re

    code, gphase = remove_include_and_comment(code)

    lines = code.split(";")
    lines.insert(1, qasm_code(Instr.QISKIT_CUSTOM_INCLUDE))
    code = ";".join(lines)

    code = remove_user_gates(code)

    clean_code = []
    to_add = True

    for line in code.split("\n"):
        if line.startswith("gphase"):
            match = re.match(r"\s*(\w+)\s*", line)
            if match:
                gphase = parse_gphase_instruction(gphase, line, match)
            to_add = False
        elif line.startswith(";"):
            to_add = False
        elif "//" in line:
            line = line[:13]

        if to_add == True:
            clean_code.append(line)
        to_add = True

    return "\n".join(clean_code), gphase
