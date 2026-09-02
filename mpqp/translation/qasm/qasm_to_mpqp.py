from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, TypeAlias
from venv import logger

import numpy as np
from ply.lex import lex

if TYPE_CHECKING:
    from mpqp.core.circuit import QCircuit

from mpqp.core.instruction import Barrier
from mpqp.gates import *
from mpqp.measures import *
from mpqp.translation.qasm.lexer_utils import *
from mpqp.translation.qasm.open_qasm_2_and_3 import (
    remove_include_and_comment,
    remove_user_gates,
)

# TODO:
# if: not handle
# barrier: handled for all qubits ("q"), not for multiple qubits ("q[0],q[1]")
# no ID name handle for qreg or creg

lexer = None
Numeric: TypeAlias = int | float | complex


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
        len(tokens) < 3
        or tokens[0].type != 'OPENQASM'
        or tokens[1].type != 'REALN'
        or tokens[1].value != 2.0
        or tokens[2].type != 'SEMICOLON'
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


def _TokenSwitch(circuit: QCircuit, tokens: list[LexToken], idx: int) -> int:
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
        return _TokenGate(circuit, tokens, idx)
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

    c_targets = None
    if tokens[idx].type == 'ARROW':
        c_targets = []
        idx += 1
        while tokens[idx].type != 'SEMICOLON':
            if tokens[idx].type == 'ID' and tokens[idx + 1].type == 'SEMICOLON':
                c_targets = None
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


def _TokenGate(circuit: QCircuit, tokens: list[LexToken], idx: int) -> int:
    token = tokens[idx]
    idx += 1
    token_value = token.value.lower()
    if token_value in single_qubits_gate_qasm:
        return _Gate_single_qubits(circuit, token_value, tokens, idx)
    elif token_value in two_qubits_gate_qasm:
        return _Gate_two_qubits(circuit, token_value, tokens, idx)
    elif token_value in one_parametrized_gate_qasm:
        return _Gate_one_parametrized(circuit, token_value, tokens, idx)
    elif token_value in u_gate_qasm:
        return _Gate_U(circuit, token_value, tokens, idx)
    elif token_value in two_qubits_parametrized_gate_qasm:
        return _Gate_two_qubits_parametrized(circuit, token_value, tokens, idx)
    elif token_value == "ccx":
        return _Gate_tof(circuit, tokens, idx)
    else:
        raise SyntaxError(f"TokenGate: {idx} {token.value}")


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
    circuit: QCircuit, gate_str: str, tokens: list[LexToken], idx: int
) -> int:
    if tokens[idx].type != 'LPAREN':
        raise SyntaxError(f"Gate_one_parametrized: {idx} {tokens[idx]}")
    idx += 1
    parameter, idx = _eval_expr(tokens, idx)
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


def _evaluate_numeric_expression(node: ast.AST) -> Numeric:
    """Evaluate the arithmetic subset accepted in OpenQASM gate parameters."""
    if isinstance(node, ast.Expression):
        return _evaluate_numeric_expression(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "np" and node.attr == "pi":
            return np.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_numeric_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_numeric_expression(node.left)
        right = _evaluate_numeric_expression(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise SyntaxError("Unsupported OpenQASM numeric expression")


def _eval_expr(tokens: list[LexToken], idx: int) -> tuple[Any, int]:

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
        elif tokens[idx].type == 'ID' and tokens[idx].value == 'e':
            expr += 'e'
            idx += 1

            if tokens[idx].type in ('PLUS', 'MINUS'):
                expr += tokens[idx].value
                idx += 1

            if tokens[idx].type not in ('INTN', 'REALN'):
                raise SyntaxError("Invalid scientific notation")

            expr += str(tokens[idx].value)
        elif check_num_expr(tokens[idx].type):
            raise SyntaxError(f"not a nb or expr: {idx}, {tokens[idx]}")
        elif tokens[idx].type == 'PI':
            expr += "np.pi"
        else:
            expr += str(tokens[idx].value)
        idx += 1
    try:
        parsed_expr = ast.parse(expr.replace("^", "**"), mode="eval")
        return _evaluate_numeric_expression(parsed_expr), idx + 1
    except (ArithmeticError, SyntaxError, TypeError, ValueError) as error:
        raise SyntaxError(f"Invalid OpenQASM numeric expression: {expr}") from error


def _Gate_one_parametrized(
    circuit: QCircuit, gate_str: str, tokens: list[LexToken], idx: int
) -> int:
    if tokens[idx].type != 'LPAREN':
        raise SyntaxError(f"Gate_one_parametrized: {idx} {tokens[idx]}")
    idx += 1
    parameter, idx = _eval_expr(tokens, idx)

    if check_Id(tokens, idx):
        raise SyntaxError(
            f'Gate_two_qubits: {" ".join(token.value for token in tokens[idx : idx + 3])}'
        )
    target = tokens[idx + 2].value
    circuit.add(one_parametrized_gate_qasm[gate_str](parameter, target))
    return idx + 5


def _Gate_U(circuit: QCircuit, gate_str: str, tokens: list[LexToken], idx: int) -> int:
    if tokens[idx].type != 'LPAREN':
        raise SyntaxError(f"Gate_U: {idx} {tokens[idx]}")
    idx += 1

    theta, phi, lbda = 0, 0, 0
    if gate_str == 'u1':
        theta, idx = _eval_expr(tokens, idx)
    elif gate_str == 'u2':
        theta, idx = _eval_expr(tokens, idx)
        phi, idx = _eval_expr(tokens, idx)
    elif gate_str == 'u3' or gate_str == 'u' or gate_str == 'U':
        theta, idx = _eval_expr(tokens, idx)
        phi, idx = _eval_expr(tokens, idx)
        lbda, idx = _eval_expr(tokens, idx)

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

    try:
        parsed = ast.parse(expr, mode="eval").body
        if not (
            isinstance(parsed, ast.Call)
            and isinstance(parsed.func, ast.Name)
            and parsed.func.id == "CustomGate"
            and 2 <= len(parsed.args) <= 3
            and not parsed.keywords
        ):
            raise ValueError("Only a CustomGate constructor is allowed")

        matrix_call = parsed.args[0]
        if not isinstance(matrix_call, ast.Call):
            raise ValueError("The CustomGate matrix must be an array literal")
        is_array_constructor = (
            isinstance(matrix_call.func, ast.Name)
            and matrix_call.func.id == "array"
            or isinstance(matrix_call.func, ast.Attribute)
            and isinstance(matrix_call.func.value, ast.Name)
            and matrix_call.func.value.id == "np"
            and matrix_call.func.attr == "array"
        )
        if not (
            is_array_constructor
            and len(matrix_call.args) == 1
            and not matrix_call.keywords
        ):
            raise ValueError("The CustomGate matrix must be an array literal")

        matrix = np.array(ast.literal_eval(matrix_call.args[0]))
        targets = ast.literal_eval(parsed.args[1])
        label = ast.literal_eval(parsed.args[2]) if len(parsed.args) == 3 else None
        if not (
            isinstance(targets, list)
            and all(isinstance(target, int) and target >= 0 for target in targets)
            and (label is None or isinstance(label, str))
        ):
            raise ValueError("Invalid CustomGate targets or label")
        gate = CustomGate(matrix, targets, label)
    except (SyntaxError, ValueError, TypeError) as error:
        raise SyntaxError(f"Invalid MPQP custom gate pragma: {expr}") from error

    circuit.add(gate)
    return idx + 1


def parse_qasm2_gates(code: str) -> tuple[str, float]:
    from mpqp.translation.qasm.open_qasm_2_and_3 import (
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
