from typing import Any

# List of token names
tokens = (
    'OPENQASM',
    'QREG',
    'CREG',
    'GATE',
    'OPAQUE',
    'MEASURE',
    'RESET',
    'BARRIER',
    'EXPER',
    'IF',
    'EQ',
    'PI',
    'REALN',
    'INTN',
    'PLUS',
    'MINUS',
    'MUL',
    'DIV',
    'POW',
    'LBRACE',
    'RBRACE',
    'LPAREN',
    'RPAREN',
    'LBRACKET',
    'RBRACKET',
    'ARROW',
    'COMMA',
    'SEMICOLON',
    'ID',
)

# Reserved words
reserved = {
    'OPENQASM': 'OPENQASM',
    'qreg': 'QREG',
    'creg': 'CREG',
    'gate': 'GATE',
    'opaque': 'OPAQUE',
    'measure': 'MEASURE',
    'reset': 'RESET',
    'barrier': 'BARRIER',
    'experiment': 'EXPER',
    'if': 'IF',
    'pi': 'PI',
    '->': 'ARROW',
}


def t_REALN(t) -> float:  # pyright: ignore[reportMissingParameterType]
    r'\d+\.\d+'
    t.value = float(t.value)
    return t


def t_INTN(t) -> int:  # pyright: ignore[reportMissingParameterType]
    r'\d+'
    t.value = int(t.value)
    return t


def t_newline(t):  # pyright: ignore[reportMissingParameterType]
    r'\n+'
    t.lexer.lineno += len(t.value)


def t_ID(t):  # pyright: ignore[reportMissingParameterType]
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    return t


# Error handling rule
def t_error(t):  # pyright: ignore[reportMissingParameterType]
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)


# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'
# Regular expression rules for simple tokens
t_OPENQASM = r'OPENQASM'
t_PI = r'pi'
t_QREG = r'qreg'
t_CREG = r'creg'
t_GATE = r'gate'
t_OPAQUE = r'opaque'
t_MEASURE = r'measure'
t_RESET = r'reset'
t_BARRIER = r'barrier'
t_EXPER = r'experiment'
t_IF = r'if'
t_EQ = r'=='
t_PLUS = r'\+'
t_MUL = r'\*'
t_POW = r'\^'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_ARROW = r'->'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_SEMICOLON = r';'
t_MINUS = r'-'
t_DIV = r'/'

from mpqp.gates import *


single_qubits_gate_qasm = {
    "h": H,
    "x": X,
    "y": Y,
    "z": Z,
    "s": S,
    "id": Id,
    "t": T,
}

two_qubits_gate_qasm = {
    "cx": CNOT,
    "swap": SWAP,
    "cz": CZ,
}

one_parametrized_gate_qasm = {
    "p": P,
    "rx": Rx,
    "ry": Ry,
    "rz": Rz,
}

two_qubits_parametrized_gate_qasm = {
    "cp": CP,
}


u_gate_qasm = {
    "U": U,
    "u": U,
    "u1": U,
    "u2": U,
    "u3": U,
}

gate_qasm = {
    "ccx": TOF,
}

LexToken = Any


def check_num_expr(token: LexToken) -> bool:
    return (
        token != 'MINUS'
        and token != 'PLUS'
        and token != 'DIV'
        and token != 'MUL'
        and token != 'POW'
        and token != 'REALN'
        and token != 'INTN'
        and token != 'PI'
    )


def check_Id(tokens: list[LexToken], idx: int) -> bool:
    return (
        tokens[idx].type != 'ID'
        or tokens[idx + 1].type != 'LBRACKET'
        or tokens[idx + 2].type != 'INTN'
        or tokens[idx + 3].type != 'RBRACKET'
    )
