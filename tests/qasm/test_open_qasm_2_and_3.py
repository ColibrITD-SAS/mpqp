from __future__ import annotations

import re

import pytest

from mpqp.qasm.open_qasm_2_and_3 import (
    open_qasm_file_conversion_2_to_3,
    open_qasm_hard_includes,
    parse_user_gates,
    remove_user_gates,
)

qasm_folder = "tests/qasm/qasm_examples/"


def test_qasm_hard_import():
    with_include_fn = qasm_folder + "with_include.qasm"
    with open(with_include_fn) as f1:
        with open(qasm_folder + "without_include.qasm") as f2:
            assert open_qasm_hard_includes(f1.read(), {with_include_fn}) == f2.read()


def test_late_gate_def():
    with pytest.raises(ValueError):
        open_qasm_file_conversion_2_to_3(qasm_folder + "late_gate_def.qasm")


def test_in_time_gate_def():
    file_name = qasm_folder + "in_time_gate_def.qasm"
    converted_file_name = qasm_folder + "in_time_gate_def_converted.qasm"
    with open(converted_file_name, "r") as f:
        assert open_qasm_file_conversion_2_to_3(file_name) == f.read()


def test_circular_dependency_detection():
    with pytest.raises(RuntimeError) as e:
        open_qasm_file_conversion_2_to_3(
            qasm_folder + "circular_dep1.qasm",
        )
    assert "Circular dependency" in str(e.value)


def test_circular_dependency_detection_false_positive():
    try:
        open_qasm_file_conversion_2_to_3(
            qasm_folder + "circular_dep_a.qasm",
        )
    except RuntimeError:
        assert False, f"Circular dependency raised while it shouldn't"


def normalize_whitespace(s: str) -> str:
    """Helper function to normalize the whitespace by collapsing multiple spaces and removing leading/trailing spaces."""
    return re.sub(r'\s+', ' ', s.strip())


@pytest.mark.parametrize(
    "qasm_code, expected_gates, expected_stripped_code",
    [
        (
            """OPENQASM 2.0;
            gate rzz(theta) a,b {
                cx a,b;
                u1(theta) b;
                cx a,b;
            }
            qreg q[3];
            creg c[2];
            rzz(0.2) q[1], q[2];
            measure q[2] -> c[0];""",
            [
                {
                    "name": "rzz",
                    "parameters": ["theta"],
                    "qubits": ["a", "b"],
                    "instructions": ["cx a,b;", "u1(theta) b;", "cx a,b;"],
                }
            ],
            """OPENQASM 2.0;
            qreg q[3];
            creg c[2];
            rzz(0.2) q[1], q[2];
            measure q[2] -> c[0];""",
        ),
        (
            """OPENQASM 2.0;
            gate my_gate a,b {
                h a;
                cx a,b;
            }
            qreg q[2];
            creg c[2];
            my_gate q[0],q[1];
            measure q -> c;""",
            [
                {
                    "name": "my_gate",
                    "parameters": [],
                    "qubits": ["a", "b"],
                    "instructions": ["h a;", "cx a,b;"],
                }
            ],
            """OPENQASM 2.0;
            qreg q[2];
            creg c[2];
            my_gate q[0],q[1];
            measure q -> c;""",
        ),
        (
            """OPENQASM 2.0;
            qreg q[3];
            cx q[0],q[1];
            cx q[1],q[2];""",
            [],
            """OPENQASM 2.0;
            qreg q[3];
            cx q[0],q[1];
            cx q[1],q[2];""",
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";""",
            [
                {
                    'name': 'u3',
                    'parameters': ['theta', 'phi', 'lambda'],
                    'qubits': ['q'],
                    'instructions': ['U(theta,phi,lambda) q;'],
                },
                {
                    'name': 'u2',
                    'parameters': ['phi', 'lambda'],
                    'qubits': ['q'],
                    'instructions': ['U(pi/2,phi,lambda) q;'],
                },
                {
                    'name': 'u1',
                    'parameters': ['lambda'],
                    'qubits': ['q'],
                    'instructions': ['U(0,0,lambda) q;'],
                },
                {
                    'name': 'u0',
                    'parameters': ['gamma'],
                    'qubits': ['q'],
                    'instructions': ['U(0,0,0) q;'],
                },
                {
                    'name': 'cx',
                    'parameters': [],
                    'qubits': ['c', 't'],
                    'instructions': ['CX c,t;'],
                },
                {
                    'name': 'id',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['U(0,0,0) a;'],
                },
                {
                    'name': 'u',
                    'parameters': ['theta', 'phi', 'lambda'],
                    'qubits': ['q'],
                    'instructions': ['U(theta,phi,lambda) q;'],
                },
                {
                    'name': 'p',
                    'parameters': ['lambda'],
                    'qubits': ['q'],
                    'instructions': ['U(0,0,lambda) q;'],
                },
                {
                    'name': 'x',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['u3(pi,0,pi) a;'],
                },
                {
                    'name': 'y',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['u3(pi,pi/2,pi/2) a;'],
                },
                {
                    'name': 'z',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['u1(pi) a;'],
                },
                {
                    'name': 'h',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['u2(0,pi) a;'],
                },
                {
                    'name': 's',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['u1(pi/2) a;'],
                },
                {
                    'name': 'sdg',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['u1(-pi/2) a;'],
                },
                {
                    'name': 't',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['u1(pi/4) a;'],
                },
                {
                    'name': 'tdg',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['u1(-pi/4) a;'],
                },
                {
                    'name': 'sx',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['sdg a;', 'h a;', 'sdg a;'],
                },
                {
                    'name': 'sxdg',
                    'parameters': [],
                    'qubits': ['a'],
                    'instructions': ['s a;', 'h a;', 's a;'],
                },
                {
                    'name': 'rx',
                    'parameters': ['theta'],
                    'qubits': ['a'],
                    'instructions': ['u3(theta,-pi/2,pi/2) a;'],
                },
                {
                    'name': 'ry',
                    'parameters': ['theta'],
                    'qubits': ['a'],
                    'instructions': ['u3(theta,0,0) a;'],
                },
                {
                    'name': 'rz',
                    'parameters': ['phi'],
                    'qubits': ['a'],
                    'instructions': ['u1(phi) a;'],
                },
                {
                    'name': 'swap',
                    'parameters': [],
                    'qubits': ['a', 'b'],
                    'instructions': ['cx a,b;', 'cx b,a;', 'cx a,b;'],
                },
                {
                    'name': 'cz',
                    'parameters': [],
                    'qubits': ['a', 'b'],
                    'instructions': ['h b;', 'cx a,b;', 'h b;'],
                },
                {
                    'name': 'cy',
                    'parameters': [],
                    'qubits': ['a', 'b'],
                    'instructions': ['sdg b;', 'cx a,b;', 's b;'],
                },
                {
                    'name': 'ch',
                    'parameters': [],
                    'qubits': ['a', 'b'],
                    'instructions': [
                        'h b;',
                        'sdg b;',
                        'cx a,b;',
                        'h b;',
                        't b;',
                        'cx a,b;',
                        't b;',
                        'h b;',
                        's b;',
                        'x b;',
                        's a;',
                    ],
                },
                {
                    'name': 'ccx',
                    'parameters': [],
                    'qubits': ['a', 'b', 'c'],
                    'instructions': [
                        'h c;',
                        'cx b,c;',
                        'tdg c;',
                        'cx a,c;',
                        't c;',
                        'cx b,c;',
                        'tdg c;',
                        'cx a,c;',
                        't b;',
                        't c;',
                        'h c;',
                        'cx a,b;',
                        't a;',
                        'tdg b;',
                        'cx a,b;',
                    ],
                },
                {
                    'name': 'crz',
                    'parameters': ['lambda'],
                    'qubits': ['a', 'b'],
                    'instructions': [
                        'u1(lambda/2) b;',
                        'cx a,b;',
                        'u1(-lambda/2) b;',
                        'cx a,b;',
                    ],
                },
                {
                    'name': 'cu1',
                    'parameters': ['lambda'],
                    'qubits': ['a', 'b'],
                    'instructions': [
                        'u1(lambda/2) a;',
                        'cx a,b;',
                        'u1(-lambda/2) b;',
                        'cx a,b;',
                        'u1(lambda/2) b;',
                    ],
                },
                {
                    'name': 'cu3',
                    'parameters': ['theta', 'phi', 'lambda'],
                    'qubits': ['c', 't'],
                    'instructions': [
                        '// implements controlled-U(theta,phi,lambda) with  target t and control c\n    u1((lambda-phi)/2) t;',
                        'cx c,t;',
                        'u3(-theta/2,0,-(phi+lambda)/2) t;',
                        'cx c,t;',
                        'u3(theta/2,phi,0) t;',
                    ],
                },
                {
                    'name': 'cswap',
                    'parameters': [],
                    'qubits': ['a', 'b', 'c'],
                    'instructions': ['cx c,b;', 'ccx a,b,c;', 'cx c,b;'],
                },
                {
                    'name': 'crx',
                    'parameters': ['lambda'],
                    'qubits': ['a', 'b'],
                    'instructions': [
                        'u1(pi/2) b;',
                        'cx a,b;',
                        'u3(-lambda/2,0,0) b;',
                        'cx a,b;',
                        'u3(lambda/2,-pi/2,0) b;',
                    ],
                },
                {
                    'name': 'cry',
                    'parameters': ['lambda'],
                    'qubits': ['a', 'b'],
                    'instructions': [
                        'ry(lambda/2) b;',
                        'cx a,b;',
                        'ry(-lambda/2) b;',
                        'cx a,b;',
                    ],
                },
                {
                    'name': 'cp',
                    'parameters': ['lambda'],
                    'qubits': ['a', 'b'],
                    'instructions': [
                        'p(lambda/2) a;',
                        'cx a,b;',
                        'p(-lambda/2) b;',
                        'cx a,b;',
                        'p(lambda/2) b;',
                    ],
                },
                {
                    'name': 'cu',
                    'parameters': ['theta', 'phi', 'lambda', 'gamma'],
                    'qubits': ['c', 't'],
                    'instructions': [
                        'p(gamma) c;',
                        'p((lambda+phi)/2) c;',
                        'p((lambda-phi)/2) t;',
                        'cx c,t;',
                        'u(-theta/2,0,-(phi+lambda)/2) t;',
                        'cx c,t;',
                        'u(theta/2,phi,0) t;',
                    ],
                },
            ],
            """OPENQASM 2.0;
            include "qelib1.inc";""",
        ),
    ],
)
def test_parse_user_gates(
    qasm_code: str,
    expected_gates: list[dict[str, str | list[str]]],
    expected_stripped_code: str,
):
    user_gates, stripped_code = parse_user_gates(qasm_code)
    assert len(user_gates) == len(expected_gates)
    for gate, expected_gate in zip(user_gates, expected_gates):
        assert gate.dict() == expected_gate
    assert normalize_whitespace(stripped_code) == normalize_whitespace(
        expected_stripped_code
    )


@pytest.mark.parametrize(
    "qasm_code, expected_output",
    [
        (
            """OPENQASM 2.0;
            gate rzz(theta) a,b {
                cx a,b;
                u1(theta) b;
                cx a,b;
            }
            qreg q[3];
            creg c[2];
            rzz(0.2) q[1], q[2];
            measure q[2] -> c[0];""",
            """OPENQASM 2.0;
            qreg q[3];
            creg c[2];
            cx q[1],q[2];
            u1(0.2) q[2];
            cx q[1],q[2];
            measure q[2] -> c[0];""",
        ),
        (
            """OPENQASM 2.0;
            gate my_gate a,b {
                h a;
                cx a,b;
            }
            qreg q[2];
            creg c[2];
            my_gate q[0], q[1];
            measure q -> c;""",
            """OPENQASM 2.0;
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];
            measure q -> c;""",
        ),
        (
            """OPENQASM 2.0;
            qreg q[3];
            cx q[0],q[1];
            cx q[1],q[2];""",
            """OPENQASM 2.0;
            qreg q[3];
            cx q[0],q[1];
            cx q[1],q[2];""",
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[2];
            u1(0.2) q[1], q[2];
            measure q[2] -> c[0];""",
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[2];
            U(0,0,0.2) q[1];
            measure q[2] -> c[0];""",
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";
            gate rzz(theta) a,b {
                cx a,b;
                u1(theta) b;
                cx a,b;
            }
            qreg q[3];
            creg c[2];
            rzz(0.2) q[1] , q[2];
            measure q[2] ->  c[0];""",
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[2];
            CX q[1],q[2];
            U(0,0,0.2) q[2];
            CX q[1],q[2];
            measure q[2] -> c[0];""",
        ),
    ],
)
def test_remove_user_gates(qasm_code: str, expected_output: str):
    output = remove_user_gates(qasm_code)
    assert normalize_whitespace(output) == normalize_whitespace(expected_output)
