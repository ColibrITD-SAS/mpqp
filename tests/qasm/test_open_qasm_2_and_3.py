from __future__ import annotations

import pytest

from mpqp.qasm.open_qasm_2_and_3 import open_qasm_file_conversion_2_to_3, open_qasm_hard_includes, parse_user_gates, remove_user_gates
import re

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
            include "qelib1.inc";
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
            include "qelib1.inc";
            qreg q[3];
            creg c[2];
            rzz(0.2) q[1], q[2];
            measure q[2] -> c[0];"""
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";
            gate my_gate a,b {
                h a;
                cx a,b;
            }
            qreg q[2];
            creg c[2];
            my_gate q[0], q[1];
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
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            my_gate q[0], q[1];
            measure q -> c;"""
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            cx q[0], q[1];
            cx q[1], q[2];""",
            [],
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            cx q[0], q[1];
            cx q[1], q[2];"""
        ),
    ]
)
def test_parse_user_gates(qasm_code: str, expected_gates: list[dict[str, str | list[str]]], expected_stripped_code: str):
    user_gates, stripped_code = parse_user_gates(qasm_code)
    assert len(user_gates) == len(expected_gates)
    for gate, expected_gate in zip(user_gates, expected_gates):
        assert gate.name == expected_gate["name"]
        assert gate.parameters == expected_gate["parameters"]
        assert gate.qubits == expected_gate["qubits"]
        assert gate.instructions == expected_gate["instructions"]
    assert normalize_whitespace(stripped_code) == normalize_whitespace(expected_stripped_code)


@pytest.mark.parametrize(
    "qasm_code, expected_output",
    [
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
            rzz(0.2) q[1], q[2];
            measure q[2] -> c[0];""",
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[2];
            cx q[1], q[2];
            u1(0.2) q[2];
            cx q[1], q[2];
            measure q[2] -> c[0];"""
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";
            gate my_gate a,b {
                h a;
                cx a,b;
            }
            qreg q[2];
            creg c[2];
            my_gate q[0], q[1];
            measure q -> c;""",
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;"""
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            cx q[0], q[1];
            cx q[1], q[2];""",
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            cx q[0], q[1];
            cx q[1], q[2];"""
        ),
    ]
)
def test_remove_user_gates(qasm_code: str, expected_output: str):
    output = remove_user_gates(qasm_code)
    assert normalize_whitespace(output) == normalize_whitespace(expected_output)