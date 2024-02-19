import pytest
from mpqp.qasm import open_qasm_file_conversion_2_to_3, open_qasm_hard_includes

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


def test_circular_dependency_detection_2():
    try:
        open_qasm_file_conversion_2_to_3(
            qasm_folder + "circular_dep_a.qasm",
        )
    except RuntimeError:
        assert False, f"Circular dependency raised while it shouldn't"
