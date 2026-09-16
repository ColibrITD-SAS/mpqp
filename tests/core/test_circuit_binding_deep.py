from __future__ import annotations

import pytest

from mpqp import (
    AWSDevice,
    BasisMeasure,
    CNOT,
    ExpectationMeasure,
    H,
    IBMDevice,
    Observable,
    QCircuit,
    pX,
    pZ,
    run,
)
from mpqp.core.circuit import BindingMode, CircuitBinding


def _circuits() -> tuple[QCircuit, QCircuit]:
    c3 = QCircuit([H(0), CNOT(0, 1)], label="c3")
    c4 = QCircuit([H(0), H(1), CNOT(0, 1)], label="c4")
    return c3, c4


def _values() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    return (
        {"θ": 1.0, "phi": 1.0, "psi": 1.0},
        {"θ": 2.0, "phi": 2.0, "psi": 2.0},
        {"θ": 3.0, "phi": 3.0, "psi": 3.0},
    )


def _measurements() -> tuple[
    ExpectationMeasure,
    BasisMeasure,
    BasisMeasure,
    BasisMeasure,
    BasisMeasure,
]:
    return (
        ExpectationMeasure(Observable(pX @ pZ), label="Exp2", shots=2024),
        BasisMeasure(label="b3", shots=2024),
        BasisMeasure(targets=[0], label="b3_0", shots=2024),
        BasisMeasure(targets=[1], label="b3_1", shots=2024),
        BasisMeasure(targets=[0, 1], label="b3_01", shots=2024),
    )


def _case_1() -> dict[str, CircuitBinding]:
    c3, c4 = _circuits()
    v1, v2, _ = _values()
    return {
        "c_1_0": CircuitBinding([c3, c4]),
        "c_1_1": CircuitBinding(CircuitBinding([c3, c4])),
        "c_1_2": CircuitBinding([CircuitBinding(c3), c4]),
        "c_1_3": CircuitBinding([CircuitBinding(c3), CircuitBinding(c4)]),
        "c_1_4": CircuitBinding([c3, c4], values=[v1]),
        "c_1_5": CircuitBinding(CircuitBinding([c3, c4]), values=[v1]),
        "c_1_6": CircuitBinding([CircuitBinding(c3), c4], values=[v1]),
        "c_1_7": CircuitBinding([c3, CircuitBinding(c4)], values=[v1]),
        "c_1_8": CircuitBinding([c3, c4], values=[v1, v2], mode=BindingMode.ZIP),
        "c_1_9": CircuitBinding(
            [CircuitBinding(c3), CircuitBinding(c4)],
            values=[v1, v2],
            mode=BindingMode.ZIP,
        ),
        "c_1_10": CircuitBinding(
            [CircuitBinding(c3), c4],
            values=[v1, v2],
            mode=BindingMode.ZIP,
        ),
        "c_1_11": CircuitBinding(
            CircuitBinding([c3, c4]), values=v1, mode=BindingMode.ZIP
        ),
        "c_1_12": CircuitBinding(CircuitBinding([c3], values=v1, mode=BindingMode.ZIP)),
    }


def _embedded_cases(
    measurement: BasisMeasure | ExpectationMeasure,
) -> dict[str, CircuitBinding]:
    c3, c4 = _circuits()
    v1, v2, _ = _values()
    c3_ = c3 + QCircuit([measurement])
    c4_ = c4 + QCircuit([measurement])
    return {
        "0": CircuitBinding([c3_, c4_]),
        "1": CircuitBinding(CircuitBinding([c3_, c4_])),
        "2": CircuitBinding([CircuitBinding(c3_), c4_]),
        "3": CircuitBinding([CircuitBinding(c3_), CircuitBinding(c4_)]),
        "4": CircuitBinding([c3_, c4_], values=[v1]),
        "5": CircuitBinding(CircuitBinding([c3_, c4_]), values=[v1]),
        "6": CircuitBinding([CircuitBinding(c3_), c4_], values=[v1]),
        "7": CircuitBinding([c3_, CircuitBinding(c4_)], values=[v1]),
        "8": CircuitBinding([c3_, c4_], values=[v1, v2], mode=BindingMode.ZIP),
        "9": CircuitBinding(
            [CircuitBinding(c3_), CircuitBinding(c4_)],
            values=[v1, v2],
            mode=BindingMode.ZIP,
        ),
        "10": CircuitBinding(
            [CircuitBinding(c3_), c4_],
            values=[v1, v2],
            mode=BindingMode.ZIP,
        ),
        "11": CircuitBinding(
            CircuitBinding([c3_, c4_]), values=v1, mode=BindingMode.ZIP
        ),
        "12": CircuitBinding(CircuitBinding([c3_], values=v1, mode=BindingMode.ZIP)),
    }


def _case_2() -> dict[str, CircuitBinding]:
    _, m3, _, _, _ = _measurements()
    return {f"c_2_{suffix}": binding for suffix, binding in _embedded_cases(m3).items()}


def _case_3() -> dict[str, CircuitBinding]:
    m2, _, _, _, _ = _measurements()
    return {f"c_3_{suffix}": binding for suffix, binding in _embedded_cases(m2).items()}


def _case_4() -> dict[str, CircuitBinding]:
    c3, c4 = _circuits()
    m2, m3, _, _, _ = _measurements()
    return {
        "c_4_0": CircuitBinding([c3, c4], measurements=[m2]),
        "c_4_1": CircuitBinding([CircuitBinding(c3), c4], measurements=[m2]),
        "c_4_2": CircuitBinding(
            [CircuitBinding(c3), CircuitBinding(c4)], measurements=[m2]
        ),
        "c_4_3": CircuitBinding(CircuitBinding([c3, c4], measurements=[m2])),
        "c_4_4": CircuitBinding(CircuitBinding([c3, c4]), measurements=[m2]),
        "c_4_5": CircuitBinding([c3, c4], measurements=[m3]),
        "c_4_6": CircuitBinding([CircuitBinding(c3), c4], measurements=[m3]),
        "c_4_7": CircuitBinding(
            [CircuitBinding(c3), CircuitBinding(c4)], measurements=[m3]
        ),
        "c_4_8": CircuitBinding(CircuitBinding([c3, c4], measurements=[m3])),
        "c_4_9": CircuitBinding(CircuitBinding([c3, c4]), measurements=[m3]),
    }


def _case_5() -> dict[str, CircuitBinding]:
    c3, c4 = _circuits()
    v1, v2, _ = _values()
    _, m3, m3_0, m3_1, _ = _measurements()
    c3_ = c3 + QCircuit([m3])
    c4_ = c4 + QCircuit([m3])
    return {
        "c_5_0": CircuitBinding([CircuitBinding(c3_), c4_], values=[v1, v2]),
        "c_5_1": CircuitBinding([c4_, CircuitBinding(c3_)], values=[v1, v2]),
        "c_5_2": CircuitBinding(
            [CircuitBinding(c3_), c4_],
            values=[v1, v2],
            mode=BindingMode.ZIP,
        ),
        "c_5_3": CircuitBinding(
            [
                c4_,
                CircuitBinding(
                    c3,
                    measurements=[m3_1, m3_0],
                    mode=BindingMode.PRODUCT,
                ),
            ],
            values=[v1, v2],
            mode=BindingMode.ZIP,
        ),
    }


def _case_6() -> dict[str, CircuitBinding]:
    c3, c4 = _circuits()
    v1, v2, v3 = _values()
    _, _, m3_0, m3_1, m3_01 = _measurements()
    return {
        "c_6_0": CircuitBinding(
            [
                c3,
                CircuitBinding(c4, values=[v1, v2], measurements=[m3_01]),
            ],
            measurements=[m3_0, m3_1],
        ),
        "c_6_1": CircuitBinding(
            [
                c3,
                CircuitBinding(
                    c4,
                    values=[v3],
                    measurements=[m3_01],
                    mode=BindingMode.ZIP,
                ),
            ],
            values=[v1, v2],
            measurements=[m3_0, m3_1],
        ),
        "c_6_2": CircuitBinding(
            [
                c3,
                CircuitBinding(
                    c4,
                    values=[v1, v2],
                    measurements=[m3_0, m3_1],
                    mode=BindingMode.ZIP,
                ),
            ],
            measurements=[m3_01],
        ),
    }


def _all_cases() -> dict[str, CircuitBinding]:
    return _case_1() | _case_2() | _case_3() | _case_4() | _case_5() | _case_6()


EXPECTED_EXECUTIONS = {
    **{f"c_1_{index}": 1 if index == 12 else 2 for index in range(13)},
    **{f"c_2_{index}": 1 if index == 12 else 2 for index in range(13)},
    **{f"c_3_{index}": 1 if index == 12 else 2 for index in range(13)},
    **{f"c_4_{index}": 2 for index in range(10)},
    "c_5_0": 4,
    "c_5_1": 4,
    "c_5_2": 2,
    "c_5_3": 3,
    "c_6_0": 8,
    "c_6_1": 13,
    "c_6_2": 5,
}


def _binding(name: str) -> CircuitBinding:
    return _all_cases()[name]


@pytest.mark.parametrize(
    "name,expected_executions",
    EXPECTED_EXECUTIONS.items(),
    ids=EXPECTED_EXECUTIONS,
)
def test_deep_binding_unrolls(name: str, expected_executions: int):
    binding = _binding(name)
    executions = binding.unroll()

    assert len(executions) == expected_executions
    assert all(circuit is not None for circuit, _, _ in executions)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "name,expected_executions",
    EXPECTED_EXECUTIONS.items(),
    ids=EXPECTED_EXECUTIONS,
)
def test_qiskit_deep_binding(name: str, expected_executions: int):
    binding = _binding(name)
    result = run(binding, IBMDevice.AER_SIMULATOR)

    actual_executions = len(result.results)
    assert actual_executions == expected_executions, name


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "name,expected_executions",
    [item for item in EXPECTED_EXECUTIONS.items() if not item[0].startswith("c_1_")],
    ids=[name for name in EXPECTED_EXECUTIONS if not name.startswith("c_1_")],
)
def test_braket_deep_binding(name: str, expected_executions: int):
    binding = _binding(name)
    result = run(binding, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    actual_executions = len(result.results)
    assert actual_executions == expected_executions, name
