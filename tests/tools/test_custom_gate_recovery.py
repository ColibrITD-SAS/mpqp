"""Regression coverage for custom-unitary decomposition recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

from mpqp.tools.circuit import replace_custom_gate

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

pytestmark = pytest.mark.provider("qiskit")

# PyO3 creates this exception dynamically; it is not a public importable class.
PanicException = type(
    "PanicException", (BaseException,), {"__module__": "pyo3_runtime"}
)


def _embedded_unitary(targets: list[int]) -> QuantumCircuit:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(4, global_phase=0.37)
    matrix = (
        np.exp(0.23j)
        * np.array(
            [[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]],
            dtype=np.complex128,
        )
        / 2
    )
    circuit.unitary(matrix, list(reversed(targets)))
    return circuit


def _operator_matrix(circuit: QuantumCircuit) -> npt.NDArray[np.complex128]:
    from qiskit.quantum_info import Operator

    matrix = Operator(circuit).data
    assert isinstance(matrix, np.ndarray)
    return np.asarray(matrix, dtype=np.complex128)


@pytest.mark.parametrize("targets", [[0, 2], [3, 1]])
@pytest.mark.parametrize("as_instruction", [False, True])
@pytest.mark.parametrize("rust_panic", [False, True])
def test_replace_custom_gate_recovers_known_weyl_error(
    monkeypatch: pytest.MonkeyPatch,
    targets: list[int],
    as_instruction: bool,
    rust_panic: bool,
):
    import qiskit
    from qiskit.exceptions import QiskitError

    original = _embedded_unitary(targets)
    expected = original.copy()
    if as_instruction:
        expected.global_phase = 0
    before = _operator_matrix(original).copy()
    error_type = PanicException if rust_panic else QiskitError
    original_transpile = qiskit.transpile
    calls = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_type("TwoQubitWeylDecomposition: failed to diagonalize M2")
        return original_transpile(*args, **kwargs)

    monkeypatch.setattr(qiskit, "transpile", fail_once)
    source = original.data[0] if as_instruction else original
    recovered, phase = replace_custom_gate(source, 4, targets)

    assert calls == 2
    np.testing.assert_allclose(
        _operator_matrix(recovered), _operator_matrix(expected), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(_operator_matrix(original), before, atol=0, rtol=0)
    assert float(phase) == float(recovered.global_phase)
    assert set(recovered.count_ops()) <= {"u1", "u2", "u3", "cx"}


@pytest.mark.parametrize(
    "error",
    [
        ValueError("TwoQubitWeylDecomposition"),
        PanicException("unrelated Rust failure"),
        KeyboardInterrupt("TwoQubitWeylDecomposition"),
        SystemExit("TwoQubitWeylDecomposition"),
        type("PanicException", (BaseException,), {"__module__": "other_library"})(
            "TwoQubitWeylDecomposition"
        ),
    ],
)
def test_replace_custom_gate_propagates_unrelated_errors(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
):
    import qiskit

    calls = 0

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(qiskit, "transpile", fail)
    with pytest.raises(type(error)) as raised:
        replace_custom_gate(_embedded_unitary([0, 2]), 4, [0, 2])
    assert raised.value is error
    assert calls == 1


def test_replace_custom_gate_does_not_retry_failed_recovery(
    monkeypatch: pytest.MonkeyPatch,
):
    import qiskit
    from qiskit.exceptions import QiskitError

    calls = 0
    retry_error = QiskitError("recovery failed")

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PanicException("TwoQubitWeylDecomposition")
        raise retry_error

    monkeypatch.setattr(qiskit, "transpile", fail)
    with pytest.raises(QiskitError) as raised:
        replace_custom_gate(_embedded_unitary([0, 2]), 4, [0, 2])
    assert raised.value is retry_error
    assert calls == 2


def test_replace_custom_gate_ci_weyl_matrix():
    from qiskit import QuantumCircuit

    # Intermediate matrix from PR #230's Python 3.11 job on October 22, 2026:
    # https://github.com/ColibrITD-SAS/mpqp/actions/runs/39758401989/job/119756162319
    matrix = np.array(
        [
            [
                complex(0.00020961568178717548, -0.0002995256341753557),
                complex(-0.613913323995569, 0.3511332513624105),
                complex(0.36258201429860576, 0.6069176848634146),
                complex(0.0007927362280353931, -0.0005612749737101864),
            ],
            [
                complex(0.00017058027647478708, -0.00024374686595805012),
                complex(-0.12848444315750165, 0.6952031352376132),
                complex(-0.6977714707237651, -0.11531604827411081),
                complex(0.00006346587995391695, 0.0011998914814278705),
            ],
            [
                complex(-0.033288919534519554, 0.9774821523802169),
                complex(-0.00004000579537747281, 0.0003745794611131671),
                complex(0.00028746242427017145, -0.000006265205628390314),
                complex(0.1539652432522095, 0.14041000285065022),
            ],
            [
                complex(-0.193229619174431, 0.07799090795448911),
                complex(-0.00025050143509294863, 0.0001198727232056108),
                complex(-0.0013667899433655575, -0.0006630030097078943),
                complex(0.3092580386739108, -0.9278668079296336),
            ],
        ],
        dtype=np.complex128,
    )
    circuit = QuantumCircuit(2)
    circuit.unitary(matrix, [1, 0])
    expected = _operator_matrix(circuit).copy()
    recovered, phase = replace_custom_gate(circuit, 2, [0, 1])
    # Compare with the original input, independently of the projection helper.
    np.testing.assert_allclose(
        _operator_matrix(recovered), expected, atol=1e-10, rtol=0
    )
    assert float(phase) == float(recovered.global_phase)
