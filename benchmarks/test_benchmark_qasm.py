"""Benchmarks for the QASM import/export paths."""

from typing import Any

import pytest

from mpqp import QCircuit
from mpqp.translation.qasm.mpqp_to_qasm import mpqp_to_qasm2
from mpqp.translation.qasm.qasm_to_mpqp import qasm2_parse


pytestmark = pytest.mark.performance


def test_qasm2_export(benchmark: Any, medium_circuit: QCircuit) -> None:
    qasm, global_phase = benchmark(mpqp_to_qasm2, medium_circuit)
    assert qasm.startswith("OPENQASM 2.0;")
    assert global_phase == 0


def test_qasm2_parse(benchmark: Any, medium_qasm: str) -> None:
    circuit = benchmark(qasm2_parse, medium_qasm)
    assert len(circuit) == 1050
