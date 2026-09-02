"""Benchmarks for core circuit operations."""

from collections.abc import Callable
from typing import Any

import pytest

from mpqp import QCircuit

pytestmark = pytest.mark.performance


def test_circuit_construction(
    benchmark: Any, circuit_factory: Callable[[int, int], QCircuit]
) -> None:
    circuit = benchmark(circuit_factory, 12, 30)
    assert len(circuit) == 1410


def test_circuit_depth(benchmark: Any, medium_circuit: QCircuit) -> None:
    depth = benchmark(medium_circuit.depth)
    assert depth > 0


def test_circuit_to_matrix(benchmark: Any, matrix_circuit: QCircuit) -> None:
    matrix = benchmark(matrix_circuit.to_matrix)
    assert matrix.shape == (128, 128)
