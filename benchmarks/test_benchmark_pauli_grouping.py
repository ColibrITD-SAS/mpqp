"""Benchmarks for Pauli-string grouping."""

from typing import Any

import pytest

from mpqp.core.instruction.measurement.pauli_string import (
    CommutingTypes,
    PauliString,
    PauliStringMonomial,
)
from mpqp.tools.pauli_grouping import pauli_grouping_greedy


pytestmark = pytest.mark.performance


def _pauli_monomials(count: int, width: int) -> list[PauliStringMonomial]:
    labels = "IXYZ"
    monomials = []
    for value in range(1, count + 1):
        encoded = value
        pattern = []
        for _ in range(width):
            pattern.append(labels[encoded % len(labels)])
            encoded //= len(labels)
        monomial = PauliString.from_str("".join(pattern))
        assert isinstance(monomial, PauliStringMonomial)
        monomials.append(monomial)
    return monomials


@pytest.mark.parametrize(
    "commuting_type", [CommutingTypes.FULL, CommutingTypes.QUBITWISE]
)
def test_pauli_grouping(benchmark: Any, commuting_type: CommutingTypes) -> None:
    monomials = _pauli_monomials(128, 8)
    groups = benchmark(pauli_grouping_greedy, monomials, commuting_type)
    assert sum(map(len, groups)) == len(monomials)
