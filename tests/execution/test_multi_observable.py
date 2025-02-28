import pytest

from mpqp import QCircuit
from mpqp.core.instruction import Observable, ExpectationMeasure
from mpqp.execution import AvailableDevice, run



def test_sequential_versus_multi(circuit: QCircuit, observables: list[Observable], device: AvailableDevice):
    seq_results = [run(circuit + QCircuit([ExpectationMeasure(obs, shots=0)]), device) for obs in observables]

    multi_result = run(circuit + QCircuit([ExpectationMeasure(observables, shots=0)]))

    for r1, r2 in zip(seq_results, multi_result.results):
        assert r1.expectation_value == r2.expectation_value
