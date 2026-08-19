import pytest
from sympy import Symbol

from mpqp.core.circuit import CircuitBinding, QCircuit, BindingMode
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.core.languages import Language
from mpqp.execution.devices import AWSDevice
from mpqp.gates import *
from mpqp.core.instruction.measurement.pauli_string import pX, pZ

t = Symbol("t")
r = Symbol("r")
c1 = QCircuit([Ry(t, 0), Rx(r, 0)])
c2 = QCircuit([Rz(t, 0), Rz(r, 0)])
o = [
    ExpectationMeasure([Observable(pZ)], optimize_measurement=False),
    ExpectationMeasure([Observable(pX - pZ)], optimize_measurement=False),
]


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circuit, nbr_jobs",
    [
        (
            CircuitBinding(
                [
                    CircuitBinding(
                        c1, values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}]
                    )
                ],
                measurements=o,
                mode=BindingMode.ZIP,
            ),
            4,
        ),
        (
            CircuitBinding(
                c1,
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                measurements=o,
                mode=BindingMode.ZIP,
            ),
            2,
        ),
        (
            CircuitBinding(
                [c1, c2],
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                measurements=o,
                mode=BindingMode.PRODUCT,
            ),
            8,
        ),
        (
            CircuitBinding(
                [c1, c2],
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                mode=BindingMode.ZIP,
            ),
            2,
        ),
        (
            CircuitBinding(
                c1,
                measurements=o,
                mode=BindingMode.ZIP,
            ),
            2,
        ),
        (
            CircuitBinding(
                [
                    c2,
                    CircuitBinding(
                        c1, values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}]
                    ),
                ],
                measurements=o,
                mode=BindingMode.ZIP,
            ),
            3,
        ),
        (
            CircuitBinding(
                [
                    c2,
                    CircuitBinding(
                        c1,
                        values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                        mode=BindingMode.ZIP,
                    ),
                ],
                measurements=o,
                mode=BindingMode.PRODUCT,
            ),
            6,
        ),
        (
            CircuitBinding(
                [c2, CircuitBinding(c1)],
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                measurements=o,
                mode=BindingMode.PRODUCT,
            ),
            8,
        ),
    ],
)
def test_translation_nbr_jobs(circuit: CircuitBinding, nbr_jobs: int):
    ps, _ = circuit.to_other_device(AWSDevice.BRAKET_LOCAL_SIMULATOR)
    assert len(ps) == nbr_jobs


@pytest.mark.parametrize(
    "c, value",
    [
        (
            CircuitBinding(
                c1,
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                measurements=ExpectationMeasure(
                    [Observable(pX - pZ)], optimize_measurement=False
                ),
                mode=BindingMode.PRODUCT,
                shots=500,
            ),
            [0.3, -0.54],
        ),
        (
            CircuitBinding(
                [c1, c2],
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                measurements=[
                    ExpectationMeasure(Observable(pX + pZ), optimize_measurement=False),
                    ExpectationMeasure(Observable(pX), optimize_measurement=False),
                ],
                mode=BindingMode.ZIP,
                shots=500,
            ),
            [1.3, 0],
        ),
    ],
)
def test_run_multiple_monomials_obs(c: CircuitBinding, value: list[float]):
    from mpqp.execution.runner import run

    res = run(c, AWSDevice.BRAKET_LOCAL_SIMULATOR).results
    for i in range(len(res)):
        exp_value = res[i].expectation_values
        assert isinstance(exp_value, float)
        assert abs(exp_value - value[i]) <= 0.1
