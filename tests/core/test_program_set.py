import pytest
from sympy import Symbol

from mpqp.core.circuit import QCircuit
from mpqp.core.circuitbinding import CircuitBinding, BindingMode
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
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
    "circuit, nbr_results, nbr_entries",
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
            2,
        ),
        (
            CircuitBinding(
                [CircuitBinding(c1, measurements=o)],
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                mode=BindingMode.ZIP,
            ),
            4,
            2,
        ),
        (
            CircuitBinding(
                c1,
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                measurements=o,
                mode=BindingMode.ZIP,
            ),
            2,
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
            4,
        ),
        (
            CircuitBinding(
                [c1, c2],
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                mode=BindingMode.ZIP,
            ),
            2,
            2,
        ),
        (
            CircuitBinding(
                c1,
                measurements=o,
                mode=BindingMode.ZIP,
            ),
            2,
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
            2,
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
            4,
        ),
        (
            CircuitBinding(
                [c2, CircuitBinding(c1)],
                values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
                measurements=o,
                mode=BindingMode.PRODUCT,
            ),
            8,
            4,
        ),
    ],
)
def test_translation_nbr_jobs(
    circuit: CircuitBinding, nbr_results: int, nbr_entries: int
):
    ps, contexts = circuit.to_other_device(AWSDevice.BRAKET_LOCAL_SIMULATOR)
    assert len(ps) == nbr_entries
    assert sum(len(entry_contexts) for entry_contexts in contexts) == nbr_results
    assert ps.total_executables == sum(
        context[3] for entry_contexts in contexts for context in entry_contexts
    )


@pytest.mark.provider("braket")
def test_groups_parameter_sets_for_one_circuit():
    binding = CircuitBinding(
        c1,
        values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
    )

    program_set, contexts = binding.to_other_device(AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert len(program_set) == 1
    assert program_set.total_executables == 2
    assert len(program_set[0].input_sets) == 2
    assert len(contexts) == 1
    assert len(contexts[0]) == 2


@pytest.mark.provider("braket")
def test_groups_simple_observables_and_parameter_sets_for_one_circuit():
    measurements = [
        ExpectationMeasure(Observable(pX), optimize_measurement=False),
        ExpectationMeasure(Observable(pZ), optimize_measurement=False),
    ]
    binding = CircuitBinding(
        c1,
        values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
        measurements=measurements,
        mode=BindingMode.PRODUCT,
    )

    program_set, contexts = binding.to_other_device(AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert len(program_set) == 1
    assert program_set.total_executables == 4
    assert len(program_set[0].input_sets) == 2
    assert len(program_set[0].observables) == 2
    assert len(contexts[0]) == 4


@pytest.mark.provider("braket")
def test_does_not_cross_product_zipped_parameters_and_observables():
    binding = CircuitBinding(
        c1,
        values=[{"t": 1.0, "r": 0.0}, {"t": 0.0, "r": 1.0}],
        measurements=[
            ExpectationMeasure(Observable(pX), optimize_measurement=False),
            ExpectationMeasure(Observable(pZ), optimize_measurement=False),
        ],
        mode=BindingMode.ZIP,
    )

    program_set, contexts = binding.to_other_device(AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert len(program_set) == 2
    assert program_set.total_executables == 2
    assert [len(entry_contexts) for entry_contexts in contexts] == [1, 1]


@pytest.mark.provider("braket")
def test_multi_observable_measure_returns_one_result():
    from mpqp.execution.runner import run

    binding = CircuitBinding(
        c1,
        values={"t": 0.0, "r": 0.0},
        measurements=ExpectationMeasure(
            [
                Observable(pZ, label="Z"),
                Observable(pX - pZ, label="X-Z"),
            ],
            shots=1000,
            optimize_measurement=False,
        ),
    )

    program_set, contexts = binding.to_other_device(
        AWSDevice.BRAKET_LOCAL_SIMULATOR
    )
    result = run(binding, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert len(program_set) == 2
    assert {context[4] for entry_contexts in contexts for context in entry_contexts} == {
        0
    }
    assert len(result.results) == 1
    assert result.results[0].expectation_values == pytest.approx(
        {"Z": 1.0, "X-Z": -1.0}, abs=0.15
    )


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
        assert abs(exp_value - value[i]) <= 0.2
