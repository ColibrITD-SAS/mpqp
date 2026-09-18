from __future__ import annotations
from typing import Any, Callable


import numpy as np
import pytest

from mpqp.core.circuitbinding import BindingMode, CircuitBinding
from mpqp.execution.devices import (
    AvailableDevice,
    IBMDevice,
    AWSDevice,
)

from mpqp import (
    CNOT,
    BasisMeasure,
    ExpectationMeasure,
    H,
    IBMDevice,
    Language,
    Observable,
    QCircuit,
    U,
    pI,
    pZ,
    pX,
    run,
)
from mpqp.execution.result import BatchResult
from sympy import Symbol

theta, phi, psi = Symbol('θ'), Symbol('phi'), Symbol('psi')
c1 = QCircuit([U(theta, phi, psi, 0)], label="c1")
c2 = QCircuit([H(0)], label="c2")
c3 = QCircuit([H(0), CNOT(0, 1)], label="c3")
c4 = QCircuit([H(0), H(1), CNOT(0, 1)], label="c4")

v1 = {'θ': 1.0, 'phi': 1.0, 'psi': 1.0}
v2 = {'θ': 2.0, 'phi': 2.0, 'psi': 2.0}
v3 = {'θ': 3.0, 'phi': 3.0, 'psi': 3.0}
v4 = {'θ': 4.0, 'phi': 4.0, 'psi': 4.0}

m1 = ExpectationMeasure(Observable(pI), label="Exp1", shots=2024)
m2 = ExpectationMeasure(Observable(pX @ pZ), label="Exp2", shots=2024)
m3 = BasisMeasure(label="b3", shots=2024)
m4 = None

m_I = ExpectationMeasure(Observable(pI), label="Exp_I", shots=2024)
m_Z = ExpectationMeasure(Observable(pZ), label="Exp_Z", shots=2024)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("mode", list(BindingMode))
def test_qiskit_to_other_device_grouped_shapes(mode: BindingMode):
    binding = CircuitBinding(c1, values=[v1, v2], measurements=[m_I, m_Z], mode=mode)
    pubs_with_context = binding.to_other_device(IBMDevice.AER_SIMULATOR)
    assert len(pubs_with_context) == 1
    pub, contexts = pubs_with_context[0]
    circuit, observables, parameters = pub  # type: ignore
    expected = (
        [(v1, m_I), (v2, m_Z)]
        if mode == BindingMode.ZIP
        else [(v1, m_I), (v1, m_Z), (v2, m_I), (v2, m_Z)]
    )
    assert len(contexts) == len(expected)
    for row, (values, measure) in enumerate(expected):
        assert observables[row] == [  # type: ignore
            o.to_other_language(Language.QISKIT) for o in measure.observables
        ]
        assert parameters[row] == [values[p.name] for p in circuit.parameters]
        assert contexts[row].values == values


@pytest.mark.provider("qiskit")
def test_qiskit_to_other_device_recursive_bindings():
    inner = CircuitBinding(c1, measurements=m1)
    binding = CircuitBinding(inner, values=[v1, v2])
    pubs_with_context = binding.to_other_device(IBMDevice.AER_SIMULATOR)
    assert len(pubs_with_context) == 1
    pub, contexts = pubs_with_context[0]
    circuit, observables, parameters = pub  # type: ignore
    for row, values in enumerate([v1, v2]):
        assert observables[row] == [  # type: ignore
            o.to_other_language(Language.QISKIT) for o in m1.observables
        ]
        assert parameters[row] == [values[p.name] for p in circuit.parameters]
        assert contexts[row].values == values


@pytest.fixture
def product_tests_observable():
    cb_prod = CircuitBinding(
        circuits=c1, values=[v1, v2], measurements=m1, mode=BindingMode.PRODUCT
    )

    def val_prod(res: BatchResult):
        assert isinstance(res, BatchResult)
        for r in res.results:
            val = (
                list(r.expectation_values.values())[0]
                if isinstance(r.expectation_values, dict)
                else r.expectation_values
            )
            assert val == pytest.approx(1.0, abs=1e-5)

    cb_single = CircuitBinding(circuits=c3, measurements=m2, mode=BindingMode.PRODUCT)

    def val_single(res: BatchResult):
        assert len(res.results) == 1
        val = (
            list(res[0].expectation_values.values())[0]
            if isinstance(res[0].expectation_values, dict)
            else res[0].expectation_values
        )
        assert val == pytest.approx(0.0, abs=1e-1)

    cb_prod_ind = CircuitBinding(
        circuits=c1, values=[v1, v2, v3], measurements=m1, mode=BindingMode.PRODUCT
    )

    def val_prod_ind(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 3
        for r in res.results:
            val = (
                list(r.expectation_values.values())[0]
                if isinstance(r.expectation_values, dict)
                else r.expectation_values
            )
            assert val == pytest.approx(1.0, abs=1e-5)

    return [(cb_prod, val_prod), (cb_single, val_single), (cb_prod_ind, val_prod_ind)]


@pytest.fixture
def product_tests_sample():
    cb_prod = CircuitBinding(
        circuits=[c2, c3], measurements=m3, mode=BindingMode.PRODUCT
    )

    def val_prod(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 2

        r_c2 = res.results[0]
        assert r_c2.counts[0] == pytest.approx(1012, abs=50)
        assert r_c2.counts[1] == pytest.approx(1012, abs=50)

        r_c3 = res.results[1]
        assert r_c3.counts[0] == pytest.approx(1012, abs=50)
        assert r_c3.counts[3] == pytest.approx(1012, abs=50)

    return [(cb_prod, val_prod)]


@pytest.fixture
def product_tests_state_vector():
    cb_sv = CircuitBinding(circuits=[c1, c2, c3], values=v1, mode=BindingMode.PRODUCT)

    def val_sv(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 3
        assert np.allclose(
            res.results[0].state_vector.vector, [0.87758, 0.25903 + 0.40342j], atol=1e-5
        )
        assert np.allclose(
            res.results[1].state_vector.vector, [0.70710678, 0.70710678], atol=1e-5
        )
        assert np.allclose(
            res.results[2].state_vector.vector,
            [0.70710678, 0, 0, 0.70710678],
            atol=1e-5,
        )

    return [(cb_sv, val_sv)]


@pytest.fixture
def zip_tests_observable():
    cb_zip = CircuitBinding(
        circuits=c3,
        values=[v1, v2, v3],
        measurements=[m1, m2, m_Z],
        mode=BindingMode.ZIP,
    )

    def val_zip(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 3
        values = [1.0, 0, 0]
        for i, r in enumerate(res.results):
            val = (
                list(r.expectation_values.values())[0]
                if isinstance(r.expectation_values, dict)
                else r.expectation_values
            )
            assert val == pytest.approx(values[i], abs=1e-1)

    return [(cb_zip, val_zip)]


@pytest.fixture
def zip_tests_sample():
    cb_zip = CircuitBinding(
        circuits=[c2, c4], values=m4, measurements=[m3, m3], mode=BindingMode.ZIP
    )

    def val_zip(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 2

        r_c2 = res.results[0]
        assert r_c2.counts[0] == pytest.approx(1012, abs=50)
        assert r_c2.counts[1] == pytest.approx(1012, abs=50)

        r_c4 = res.results[1]
        assert r_c4.counts[0] == pytest.approx(505, abs=50)
        assert r_c4.counts[1] == pytest.approx(505, abs=50)
        assert r_c4.counts[2] == pytest.approx(505, abs=50)
        assert r_c4.counts[3] == pytest.approx(505, abs=50)

    return [(cb_zip, val_zip)]


@pytest.fixture
def zip_tests_recursive():
    cb_inner = CircuitBinding(circuits=c1, measurements=m1, mode=BindingMode.ZIP)
    cb_outer = CircuitBinding(
        circuits=cb_inner, values=[v1, v2], mode=BindingMode.PRODUCT
    )

    def val_outer(res: BatchResult):
        assert isinstance(res, BatchResult)
        assert len(res.results) == 2
        for r in res.results:
            val = (
                list(r.expectation_values.values())[0]
                if isinstance(r.expectation_values, dict)
                else r.expectation_values
            )
            assert val == pytest.approx(1.0, abs=1e-5)

    return [(cb_outer, val_outer)]


def execute_and_validate(
    bindings_and_validators: list[tuple[CircuitBinding, Callable[..., Any]]],
    device: AvailableDevice,
):
    for binding, validator in bindings_and_validators:
        print(binding)
        result = run(binding, device=device)
        validator(result)


@pytest.mark.provider("qiskit")
def test_qiskit_product_observable(
    product_tests_observable: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(product_tests_observable, device=IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("qiskit")
def test_qiskit_product_sample(
    product_tests_sample: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(product_tests_sample, device=IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("qiskit")
def test_qiskit_product_state_vector(
    product_tests_state_vector: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(product_tests_state_vector, device=IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("qiskit")
def test_qiskit_zip_observable(
    zip_tests_observable: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(zip_tests_observable, device=IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("qiskit")
def test_qiskit_zip_sample(
    zip_tests_sample: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(zip_tests_sample, device=IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("qiskit")
def test_qiskit_zip_recursive(
    zip_tests_recursive: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(zip_tests_recursive, device=IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("braket")
def test_braket_product_observable(
    product_tests_observable: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(
        product_tests_observable, device=AWSDevice.BRAKET_LOCAL_SIMULATOR
    )


@pytest.mark.provider("braket")
def test_braket_zip_observable(
    zip_tests_observable: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(zip_tests_observable, device=AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("braket")
def test_braket_zip_recursive(
    zip_tests_recursive: list[tuple[CircuitBinding, Callable[..., Any]]],
):
    execute_and_validate(zip_tests_recursive, device=AWSDevice.BRAKET_LOCAL_SIMULATOR)
