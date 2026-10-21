"""IBM binding broadcasting and measurement grouping, without remote jobs."""

import numpy as np
import pytest
from sympy import Symbol

from mpqp import (
    BindingMode,
    CircuitBinding,
    ExpectationMeasure,
    IBMDevice,
    Observable,
    QCircuit,
    Ry,
    pI,
    pZ,
)
from mpqp.execution.job import Job, JobType
from mpqp.execution.providers.ibm import (
    _extract_ibm_binding_results,
    _prepare_ibm_binding,
)

pytestmark = pytest.mark.provider("qiskit")


@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
def test_binding_groups_commuting_observables(parameterized, ragged):
    from qiskit.primitives import BackendEstimatorV2
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit_aer import AerSimulator

    theta = Symbol("theta")
    circuit = QCircuit([Ry(theta if parameterized else 0.0, 0)], nb_qubits=2)
    observables = [
        Observable(pZ @ pI, "first"),
        Observable(pI @ pZ, "second"),
        Observable(pZ @ pZ, "parity"),
    ]
    measures = [
        ExpectationMeasure(observables, shots=64),
        ExpectationMeasure(observables[:1] if ragged else observables, shots=64),
    ]
    binding = CircuitBinding(
        circuit,
        values=[{theta: 0.0}, {theta: np.pi}] if parameterized else None,
        measurements=measures,
        mode=BindingMode.ZIP,
    )
    device = IBMDevice.AER_SIMULATOR
    prepared = _prepare_ibm_binding(Job(JobType.OBSERVABLE, binding, device))
    assert len(prepared.pubs) == 1
    pub = EstimatorPub.coerce(prepared.pubs[0])
    estimator = BackendEstimatorV2(
        backend=AerSimulator(), options={"abelian_grouping": True}
    )
    # One measurement basis per parameter set, not per observable.
    assert len(estimator._preprocess_pub(pub).circuits) == (2 if parameterized else 1)
    if parameterized:
        assert pub.parameter_values.shape == (2, 1)
        assert pub.shape == (2, 3)

    raw = estimator.run(prepared.pubs, precision=0.125).result()
    result = _extract_ibm_binding_results(raw, prepared, device)
    assert len(result.results) == 2
    for index, item in enumerate(result.results):
        expected = {
            "first": -1.0 if parameterized and index == 1 else 1.0,
            "second": 1.0,
            "parity": -1.0 if parameterized and index == 1 else 1.0,
        }
        if ragged and index == 1:
            assert item.expectation_values == pytest.approx(expected["first"])
        else:
            assert item.expectation_values == pytest.approx(expected)
    assert circuit.measurements == []
