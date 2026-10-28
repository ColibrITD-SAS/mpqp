import sys

import numpy as np
import pytest
from sympy import Symbol

from mpqp import (
    CNOT,
    AWSDevice,
    BasisMeasure,
    ExpectationMeasure,
    H,
    Observable,
    QCircuit,
    Result,
    Ry,
    X,
    run,
)
from mpqp.core.binding import CircuitBinding
from mpqp.execution.providers.aws import (
    _ordered_measurement_probabilities,  # pyright: ignore[reportPrivateUsage]
)
from mpqp.execution.providers.aws import (
    run_braket,
    submit_job_braket,
)
from mpqp.execution.result import BatchResult
from mpqp.execution.runner import generate_job
from mpqp.measures import pI, pX, pZ
from mpqp.noise import BitFlip


@pytest.mark.provider("braket")
@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize(
    ("targets", "expected_probabilities"),
    [
        (None, [0.25, 0.25, 0.25, 0.25]),
        ([0], [0, 0.5, 0, 0.5]),
    ],
    ids=["dynamic", "subset"],
)
def test_braket_binding_noise_targets(
    cached: bool,
    targets: list[int] | None,
    expected_probabilities: list[float],
):
    circuit = QCircuit([X(0), X(1)])
    device = AWSDevice.BRAKET_LOCAL_SIMULATOR
    if cached:
        circuit.transpiled_circuit = circuit.to_other_device(device)
    binding = CircuitBinding(
        circuit,
        measurements=BasisMeasure([0, 1], shots=2048),
        noises=[BitFlip(0.5, targets=targets)],
    )

    result = run(binding, device)

    assert isinstance(result, BatchResult)
    np.testing.assert_allclose(
        result.results[0].probabilities, expected_probabilities, atol=0.06, rtol=0
    )


@pytest.mark.provider("braket")
def test_braket_binding_empty_noise_targets():
    circuit = QCircuit([X(0)])
    device = AWSDevice.BRAKET_LOCAL_SIMULATOR
    circuit.transpiled_circuit = circuit.to_other_device(device)
    binding = CircuitBinding(
        circuit,
        measurements=BasisMeasure([0], shots=100),
        noises=[BitFlip(0.5, targets=[])],
    )

    result = run(binding, device)

    assert isinstance(result, BatchResult)
    np.testing.assert_array_equal(result.results[0].probabilities, [0, 1])


def test_ordered_measurement_probabilities_follows_observable_targets():
    probabilities = _ordered_measurement_probabilities(
        {"10": 0.75, "01": 0.25},
        measured_qubits=[0, 2],
        targets=[2, 0],
    )

    np.testing.assert_array_equal(probabilities, [0, 0.75, 0.25, 0])

    with pytest.raises(ValueError, match="did not measure observable target 1"):
        _ordered_measurement_probabilities(
            {"10": 1}, measured_qubits=[0, 2], targets=[2, 1]
        )


def test_generate_braket_job_preserves_matrix_observable_representation():
    matrix = np.kron(
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.eye(2, dtype=np.complex128),
    )
    circuit = QCircuit(
        [
            H(2),
            ExpectationMeasure(
                Observable(matrix),
                targets=[2, 0],
                shots=100,
                optimize_measurement=True,
            ),
        ]
    )

    job = generate_job(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    measure = job.measure
    assert isinstance(measure, ExpectationMeasure)
    assert measure.targets == [0, 2]
    assert (
        measure.observables[0]._pauli_string  # pyright: ignore[reportPrivateUsage]
        is None
    )
    np.testing.assert_array_equal(
        measure.observables[0].matrix,
        np.kron(
            np.eye(2, dtype=np.complex128),
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
        ),
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circuit",
    [
        QCircuit([H(2)]),
        QCircuit([H(0), H(2)]),
        QCircuit([H(0), CNOT(1, 3), H(2)]),
        QCircuit([H(2)], nb_qubits=5),
    ],
)
def test_braket_non_contiguous_qubits(circuit: QCircuit):
    run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("braket")
def test_submit_job_braket_binds_task_id():
    job = generate_job(QCircuit([H(0)]), AWSDevice.BRAKET_LOCAL_SIMULATOR)

    job_id, task = submit_job_braket(job)

    assert job_id == task.id
    assert job.id == task.id


@pytest.mark.provider("braket")
def test_run_braket_preserves_task_id_in_result():
    job = generate_job(QCircuit([H(0)]), AWSDevice.BRAKET_LOCAL_SIMULATOR)

    result = run_braket(job)

    assert isinstance(result, Result)
    assert result.job is job
    assert isinstance(result.job.id, str)
    assert result.job.id


@pytest.mark.provider("braket")
def test_braket_circuit_binding_binds_task_ids():
    theta = Symbol("theta")
    binding = CircuitBinding(
        QCircuit([Ry(theta, 0)]),
        values=[{theta: 0.0}, {theta: np.pi}],
    )
    job = generate_job(binding, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    result = run_braket(job)

    assert isinstance(result, BatchResult)
    assert isinstance(job.id, list)
    assert len(result.results) == 2
    assert [execution.job.id for execution in result.results] == job.id


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "observables, optimize_measurement, shots",
    [
        (
            [
                Observable(np.array([[1, 0.2], [0.2, -1]], dtype=np.complex128)),
                Observable(np.array([[0.5, 0.3j], [-0.3j, -0.5]], dtype=np.complex128)),
            ],
            False,
            0,
        ),
        ([Observable(pX), Observable(pZ)], True, 0),
        ([Observable(pX), Observable(pZ)], False, 0),
        (
            [
                Observable(np.array([[1, 0.2], [0.2, -1]], dtype=np.complex128)),
                Observable(np.array([[0.5, 0.3j], [-0.3j, -0.5]], dtype=np.complex128)),
            ],
            False,
            100,
        ),
        ([Observable(pX), Observable(pZ)], True, 100),
        ([Observable(pX), Observable(pZ)], False, 100),
    ],
    ids=[
        "exact-hermitian",
        "exact-pauli-grouping",
        "exact-pauli",
        "sampled-hermitian",
        "sampled-pauli-grouping",
        "sampled-pauli",
    ],
)
def test_run_braket_observables_bind_single_task_id(
    observables: list[Observable],
    optimize_measurement: bool,
    shots: int,
):
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                observables,
                shots=shots,
                optimize_measurement=optimize_measurement,
            ),
        ]
    )

    result = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert isinstance(result.job.id, str)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("shots", [0, 100], ids=["exact", "sampled"])
def test_run_braket_preserves_pauli_coefficient(shots: int):
    observable = Observable(2 * pX)
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                observable,
                shots=shots,
                optimize_measurement=False,
            ),
        ]
    )

    result = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert result.expectation_values == pytest.approx(2)
    measure = result.job.measure
    assert isinstance(measure, ExpectationMeasure)
    assert measure.observables[0]._matrix is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.provider("braket")
def test_run_braket_exact_pauli_sums_do_not_materialize_matrices():
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                [Observable(2 * pX + pZ), Observable(-pX)],
                shots=0,
                optimize_measurement=False,
            ),
        ]
    )

    result = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert isinstance(result.job.id, str)
    assert result.expectation_values == pytest.approx(
        {"observable_0": 2, "observable_1": -1}
    )
    measure = result.job.measure
    assert isinstance(measure, ExpectationMeasure)
    assert all(
        observable._matrix is None  # pyright: ignore[reportPrivateUsage]
        for observable in measure.observables
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize("shots", [0, 100], ids=["exact", "sampled"])
def test_run_braket_matrix_does_not_materialize_pauli(shots: int):
    matrix = np.kron(
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.eye(2, dtype=np.complex128),
    )
    circuit = QCircuit(
        [
            X(0),
            H(2),
            ExpectationMeasure(
                Observable(matrix),
                targets=[2, 0],
                shots=shots,
                optimize_measurement=True,
            ),
        ]
    )

    result = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert isinstance(result.job.id, str)
    assert result.expectation_values == pytest.approx(1)
    measure = result.job.measure
    assert isinstance(measure, ExpectationMeasure)
    assert measure.targets == [0, 2]
    assert (
        measure.observables[0]._pauli_string  # pyright: ignore[reportPrivateUsage]
        is None
    )


@pytest.mark.provider("braket")
def test_run_braket_mixed_observables_preserve_representations():
    matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                [Observable(matrix), Observable(2 * pX)],
                shots=100,
                optimize_measurement=True,
            ),
        ]
    )

    result = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert isinstance(result.job.id, str)
    assert result.expectation_values == pytest.approx(
        {"observable_0": 1, "observable_1": 2}
    )
    measure = result.job.measure
    assert isinstance(measure, ExpectationMeasure)
    assert (
        measure.observables[0]._pauli_string  # pyright: ignore[reportPrivateUsage]
        is None
    )
    assert measure.observables[1]._matrix is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.provider("braket")
def test_run_braket_optimized_pre_measure_preserves_target_order():
    circuit = QCircuit(
        [
            X(0),
            H(2),
            ExpectationMeasure(
                [Observable(pX @ pI), Observable(pI @ pZ)],
                targets=[2, 0],
                shots=100,
                optimize_measurement=True,
            ),
        ]
    )

    result = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert isinstance(result.job.id, str)
    assert result.expectation_values == pytest.approx(
        {"observable_0": 1, "observable_1": -1}
    )


@pytest.mark.provider("braket")
def test_run_braket_hermitian_pre_measures_preserve_results():
    pauli_x_matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                [Observable(pauli_x_matrix), Observable(2 * pauli_x_matrix)],
                targets=[0],
                shots=100,
                optimize_measurement=False,
            ),
        ]
    )

    result = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert isinstance(result.job.id, str)
    assert result.expectation_values == pytest.approx(
        {"observable_0": 1, "observable_1": 2}
    )


@pytest.mark.provider("braket")
def remote_braket_execution_binds_task_arn():
    circuit = QCircuit([H(0), BasisMeasure([0], shots=1)])

    result = run(circuit, AWSDevice.BRAKET_SV1_SIMULATOR)

    assert isinstance(result.job.id, str)
    assert result.job.id.startswith("arn:aws:braket:")


if "--long-costly" in sys.argv:
    test_remote_braket_execution_binds_task_arn = remote_braket_execution_binds_task_arn
