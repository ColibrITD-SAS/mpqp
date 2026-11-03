import numpy as np
import pytest

from mpqp import (
    ExpectationMeasure,
    BasisMeasure,
    X,
    H,
    Observable,
    QCircuit,
    Rx,
    pI,
    pX,
    pY,
    pZ,
    run,
    IBMDevice,
    AWSDevice,
)
from mpqp.core.instruction.measurement import PauliString
from mpqp.execution import adjust_measure
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.providers.providers_params import QiskitParams
from mpqp.execution.result import Result, Sample
from mpqp.tools.errors import NumberQubitsError
from mpqp.tools.maths import matrix_eq


def test_diagonal_optimization_preserves_provider_parameters(
    monkeypatch: pytest.MonkeyPatch,
):
    params = QiskitParams(
        instance="selected-instance",
        options={"environment": {"private": True, "job_tags": ["experiment"]}},
    )
    monkeypatch.setattr(params, "qctrl_function", object())

    def execute(job: Job, qiskit_params: QiskitParams | None = None) -> Result:
        assert qiskit_params is params
        assert job.job_type == JobType.SAMPLE
        assert isinstance(job.measure, BasisMeasure)
        assert job.measure.shots == 100
        job.id = "sampling-job-id"
        job.status = JobStatus.DONE
        return Result(job, [Sample(1, index=1, count=100)], shots=100)

    monkeypatch.setattr("mpqp.execution.runner.run_ibm", execute)
    circuit = QCircuit(
        [
            X(0),
            ExpectationMeasure(Observable(pZ), shots=100, optim_diagonal=True),
        ]
    )

    result = run(circuit, IBMDevice.IBM_FEZ, provider_params=params)

    assert isinstance(result, Result)
    assert result.expectation_values == pytest.approx(-1.0)
    assert result.job.id == "sampling-job-id"


@pytest.mark.parametrize(
    "measure_targets, circuit, nb_ids_before, nb_ids_after",
    [
        ([0, 1], QCircuit([H(0), Rx(1.76, 2)]), 0, 1),
        ([1, 2], QCircuit([H(0), Rx(1.76, 2)]), 1, 0),
        ([1, 2], QCircuit([H(0), Rx(1.76, 3)]), 1, 1),
    ],
)
def test_adjust_measure(
    measure_targets: list[int],
    circuit: QCircuit,
    nb_ids_before: int,
    nb_ids_after: int,
):
    obs_matrix = np.array(
        [
            [0.63, 0.5, 1, 1],
            [0.5, 0.82, 1, 1],
            [1, 1, 1, 0.33],
            [1, 1, 0.33, 0.3],
        ],
    )
    measure = ExpectationMeasure(Observable(obs_matrix), measure_targets)
    adjusted_observable_matrix = np.kron(
        np.kron(
            np.eye(2**nb_ids_before, dtype=np.complex128), measure.observables[0].matrix
        ),
        np.eye(2**nb_ids_after),
    )
    assert matrix_eq(
        adjust_measure(measure, circuit).observables[0].matrix,
        adjusted_observable_matrix,
    )


@pytest.mark.parametrize(
    ("observable", "measure_targets", "circuit_size", "expected_observable"),
    [
        (pX @ pZ, [0, 2], 3, pX @ pI @ pZ),  # ordered, non-contiguous
        (pX @ pZ, [2, 0], 3, pZ @ pI @ pX),  # unordered, non-contiguous
        (
            pX @ pY @ pZ,
            [1, 2, 0],
            4,
            pZ @ pX @ pY @ pI,
        ),  # unordered targets, multiple positions reordered
    ],
)
def test_adjust_measure_target_order(
    observable: PauliString,
    measure_targets: list[int],
    circuit_size: int,
    expected_observable: PauliString,
):
    with pytest.warns(
        UserWarning,
        match=(
            r"^Non contiguous or non sorted observable target will introduce "
            r"additional CNOT/SWAP gates\.$"
        ),
    ):
        measure = ExpectationMeasure(Observable(observable), measure_targets)

        adjusted_measure = adjust_measure(measure, QCircuit(circuit_size))

        assert adjusted_measure.targets == list(range(circuit_size))
        assert matrix_eq(
            adjusted_measure.observables[0].matrix,
            expected_observable.to_matrix(),
        )


def test_adjust_measure_matrix_reordering():
    observable = Observable((pX @ pY @ pZ).to_matrix())
    with pytest.warns(
        UserWarning,
        match=(
            r"^Non contiguous or non sorted observable target will introduce "
            r"additional CNOT/SWAP gates\.$"
        ),
    ):
        measure = ExpectationMeasure(
            observable,
            targets=[1, 2, 0],
            optimize_measurement=False,
        )
        original_matrix = observable.matrix

        adjusted_measure = adjust_measure(measure, QCircuit(3))

        assert matrix_eq(
            adjusted_measure.observables[0].matrix,
            (pZ @ pX @ pY).to_matrix(),
        )
        assert matrix_eq(measure.observables[0].matrix, original_matrix)


def test_adjust_measure_targets_mismatch():
    with pytest.raises(
        NumberQubitsError, match="Target size 2 doesn't match observable size 1"
    ):
        ExpectationMeasure(Observable(pX), targets=[0, 1])


@pytest.mark.provider("qiskit", "braket")
def test_run_batch_order_is_circuit_major():
    c1 = QCircuit([X(0), BasisMeasure([0], shots=10)], label="c1")
    c2 = QCircuit([BasisMeasure([0], shots=10)], label="c2")
    devices = [IBMDevice.AER_SIMULATOR, AWSDevice.BRAKET_LOCAL_SIMULATOR]
    batch = run([c1, c2], devices)
    order = [
        (r.job.circuit.label, r.device)  # pyright: ignore[reportAttributeAccessIssue]
        for r in batch.results
    ]
    assert order == [
        ("c1", devices[0]),
        ("c1", devices[1]),
        ("c2", devices[0]),
        ("c2", devices[1]),
    ]
