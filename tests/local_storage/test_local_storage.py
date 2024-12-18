from __future__ import annotations

import inspect
import os
from copy import deepcopy
from types import TracebackType
from typing import Optional, Type

import pytest

from mpqp.all import *
from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.local_storage.load import (
    get_all_jobs,
    get_all_results,
    get_jobs_with_id,
    get_jobs_with_job,
    get_jobs_with_result,
    get_results_with_id,
    get_results_with_job_id,
    get_results_with_result,
    get_results_with_result_and_job,
    jobs_db_to_mpqp,
    results_db_to_mpqp,
)
from mpqp.local_storage.queries import (
    fetch_all_jobs,
    fetch_all_results,
    fetch_jobs_with_id,
    fetch_jobs_with_job,
    fetch_jobs_with_result,
    fetch_results_with_id,
    fetch_results_with_job,
    fetch_results_with_job_id,
    fetch_results_with_result,
    fetch_results_with_result_and_job,
)
from mpqp.local_storage.save import insert_jobs, insert_results
from mpqp.local_storage.setup import (
    DictDB,
    setup_db,
)
from mpqp.local_storage.delete import (
    remove_all_with_job_id,
    remove_jobs_with_id,
    remove_jobs_with_jobs_db,
    remove_results_with_id,
    remove_results_with_job,
    remove_results_with_job_id,
    remove_results_with_result,
    remove_results_with_results_db,
)


@pytest.fixture
def mock_db_results() -> list[dict[str, DictDB | Result]]:
    return [
        {
            'result_db': {
                'id': 1,
                'job_id': 1,
                'data': '"[Sample(2, index=0, count=532, probability=0.51953125), Sample(2, index=3, count=492, probability=0.48046875)]"',
                'error': None,
                'shots': 1024,
                'created_at': '2024-12-04 09:44:53',
            },
            'result': Result(
                Job(
                    JobType.SAMPLE,
                    QCircuit(
                        [
                            H(0),
                            CNOT(0, 1),
                            BasisMeasure(
                                [0, 1], c_targets=[0, 1], basis=ComputationalBasis()
                            ),
                        ],
                        nb_qubits=2,
                        nb_cbits=2,
                        label="H CX BM",
                    ),
                    IBMDevice.AER_SIMULATOR,
                    BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis()),
                ),
                [
                    Sample(2, index=0, count=532, probability=0.51953125),
                    Sample(2, index=3, count=492, probability=0.48046875),
                ],
                None,
                1024,
            ),
        },
        {
            'result_db': {
                'id': 2,
                'job_id': 1,
                'data': '"[Sample(2, index=0, count=489, probability=0.4775390625), Sample(2, index=3, count=535, probability=0.5224609375)]"',
                'error': None,
                'shots': 1024,
                'created_at': '2024-12-04 09:44:57',
            },
            'result': Result(
                Job(
                    JobType.SAMPLE,
                    QCircuit(
                        [
                            H(0),
                            CNOT(0, 1),
                            BasisMeasure(
                                [0, 1], c_targets=[0, 1], basis=ComputationalBasis()
                            ),
                        ],
                        nb_qubits=2,
                        nb_cbits=2,
                        label="H CX BM",
                    ),
                    IBMDevice.AER_SIMULATOR,
                    BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis()),
                ),
                [
                    Sample(2, index=0, count=489, probability=0.4775390625),
                    Sample(2, index=3, count=535, probability=0.5224609375),
                ],
                None,
                1024,
            ),
        },
        {
            'result_db': {
                'id': 3,
                'job_id': 2,
                'data': '"[Sample(2, index=0, count=507, probability=0.4951171875), Sample(2, index=3, count=517, probability=0.5048828125)]"',
                'error': None,
                'shots': 1024,
                'created_at': '2024-12-04 09:44:57',
            },
            'result': Result(
                Job(
                    JobType.SAMPLE,
                    QCircuit(
                        [
                            H(0),
                            CNOT(0, 1),
                            BasisMeasure(
                                [0, 1], c_targets=[0, 1], basis=ComputationalBasis()
                            ),
                        ],
                        nb_qubits=2,
                        nb_cbits=2,
                        label="H CX BM",
                    ),
                    GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
                    BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis()),
                ),
                [
                    Sample(2, index=0, count=507, probability=0.4951171875),
                    Sample(2, index=3, count=517, probability=0.5048828125),
                ],
                None,
                1024,
            ),
        },
        {
            'result_db': {
                'id': 4,
                'job_id': 3,
                'data': '"[Sample(1, index=0, count=502, probability=0.490234375), Sample(1, index=1, count=522, probability=0.509765625)]"',
                'error': None,
                'shots': 1024,
                'created_at': '2024-12-04 09:44:58',
            },
            'result': Result(
                Job(
                    JobType.SAMPLE,
                    QCircuit(
                        [
                            H(0),
                            BasisMeasure(
                                [0], c_targets=[0], basis=ComputationalBasis()
                            ),
                        ],
                        nb_qubits=1,
                        nb_cbits=1,
                        label="H BM",
                    ),
                    IBMDevice.AER_SIMULATOR,
                    BasisMeasure([0], c_targets=[0], basis=ComputationalBasis()),
                ),
                [
                    Sample(1, index=0, count=502, probability=0.490234375),
                    Sample(1, index=1, count=522, probability=0.509765625),
                ],
                None,
                1024,
            ),
        },
        {
            'result_db': {
                'id': 5,
                'job_id': 4,
                'data': '"[Sample(1, index=0, count=533, probability=0.5205078125), Sample(1, index=1, count=491, probability=0.4794921875)]"',
                'error': None,
                'shots': 1024,
                'created_at': '2024-12-04 09:44:58',
            },
            'result': Result(
                Job(
                    JobType.SAMPLE,
                    QCircuit(
                        [
                            H(0),
                            BasisMeasure(
                                [0], c_targets=[0], basis=ComputationalBasis()
                            ),
                        ],
                        nb_qubits=1,
                        nb_cbits=1,
                        label="H BM",
                    ),
                    GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
                    BasisMeasure([0], c_targets=[0], basis=ComputationalBasis()),
                ),
                [
                    Sample(1, index=0, count=533, probability=0.5205078125),
                    Sample(1, index=1, count=491, probability=0.4794921875),
                ],
                None,
                1024,
            ),
        },
        {
            'result_db': {
                'id': 6,
                'job_id': 5,
                'data': '"StateVector([1, 0, 0, 0])"',
                'error': '"0"',
                'shots': 0,
                'created_at': '2024-12-04 09:47:39',
            },
            'result': Result(
                Job(
                    JobType.STATE_VECTOR,
                    QCircuit([], nb_qubits=2, label="circuit 1"),
                    IBMDevice.AER_SIMULATOR,
                ),
                StateVector([1, 0, 0, 0]),  # pyright: ignore[reportArgumentType]
                0,
                0,
            ),
        },
        {
            'result_db': {
                'id': 7,
                'job_id': 6,
                'data': '"StateVector([1, 0, 0, 0])"',
                'error': '"0"',
                'shots': 0,
                'created_at': '2024-12-04 09:47:39',
            },
            'result': Result(
                Job(
                    JobType.STATE_VECTOR,
                    QCircuit([Id(0), Id(1)], nb_qubits=2, label="Id"),
                    IBMDevice.AER_SIMULATOR,
                ),
                StateVector([1, 0, 0, 0]),  # pyright: ignore[reportArgumentType]
                0,
                0,
            ),
        },
    ]


@pytest.fixture
def mock_db_jobs() -> list[dict[str, DictDB | Job]]:
    return [
        {
            'job_db': {
                'id': 1,
                'type': 'SAMPLE',
                'circuit': '"QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis())], nb_qubits=2, nb_cbits=2, label=\\"H CX BM\\")"',
                'device': 'IBMDevice.AER_SIMULATOR',
                'measure': '"BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis())"',
                'created_at': '2024-12-04 09:44:53',
            },
            'job': Job(
                JobType.SAMPLE,
                QCircuit(
                    [
                        H(0),
                        CNOT(0, 1),
                        BasisMeasure(
                            [0, 1], c_targets=[0, 1], basis=ComputationalBasis()
                        ),
                    ],
                    nb_qubits=2,
                    nb_cbits=2,
                    label="H CX BM",
                ),
                IBMDevice.AER_SIMULATOR,
                BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis()),
            ),
        },
        {
            'job_db': {
                'id': 2,
                'type': 'SAMPLE',
                'circuit': '"QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis())], nb_qubits=2, nb_cbits=2, label=\\"H CX BM\\")"',
                'device': 'GOOGLEDevice.CIRQ_LOCAL_SIMULATOR',
                'measure': '"BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis())"',
                'created_at': '2024-12-04 09:44:57',
            },
            'job': Job(
                JobType.SAMPLE,
                QCircuit(
                    [
                        H(0),
                        CNOT(0, 1),
                        BasisMeasure(
                            [0, 1], c_targets=[0, 1], basis=ComputationalBasis()
                        ),
                    ],
                    nb_qubits=2,
                    nb_cbits=2,
                    label="H CX BM",
                ),
                GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
                BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis()),
            ),
        },
        {
            'job_db': {
                'id': 3,
                'type': 'SAMPLE',
                'circuit': '"QCircuit([H(0), BasisMeasure([0], c_targets=[0], basis=ComputationalBasis())], nb_qubits=1, nb_cbits=1, label=\\"H BM\\")"',
                'device': 'IBMDevice.AER_SIMULATOR',
                'measure': '"BasisMeasure([0], c_targets=[0], basis=ComputationalBasis())"',
                'created_at': '2024-12-04 09:44:58',
            },
            'job': Job(
                JobType.SAMPLE,
                QCircuit(
                    [
                        H(0),
                        BasisMeasure([0], c_targets=[0], basis=ComputationalBasis()),
                    ],
                    nb_qubits=1,
                    nb_cbits=1,
                    label="H BM",
                ),
                IBMDevice.AER_SIMULATOR,
                BasisMeasure([0], c_targets=[0], basis=ComputationalBasis()),
            ),
        },
        {
            'job_db': {
                'id': 4,
                'type': 'SAMPLE',
                'circuit': '"QCircuit([H(0), BasisMeasure([0], c_targets=[0], basis=ComputationalBasis())], nb_qubits=1, nb_cbits=1, label=\\"H BM\\")"',
                'device': 'GOOGLEDevice.CIRQ_LOCAL_SIMULATOR',
                'measure': '"BasisMeasure([0], c_targets=[0], basis=ComputationalBasis())"',
                'created_at': '2024-12-04 09:44:58',
            },
            'job': Job(
                JobType.SAMPLE,
                QCircuit(
                    [
                        H(0),
                        BasisMeasure([0], c_targets=[0], basis=ComputationalBasis()),
                    ],
                    nb_qubits=1,
                    nb_cbits=1,
                    label="H BM",
                ),
                GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
                BasisMeasure([0], c_targets=[0], basis=ComputationalBasis()),
            ),
        },
        {
            'job_db': {
                'id': 5,
                'type': 'STATE_VECTOR',
                'circuit': '"QCircuit([], nb_qubits=2, label=\\"circuit 1\\")"',
                'device': 'IBMDevice.AER_SIMULATOR',
                'measure': None,
                'created_at': '2024-12-04 09:47:39',
            },
            'job': Job(
                JobType.STATE_VECTOR,
                QCircuit([], nb_qubits=2, label="circuit 1"),
                IBMDevice.AER_SIMULATOR,
            ),
        },
        {
            'job_db': {
                'id': 6,
                'type': 'STATE_VECTOR',
                'circuit': '"QCircuit([Id(0), Id(1)], nb_qubits=2, label=\\"Id\\")"',
                'device': 'IBMDevice.AER_SIMULATOR',
                'measure': None,
                'created_at': '2024-12-04 09:47:39',
            },
            'job': Job(
                JobType.STATE_VECTOR,
                QCircuit([Id(0), Id(1)], nb_qubits=2, label="Id"),
                IBMDevice.AER_SIMULATOR,
            ),
        },
    ]


class DBRunner:  # TODO: should be merge the two DbRunners ?
    def __init__(self):
        self.database_name = inspect.stack()[1].function
        self.save_db = get_env_variable("DATA_BASE")

    def __enter__(self):
        import shutil

        db_original = os.path.join(os.getcwd(), f"tests/test_database.db")
        db_temp = os.path.join(os.getcwd(), f"tests/test_{self.database_name}.db")

        shutil.copyfile(db_original, db_temp)
        setup_db(db_temp)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional["TracebackType"],
    ):
        os.remove(os.path.join(os.getcwd(), f"tests/test_{self.database_name}.db"))
        save_env_variable("DATA_BASE", self.save_db)


def test_get_all_jobs(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        jobs = get_all_jobs()

        for job, mock_db_job in zip(jobs, mock_db_jobs):
            assert mock_db_job["job"] == job


def test_fetch_all_jobs(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        jobs = fetch_all_jobs()

        for job, mock_db_job in zip(jobs, mock_db_jobs):
            assert mock_db_job["job_db"] == job


def test_get_all_results(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        results = get_all_results()

        for result, mock_db_result in zip(results, mock_db_results):
            assert result == mock_db_result["result"]


def test_fetch_all_results(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        results = fetch_all_results()

        for result, mock_db_result in zip(results, mock_db_results):
            assert result == mock_db_result["result_db"]


def test_fetch_jobs_with_job(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job = mock_db_jobs[0]['job']
        expected_job = mock_db_jobs[0]['job_db']
        assert isinstance(job, Job)
        fetched_jobs = fetch_jobs_with_job(job)

        for fetched_job in fetched_jobs:
            assert fetched_job == expected_job


def test_get_job_with_id(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job_id = 1
        fetched_jobs = get_jobs_with_id(job_id)

        expected_jobs = []
        for job in mock_db_jobs:
            job_db = job['job_db']
            assert isinstance(job_db, dict)
            if job_db['id'] == job_id:
                expected_jobs.append(job['job'])

        for fetched_job, expected_job in zip(fetched_jobs, expected_jobs):
            assert fetched_job == expected_job


def test_fetch_jobs_with_id(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job_id = 1
        fetched_jobs = fetch_jobs_with_id(job_id)

        expected_jobs = []
        for job in mock_db_jobs:
            job_db = job['job_db']
            assert isinstance(job_db, dict)
            if job_db['id'] == job_id:
                expected_jobs.append(job_db)

        for fetched_job, expected_job in zip(fetched_jobs, expected_jobs):
            assert fetched_job == expected_job


def test_get_results_with_result(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        result = mock_db_results[0]['result']
        assert isinstance(result, Result)
        fetched_results = get_results_with_result(result)

        for fetched_result in fetched_results:
            assert fetched_result == result


def test_fetch_results_with_result(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        result = mock_db_results[0]['result']
        assert isinstance(result, Result)
        fetched_results = fetch_results_with_result(result)
        expected_result = mock_db_results[0]['result_db']

        for fetched_result in fetched_results:
            assert fetched_result == expected_result


def test_get_results_with_job_id(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        job_id = 1
        fetched_results = get_results_with_job_id(job_id)

        expected_results = []
        for result in mock_db_results:
            results_db = result['result_db']
            assert isinstance(results_db, dict)
            if results_db['job_id'] == job_id:
                expected_results.append(result['result'])

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_fetch_results_with_job_id(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        job_id = 1
        fetched_results = fetch_results_with_job_id(job_id)

        expected_results = []
        for result in mock_db_results:
            results_db = result['result_db']
            assert isinstance(results_db, dict)
            if results_db['job_id'] == job_id:
                expected_results.append(results_db)

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_get_result_with_id(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        result_id = 1
        fetched_results = get_results_with_id(result_id)

        expected_results = []
        for result in mock_db_results:
            results_db = result['result_db']
            assert isinstance(results_db, dict)
            if results_db['id'] == result_id:
                expected_results.append(result['result'])

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_fetch_results_with_id(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        result_id = 1
        fetched_results = fetch_results_with_id(result_id)

        expected_results = []
        for result in mock_db_results:
            results_db = result['result_db']
            assert isinstance(results_db, dict)
            if results_db['id'] == result_id:
                expected_results.append(results_db)

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_fetch_results_with_job(
    mock_db_jobs: list[dict[str, DictDB | Job]],
    mock_db_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        job = mock_db_jobs[0]['job']
        assert isinstance(job, Job)
        fetched_results = fetch_results_with_job(job)

        expected_results = []
        for result in mock_db_results:
            results = result['result']
            assert isinstance(results, Result)
            if results.job == job:
                expected_results.append(result['result_db'])

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_get_results_with_result_and_job(
    mock_db_results: list[dict[str, DictDB | Result]]
):
    with DBRunner():
        result = mock_db_results[0]['result']
        assert isinstance(result, Result)
        fetched_results = get_results_with_result_and_job(result)

        for fetched_result in fetched_results:
            assert fetched_result == result


def test_fetch_results_with_result_and_job(
    mock_db_results: list[dict[str, DictDB | Result]]
):
    with DBRunner():
        result = mock_db_results[0]['result']
        assert isinstance(result, Result)
        fetched_results = fetch_results_with_result_and_job(result)
        expected_result = mock_db_results[0]['result_db']

        for fetched_result in fetched_results:
            assert fetched_result == expected_result


def test_get_jobs_with_result(
    mock_db_results: list[dict[str, DictDB | Result]],
    mock_db_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        result = mock_db_results[0]['result']
        assert isinstance(result, Result)
        fetched_jobs = get_jobs_with_result(result)

        expected_jobs = []
        for result in mock_db_results:
            results_db = result['result_db']
            assert isinstance(results_db, dict)
            for job in mock_db_jobs:
                job_db = job['job_db']
                assert isinstance(job_db, dict)
                if results_db['job_id'] == job_db['id']:
                    expected_jobs.append(job['job'])

        for expected_job, fetched_job in zip(expected_jobs, fetched_jobs):
            assert fetched_job == expected_job


def test_fetch_jobs_with_result(
    mock_db_results: list[dict[str, DictDB | Result]],
    mock_db_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        result = mock_db_results[0]['result']
        assert isinstance(result, Result)
        fetched_jobs = fetch_jobs_with_result(result)

        results_db = mock_db_results[0]['result_db']
        assert isinstance(results_db, dict)
        expected_jobs = []
        for job in mock_db_jobs:
            job_db = job['job_db']
            assert isinstance(job_db, dict)
            if results_db['job_id'] == job_db['id'] and not job_db in expected_jobs:
                expected_jobs.append(job_db)

        for expected_job, fetched_job in zip(expected_jobs, fetched_jobs):
            assert fetched_job == expected_job


def test_get_jobs_with_job(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job = mock_db_jobs[0]['job']
        assert isinstance(job, Job)
        fetched_jobs = get_jobs_with_job(job)

        for fetched_job in fetched_jobs:
            assert fetched_job == job


def test_db_to_mpqp(
    mock_db_results: list[dict[str, DictDB | Result]],
    mock_db_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        for job in mock_db_jobs:
            job_db = job['job_db']
            assert isinstance(job_db, dict)
            assert job['job'] == jobs_db_to_mpqp(job_db)[0]

        for result in mock_db_results:
            result_db = result['result_db']
            assert isinstance(result_db, dict)
            assert result['result'] == results_db_to_mpqp(result_db)[0]


@pytest.fixture
def circuits_type():
    circuit_state_vector = QCircuit([H(0), H(1), CNOT(0, 1)])

    circuit_samples = deepcopy(circuit_state_vector)
    circuit_samples.add(BasisMeasure())

    observable = np.array([[4, 2, 3, 8], [2, -3, 1, 0], [3, 1, -1, 5], [8, 0, 5, 2]])
    circuit_observable = deepcopy(circuit_state_vector)
    circuit_observable.add(ExpectationMeasure(Observable(observable)))

    return [
        circuit_state_vector,
        circuit_samples,
        circuit_observable,
    ]


def test_db_insert(circuits_type: list[QCircuit]):
    with DBRunner():
        for circuit in circuits_type:
            results = run(circuit, IBMDevice.AER_SIMULATOR)
            ids = insert_results(results)
            print(ids)
            for id in ids:
                assert id is not None
                assert results == get_results_with_id(id)[0]


def test_insert_job(circuits_type: list[QCircuit]):
    with DBRunner():
        for circuit in circuits_type:
            results = run(circuit, IBMDevice.AER_SIMULATOR)
            assert isinstance(results, Result)
            ids = insert_jobs(results.job)
            for id in ids:
                assert id is not None
                assert results.job == get_jobs_with_id(id)[0]


def test_remove_all_with_job_id():
    with DBRunner():
        remove_all_with_job_id(1)
        jobs = fetch_jobs_with_id(1)
        assert len(jobs) == 0
        results = fetch_results_with_job_id(1)
        assert len(results) == 0


def test_remove_jobs_with_id():
    with DBRunner():
        remove_jobs_with_id(1)
        jobs = fetch_jobs_with_id(1)
        assert len(jobs) == 0


def test_remove_jobs_with_jobs_db(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job_db = mock_db_jobs[0]['job_db']
        assert isinstance(job_db, dict)
        remove_jobs_with_jobs_db(job_db)
        jobs = fetch_all_jobs()
        for job in jobs:
            assert job != job_db


def test_remove_results_with_id():
    with DBRunner():
        remove_results_with_id(1)
        results = fetch_results_with_id(1)
        assert len(results) == 0


def test_remove_results_with_result(mock_db_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        result = mock_db_results[0]['result']
        assert isinstance(result, Result)
        remove_results_with_result(result)
        results = get_all_results()
        for r in results:
            assert r != result


def test_remove_results_with_job(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job = mock_db_jobs[0]['job']
        assert isinstance(job, Job)
        job_db = mock_db_jobs[0]['job_db']
        assert isinstance(job_db, dict)
        remove_results_with_job(job)
        results = fetch_all_results()
        for r in results:
            assert r['job_id'] != job_db['id']


def test_remove_results_with_job_id(mock_db_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job_db = mock_db_jobs[0]['job_db']
        assert isinstance(job_db, dict)
        remove_results_with_job_id(job_db['id'])
        results = fetch_all_results()
        for r in results:
            assert r['job_id'] != job_db['id']


def test_remove_results_with_results_db(
    mock_db_results: list[dict[str, DictDB | Result]]
):
    with DBRunner():
        result_db = mock_db_results[0]['result_db']
        assert isinstance(result_db, dict)
        remove_results_with_results_db(result_db)
        results = fetch_all_results()
        for r in results:
            assert r != result_db
