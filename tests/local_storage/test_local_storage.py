from __future__ import annotations

import inspect
import os
from copy import deepcopy
from types import TracebackType
from typing import Optional, Type

import pytest

from mpqp.all import *
from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.local_storage.delete import (
    clear_local_storage,
    remove_all_with_job_id,
    remove_jobs_with_id,
    remove_jobs_with_jobs_local_storage,
    remove_results_with_id,
    remove_results_with_job,
    remove_results_with_job_id,
    remove_results_with_result,
    remove_results_with_results_local_storage,
)
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
    jobs_local_storage_to_mpqp,
    results_local_storage_to_mpqp,
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
from mpqp.local_storage.setup import DictDB, setup_local_storage


def create_test_local_storage():
    save_local_storage = get_env_variable("DB_PATH")
    setup_local_storage("tests/local_storage/test_local_storage.db")
    clear_local_storage()
    c = QCircuit([H(0), CNOT(0, 1), BasisMeasure()], label="H CX BM")
    result = run(c, device=IBMDevice.AER_SIMULATOR)
    insert_results(result)
    result2 = run(
        c, device=[IBMDevice.AER_SIMULATOR, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR]
    )
    insert_results(result2)

    c = QCircuit([H(0), BasisMeasure()], label="H BM")
    result3 = run(
        c, device=[IBMDevice.AER_SIMULATOR, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR]
    )
    insert_results(result3)

    c1 = QCircuit([], nb_qubits=2)
    c2 = QCircuit([Id(0), Id(1)], nb_qubits=2, label="Id")
    result12 = run([c1, c2], device=IBMDevice.AER_SIMULATOR)
    insert_results(result12)
    save_env_variable("DB_PATH", save_local_storage)


@pytest.fixture
def mock_local_storage_results() -> list[dict[str, DictDB | Result]]:
    save_local_storage = get_env_variable("DB_PATH")
    setup_local_storage("tests/local_storage/test_local_storage.db")

    results = fetch_all_results()
    local_storage_results = []
    for result in results:
        local_storage_results.append(
            {
                "result_local_storage": result,
                "result": results_local_storage_to_mpqp(result)[0],
            }
        )

    save_env_variable("DB_PATH", save_local_storage)
    return local_storage_results


@pytest.fixture
def mock_local_storage_jobs() -> list[dict[str, DictDB | Job]]:
    save_local_storage = get_env_variable("DB_PATH")
    setup_local_storage("tests/local_storage/test_local_storage.db")

    jobs = fetch_all_jobs()
    local_storage_jobs = []
    for job in jobs:
        local_storage_jobs.append(
            {"job_local_storage": job, "job": jobs_local_storage_to_mpqp(job)[0]}
        )

    save_env_variable("DB_PATH", save_local_storage)
    return local_storage_jobs


class DBRunner:  # TODO: should be merge the two DbRunners ?
    def __init__(self):
        self.database_name = inspect.stack()[1].function
        self.save_local_storage = get_env_variable("DB_PATH")

    def __enter__(self):
        import shutil

        db_original = os.path.join(
            os.getcwd(), f"tests/local_storage/test_local_storage.db"
        )
        db_temp = os.path.join(
            os.getcwd(), f"tests/local_storage/test_{self.database_name}.db"
        )

        shutil.copyfile(db_original, db_temp)
        setup_local_storage(db_temp)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional["TracebackType"],
    ):
        os.remove(
            os.path.join(
                os.getcwd(), f"tests/local_storage/test_{self.database_name}.db"
            )
        )
        save_env_variable("DB_PATH", self.save_local_storage)


def test_get_all_jobs(mock_local_storage_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        jobs = get_all_jobs()

        for job, mock_local_storage_job in zip(jobs, mock_local_storage_jobs):
            assert mock_local_storage_job["job"] == job


def test_fetch_all_jobs(mock_local_storage_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        jobs = fetch_all_jobs()

        for job, mock_local_storage_job in zip(jobs, mock_local_storage_jobs):
            assert mock_local_storage_job["job_local_storage"] == job


def test_get_all_results(mock_local_storage_results: list[dict[str, DictDB | Result]]):
    with DBRunner():
        results = get_all_results()

        for result, mock_local_storage_result in zip(
            results, mock_local_storage_results
        ):
            assert result == mock_local_storage_result["result"]


def test_fetch_all_results(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        results = fetch_all_results()

        for result, mock_local_storage_result in zip(
            results, mock_local_storage_results
        ):
            assert result == mock_local_storage_result["result_local_storage"]


def test_fetch_jobs_with_job(mock_local_storage_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job = mock_local_storage_jobs[0]['job']
        expected_job = mock_local_storage_jobs[0]['job_local_storage']
        assert isinstance(job, Job)
        fetched_jobs = fetch_jobs_with_job(job)

        for fetched_job in fetched_jobs:
            fetched_job.pop('created_at', None)
            assert isinstance(expected_job, dict)
            expected_job.pop('created_at', None)
            assert fetched_job == expected_job


def test_get_job_with_id(mock_local_storage_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job_id = 1
        fetched_jobs = get_jobs_with_id(job_id)

        expected_jobs = []
        for job in mock_local_storage_jobs:
            job_local_storage = job['job_local_storage']
            assert isinstance(job_local_storage, dict)
            if job_local_storage['id'] == job_id:
                expected_jobs.append(job['job'])

        for fetched_job, expected_job in zip(fetched_jobs, expected_jobs):
            assert fetched_job == expected_job


def test_fetch_jobs_with_id(mock_local_storage_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job_id = 1
        fetched_jobs = fetch_jobs_with_id(job_id)

        expected_jobs = []
        for job in mock_local_storage_jobs:
            job_local_storage = job['job_local_storage']
            assert isinstance(job_local_storage, dict)
            if job_local_storage['id'] == job_id:
                expected_jobs.append(job_local_storage)

        for fetched_job, expected_job in zip(fetched_jobs, expected_jobs):
            assert fetched_job == expected_job


def test_get_results_with_result(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        result = mock_local_storage_results[0]['result']
        assert isinstance(result, Result)
        fetched_results = get_results_with_result(result)

        for fetched_result in fetched_results:
            assert fetched_result == result


def test_fetch_results_with_result(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        result = mock_local_storage_results[0]['result']
        assert isinstance(result, Result)
        fetched_results = fetch_results_with_result(result)
        expected_result = mock_local_storage_results[0]['result_local_storage']

        for fetched_result in fetched_results:
            assert fetched_result == expected_result


def test_get_results_with_job_id(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        job_id = 1
        fetched_results = get_results_with_job_id(job_id)

        expected_results = []
        for result in mock_local_storage_results:
            results_local_storage = result['result_local_storage']
            assert isinstance(results_local_storage, dict)
            if results_local_storage['job_id'] == job_id:
                expected_results.append(result['result'])

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_fetch_results_with_job_id(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        job_id = 1
        fetched_results = fetch_results_with_job_id(job_id)

        expected_results = []
        for result in mock_local_storage_results:
            results_local_storage = result['result_local_storage']
            assert isinstance(results_local_storage, dict)
            if results_local_storage['job_id'] == job_id:
                expected_results.append(results_local_storage)

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_get_result_with_id(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        result_id = 1
        fetched_results = get_results_with_id(result_id)

        expected_results = []
        for result in mock_local_storage_results:
            results_local_storage = result['result_local_storage']
            assert isinstance(results_local_storage, dict)
            if results_local_storage['id'] == result_id:
                expected_results.append(result['result'])

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_fetch_results_with_id(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        result_id = 1
        fetched_results = fetch_results_with_id(result_id)

        expected_results = []
        for result in mock_local_storage_results:
            results_local_storage = result['result_local_storage']
            assert isinstance(results_local_storage, dict)
            if results_local_storage['id'] == result_id:
                expected_results.append(results_local_storage)

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            fetched_result.pop('created_at', None)
            assert isinstance(expected_result, dict)
            expected_result.pop('created_at', None)
            assert fetched_result == expected_result


def test_fetch_results_with_job(
    mock_local_storage_jobs: list[dict[str, DictDB | Job]],
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        job = mock_local_storage_jobs[0]['job']
        assert isinstance(job, Job)
        fetched_results = fetch_results_with_job(job)

        expected_results = []
        for result in mock_local_storage_results:
            results = result['result']
            assert isinstance(results, Result)
            if results.job == job:
                expected_results.append(result['result_local_storage'])

        for fetched_result, expected_result in zip(fetched_results, expected_results):
            assert fetched_result == expected_result


def test_get_results_with_result_and_job(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        result = mock_local_storage_results[0]['result']
        assert isinstance(result, Result)
        fetched_results = get_results_with_result_and_job(result)

        for fetched_result in fetched_results:
            assert fetched_result == result


def test_fetch_results_with_result_and_job(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        result = mock_local_storage_results[0]['result']
        assert isinstance(result, Result)
        fetched_results = fetch_results_with_result_and_job(result)
        expected_result = mock_local_storage_results[0]['result_local_storage']

        for fetched_result in fetched_results:
            assert fetched_result == expected_result


def test_get_jobs_with_result(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
    mock_local_storage_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        result = mock_local_storage_results[0]['result']
        assert isinstance(result, Result)
        fetched_jobs = get_jobs_with_result(result)

        expected_jobs = []
        for result in mock_local_storage_results:
            results_local_storage = result['result_local_storage']
            assert isinstance(results_local_storage, dict)
            for job in mock_local_storage_jobs:
                job_local_storage = job['job_local_storage']
                assert isinstance(job_local_storage, dict)
                if results_local_storage['job_id'] == job_local_storage['id']:
                    expected_jobs.append(job['job'])

        for expected_job, fetched_job in zip(expected_jobs, fetched_jobs):
            assert fetched_job == expected_job


def test_fetch_jobs_with_result(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
    mock_local_storage_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        result = mock_local_storage_results[0]['result']
        assert isinstance(result, Result)
        fetched_jobs = fetch_jobs_with_result(result)

        results_local_storage = mock_local_storage_results[0]['result_local_storage']
        assert isinstance(results_local_storage, dict)
        expected_jobs = []
        for job in mock_local_storage_jobs:
            job_local_storage = job['job_local_storage']
            assert isinstance(job_local_storage, dict)
            if (
                results_local_storage['job_id'] == job_local_storage['id']
                and not job_local_storage in expected_jobs
            ):
                expected_jobs.append(job_local_storage)

        for expected_job, fetched_job in zip(expected_jobs, fetched_jobs):
            assert fetched_job == expected_job


def test_get_jobs_with_job(mock_local_storage_jobs: list[dict[str, DictDB | Job]]):
    with DBRunner():
        job = mock_local_storage_jobs[0]['job']
        assert isinstance(job, Job)
        fetched_jobs = get_jobs_with_job(job)

        for fetched_job in fetched_jobs:
            assert fetched_job == job


def test_local_storage_to_mpqp(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
    mock_local_storage_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        for job in mock_local_storage_jobs:
            job_local_storage = job['job_local_storage']
            assert isinstance(job_local_storage, dict)
            assert job['job'] == jobs_local_storage_to_mpqp(job_local_storage)[0]

        for result in mock_local_storage_results:
            result_local_storage = result['result_local_storage']
            assert isinstance(result_local_storage, dict)
            assert (
                result['result']
                == results_local_storage_to_mpqp(result_local_storage)[0]
            )


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


def test_local_storage_insert(circuits_type: list[QCircuit]):
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


def test_remove_jobs_with_jobs_local_storage(
    mock_local_storage_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        job_local_storage = mock_local_storage_jobs[0]['job_local_storage']
        assert isinstance(job_local_storage, dict)
        remove_jobs_with_jobs_local_storage(job_local_storage)
        jobs = fetch_all_jobs()
        for job in jobs:
            assert job != job_local_storage


def test_remove_results_with_id():
    with DBRunner():
        remove_results_with_id(1)
        results = fetch_results_with_id(1)
        assert len(results) == 0


def test_remove_results_with_result(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        result = mock_local_storage_results[0]['result']
        assert isinstance(result, Result)
        remove_results_with_result(result)
        results = get_all_results()
        for r in results:
            assert r != result


def test_remove_results_with_job(
    mock_local_storage_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        job = mock_local_storage_jobs[0]['job']
        assert isinstance(job, Job)
        job_local_storage = mock_local_storage_jobs[0]['job_local_storage']
        assert isinstance(job_local_storage, dict)
        remove_results_with_job(job)
        results = fetch_all_results()
        for r in results:
            assert r['job_id'] != job_local_storage['id']


def test_remove_results_with_job_id(
    mock_local_storage_jobs: list[dict[str, DictDB | Job]],
):
    with DBRunner():
        job_local_storage = mock_local_storage_jobs[0]['job_local_storage']
        assert isinstance(job_local_storage, dict)
        remove_results_with_job_id(job_local_storage['id'])
        results = fetch_all_results()
        for r in results:
            assert r['job_id'] != job_local_storage['id']


def test_remove_results_with_results_local_storage(
    mock_local_storage_results: list[dict[str, DictDB | Result]],
):
    with DBRunner():
        result_local_storage = mock_local_storage_results[0]['result_local_storage']
        assert isinstance(result_local_storage, dict)
        remove_results_with_results_local_storage(result_local_storage)
        results = fetch_all_results()
        for r in results:
            assert r != result_local_storage
