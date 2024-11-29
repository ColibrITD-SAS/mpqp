from __future__ import annotations

from typing import Any, Optional
from mpqp.all import *
from mpqp.db.db_query import *
from mpqp.execution.runner import generate_job


def jobs_db_to_mpqp(jobs: Optional[list[dict[Any, Any]] | dict[Any, Any]]) -> list[Job]:
    """
    Convert a dictionary or list of dictionaries representing jobs into MPQP Job objects.

    Args:
        jobs: A dictionary or list of dictionaries retrieved from the database.

    Returns:
        A list of MPQP Job objects.

    Example:
        >>> job_db = fetch_jobs_with_id(1)
        >>> jobs_db_to_mpqp(job_db)
        [Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure())]

    """
    if jobs is None:
        return []
    jobs_mpqp = []
    if isinstance(jobs, dict):
        jobs_mpqp.append(
            generate_job(eval(eval(jobs['circuit'])), eval(jobs['device']))
        )
    else:
        for job in jobs:
            jobs_mpqp.append(
                generate_job(eval(eval(job['circuit'])), eval(job['device']))
            )

    return jobs_mpqp


def results_db_to_mpqp(
    results: Optional[list[dict[Any, Any]] | dict[Any, Any]]
) -> list[Result]:
    """
    Convert a dictionary or list of dictionaries representing results into MPQP Result objects.

    Args:
        results: A dictionary or list of dictionaries retrieved from the database.

    Returns:
        A list of MPQP Result objects.

    Example:
        >>> result_db = fetch_results_with_id([1, 2])
        >>> results = results_db_to_mpqp(result_db)
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=495, probability=0.4833984375), Sample(2, index=3, count=529, probability=0.5166015625)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=518, probability=0.505859375), Sample(2, index=3, count=506, probability=0.494140625)], None, 1024)

    """
    if results is None:
        return []
    results_mpqp = []
    if isinstance(results, dict):
        error = None if results['error'] is None else eval(results['error'])
        job = fetch_jobs_with_id(results['job_id'])
        if len(job) == 0:
            raise ValueError("Job not found for result, can not be instantiated.")
        results_mpqp.append(
            Result(
                jobs_db_to_mpqp(job)[0],
                eval(eval(results['data'])),
                error,
                results['shots'],
            )
        )
    else:
        for result in results:
            error = None if result['error'] is None else eval(result['error'])
            job = fetch_jobs_with_id(result['job_id'])
            if len(job) == 0:
                raise ValueError("Job not found for result, can not be instantiated.")
            results_mpqp.append(
                Result(
                    jobs_db_to_mpqp(job)[0],
                    eval(eval(result['data'])),
                    error,
                    result['shots'],
                )
            )

    return results_mpqp


def get_all_jobs() -> list[Job]:
    """
    Retrieve all jobs from the database and convert them into MPQP Job objects.

    Returns:
        A list of all MPQP Job objects.

    Example:
        >>> jobs = get_all_jobs()
        >>> for job in jobs:
        ...     print(job)
        Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure())
        Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure())
        Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure()], nb_qubits=1, nb_cbits=1, label="H BM"), IBMDevice.AER_SIMULATOR, BasisMeasure())
        Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure()], nb_qubits=1, nb_cbits=1, label="H BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure())
        Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, nb_cbits=None, label="None"), IBMDevice.AER_SIMULATOR, None)
        Job(JobType.STATE_VECTOR, QCircuit([Id(0), Id(1)], nb_qubits=2, nb_cbits=None, label="Id"), IBMDevice.AER_SIMULATOR, None)

    """
    jobs = fetch_all_jobs()
    return jobs_db_to_mpqp(jobs)


def get_all_results() -> list[Result]:
    """
    Retrieve all results from the database and convert them into MPQP Result objects.

    Returns:
        A list of all MPQP Result objects.

    Example:
        >>> results = get_all_results()
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=495, probability=0.4833984375), Sample(2, index=3, count=529, probability=0.5166015625)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=518, probability=0.505859375), Sample(2, index=3, count=506, probability=0.494140625)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=494, probability=0.482421875), Sample(2, index=3, count=530, probability=0.517578125)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure()], nb_qubits=1, nb_cbits=1, label="H BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(1, index=0, count=529, probability=0.5166015625), Sample(1, index=1, count=495, probability=0.4833984375)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure()], nb_qubits=1, nb_cbits=1, label="H BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure()), [Sample(1, index=0, count=524, probability=0.51171875), Sample(1, index=1, count=500, probability=0.48828125)], None, 1024)
        Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, nb_cbits=None, label="None"), IBMDevice.AER_SIMULATOR, None), StateVector([1, 0, 0, 0]), None, 0)
        Result(Job(JobType.STATE_VECTOR, QCircuit([Id(0), Id(1)], nb_qubits=2, nb_cbits=None, label="Id"), IBMDevice.AER_SIMULATOR, None), StateVector([1, 0, 0, 0]), None, 0)

    """
    results = fetch_all_results()
    return results_db_to_mpqp(results)


def get_jobs_with_job(job: Job | list[Job]) -> list[Job]:
    """
    Retrieve jobs matching the given job(s) from the database.

    Args:
        A single job or a list of jobs to search for.

    Returns:
        A list of matching MPQP Job objects.

    Example:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        >>> print(get_jobs_with_job(job))
        [Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, nb_cbits=None, label="None"), IBMDevice.AER_SIMULATOR, None)]

    """
    jobs = fetch_jobs_with_job(job)
    return jobs_db_to_mpqp(jobs)


def get_jobs_with_result(result: Result | list[Result] | BatchResult) -> list[Job]:
    """
    Retrieve jobs associated with the given result(s) from the database.

    Args:
        result: Result(s) to find associated jobs for.

    Returns:
        A list of matching MPQP Job objects.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]))
        >>> print(get_jobs_with_result(result))
        [Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, nb_cbits=None, label="None"), IBMDevice.AER_SIMULATOR, None)]

    """
    jobs = fetch_jobs_with_result(result)
    return jobs_db_to_mpqp(jobs)


def get_results_with_result_and_job(
    result: Result | list[Result] | BatchResult,
) -> list[Result]:
    """
    Retrieve results with matching result and job data.

    Args:
        result: Result(s) to search for.

    Returns:
        A list of matching MPQP Result objects.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]))
        >>> results = get_results_with_result_and_job(result)
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, nb_cbits=None, label="None"), IBMDevice.AER_SIMULATOR, None), StateVector([1, 0, 0, 0]), None, 0)

    """
    results = fetch_results_with_result_and_job(result)
    return results_db_to_mpqp(results)


def get_results_with_result(
    result: Result | list[Result] | BatchResult,
) -> list[Result]:
    """
    Retrieve results matching the given result(s) from the database.

    Args:
        result: Result(s) to search for.

    Returns:
        A list of matching MPQP Result objects.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]))
        >>> results = get_results_with_result(result)
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, nb_cbits=None, label="None"), IBMDevice.AER_SIMULATOR, None), StateVector([1, 0, 0, 0]), None, 0)
        Result(Job(JobType.STATE_VECTOR, QCircuit([Id(0), Id(1)], nb_qubits=2, nb_cbits=None, label="Id"), IBMDevice.AER_SIMULATOR, None), StateVector([1, 0, 0, 0]), None, 0)

    """
    results = fetch_results_with_result(result)
    return results_db_to_mpqp(results)


def get_result_with_id(result_id: int | list[int]) -> list[Result]:
    """
    Retrieve results with the given IDs.

    Args:
        A single result ID or a list of IDs to search for.

    Returns:
        A list of MPQP Result objects with the given IDs.

    Example:
        >>> results = get_result_with_id([1, 2, 3])
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=495, probability=0.4833984375), Sample(2, index=3, count=529, probability=0.5166015625)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=518, probability=0.505859375), Sample(2, index=3, count=506, probability=0.494140625)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=494, probability=0.482421875), Sample(2, index=3, count=530, probability=0.517578125)], None, 1024)

    """
    results = fetch_results_with_id(result_id)
    return results_db_to_mpqp(results)


def get_job_with_id(job_id: int | list[int]) -> list[Job]:
    """
    Retrieve jobs with the given IDs.

    Args:
        A single job ID or a list of IDs to search for.

    Returns:
        A list of MPQP Job objects with the given IDs.

    Example:
        >>> jobs = get_job_with_id([1, 2, 3])
        >>> for job in jobs:
        ...     print(job)
        Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure())
        Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure())
        Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure()], nb_qubits=1, nb_cbits=1, label="H BM"), IBMDevice.AER_SIMULATOR, BasisMeasure())

    """
    jobs = fetch_jobs_with_id(job_id)
    return jobs_db_to_mpqp(jobs)


def get_results_with_job_id(job_id: int | list[int]) -> list[Result]:
    """
    Retrieve results associated with the given job ID(s).

    Args:
        A single job ID or a list of IDs to search for.

    Returns:
        A list of MPQP Result objects associated with the given job ID(s).

    Example:
        >>> results = get_results_with_job_id(1)
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=495, probability=0.4833984375), Sample(2, index=3, count=529, probability=0.5166015625)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure()], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure()), [Sample(2, index=0, count=518, probability=0.505859375), Sample(2, index=3, count=506, probability=0.494140625)], None, 1024)

    """
    results = fetch_results_with_job_id(job_id)
    return results_db_to_mpqp(results)
