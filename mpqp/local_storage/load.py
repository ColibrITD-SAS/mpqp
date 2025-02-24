"""provides utility functions for converting job and result data from the local 
storage format into MPQP (:class:`~mpqp.execution.job.Job` and :class:`~mpqp.execution.result.Result`) objects. 
It also includes functions to query and retrieve jobs and results from the database.
"""

from __future__ import annotations

from typing import Optional

from mpqp.all import *
from mpqp.local_storage.queries import *
from mpqp.local_storage.setup import DictDB


def jobs_local_storage_to_mpqp(jobs: Optional[list[DictDB] | DictDB]) -> list[Job]:
    """Convert a dictionary or list of dictionaries representing jobs into MPQP Job objects.

    Args:
        jobs: A dictionary or list of dictionaries retrieved from the database.

    Returns:
        A list of MPQP Job objects.

    Example:
        >>> job_local_storage = fetch_jobs_with_id(1)
        >>> jobs_local_storage_to_mpqp(job_local_storage) # doctest: +ELLIPSIS
        [Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))]

    """
    if jobs is None:
        return []
    from numpy import array, complex64  # pyright: ignore[reportUnusedImport]

    jobs_mpqp = []
    if isinstance(jobs, dict):
        measure = eval(eval(jobs['measure'])) if jobs['measure'] is not None else None
        jobs_mpqp.append(
            Job(
                eval("JobType." + jobs['type']),
                eval(eval(jobs['circuit'])),
                eval(jobs['device']),
                measure,
            )
        )
    else:
        for job in jobs:
            measure = eval(eval(job['measure'])) if job['measure'] is not None else None
            jobs_mpqp.append(
                Job(
                    eval("JobType." + job['type']),
                    eval(eval(job['circuit'])),
                    eval(job['device']),
                    measure,
                )
            )

    return jobs_mpqp


def results_local_storage_to_mpqp(
    results: Optional[list[DictDB] | DictDB],
) -> list[Result]:
    """Convert a dictionary or list of dictionaries representing results into a
    :class:`~mpqp.execution.result.Result`.

    Args:
        results: The results retrieved from the database.

    Returns:
        The converted result(s).

    Example:
        >>> result_local_storage = fetch_results_with_id([1, 2])
        >>> results = results_local_storage_to_mpqp(result_local_storage)
        >>> for result in results:  # doctest: +ELLIPSIS
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(2, index=0, count=..., probability=0...), ...], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(2, index=0, count=..., probability=0...), ...], None, 1024)

    """
    if results is None:
        return []
    results_mpqp = []
    if isinstance(results, dict):
        error = eval(eval(results['error'])) if results['error'] is not None else None
        job = fetch_jobs_with_id(results['job_id'])
        if len(job) == 0:
            raise ValueError("Job not found for result, can not be instantiated.")
        results_mpqp.append(
            Result(
                jobs_local_storage_to_mpqp(job)[0],
                eval(eval(results['data'])),
                error,
                results['shots'],
            )
        )
    else:
        for result in results:
            error = None if result['error'] is None else eval(eval(result['error']))
            job = fetch_jobs_with_id(result['job_id'])
            if len(job) == 0:
                raise ValueError("Job not found for result, can not be instantiated.")
            results_mpqp.append(
                Result(
                    jobs_local_storage_to_mpqp(job)[0],
                    eval(eval(result['data'])),
                    error,
                    result['shots'],
                )
            )

    return results_mpqp


def get_all_jobs() -> list[Job]:
    """Retrieve all jobs from the local storage and convert them into
    :class:`~mpqp.execution.job.Job`

    Returns:
        All locally stored jobs.

    Example:
        >>> jobs = get_all_jobs()
        >>> for job in jobs: # doctest: +ELLIPSIS
        ...     print(job)
        Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
        Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
        Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
        Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
        Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)
        Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)

    """
    jobs = fetch_all_jobs()
    return jobs_local_storage_to_mpqp(jobs)


def get_all_results() -> list[Result]:
    """Retrieve all results from the local storage and convert them into
    :class:`~mpqp.execution.result.Result`

    Returns:
        All locally stored results.

    Example:
        >>> results = get_all_results()
        >>> for result in results: # doctest: +ELLIPSIS
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
        Result(Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR), StateVector(...), 0, 0)
        Result(Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR), StateVector(...), 0, 0)

    """
    results = fetch_all_results()
    return results_local_storage_to_mpqp(results)


def get_jobs_with_job(job: Job | list[Job]) -> list[Job]:
    """Retrieve job(s) matching the given job(s) attributes from the database:
        - JobType
        - Circuit
        - Device
        - Measure

    Args:
        job: Job(s) to search for.

    Returns:
        Matching job(s) corresponding to the job(s) attributes

    Example:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)
        >>> print(get_jobs_with_job(job)) # doctest: +ELLIPSIS
        [Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)]

    """
    jobs = fetch_jobs_with_job(job)
    return jobs_local_storage_to_mpqp(jobs)


def get_jobs_with_result(result: Result | list[Result] | BatchResult) -> list[Job]:
    """Retrieve job(s) associated with the given result(s) attributes from the database:
        - data
        - error
        - shots
       And also with the job attribute of the result(s):
        - JobType
        - Circuit
        - Device
        - Measure

    Args:
        result: Result(s) to find associated jobs for.

    Returns:
        Matching job(s) corresponding to the result(s) attribute and job attribute of the result(s).

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> print(get_jobs_with_result(result)) # doctest: +ELLIPSIS
        [Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)]

    """
    jobs = fetch_jobs_with_result(result)
    return jobs_local_storage_to_mpqp(jobs)


def get_results_with_result_and_job(
    result: Result | list[Result] | BatchResult,
) -> list[Result]:
    """Retrieve result(s) associated with specific result(s) attributes:
        - data
        - error
        - shots
    And also with the job attribute of the result(s):
        - JobType
        - Circuit
        - Device
        - Measure

    Args:
        result: Result(s) to search for.

    Returns:
        Matching result(s) corresponding to the result(s) attribute and job attribute of the result(s).


    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> results = get_results_with_result_and_job(result)
        >>> for result in results: # doctest: +ELLIPSIS
        ...     print(repr(result))
        Result(Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR), StateVector(...), 0, 0)

    """
    results = fetch_results_with_result_and_job(result)
    return results_local_storage_to_mpqp(results)


def get_results_with_result(
    result: Result | list[Result] | BatchResult,
) -> list[Result]:
    """Retrieve result(s) matching specific result(s) attributes:
        - data
        - error
        - shots

    Args:
        result: Result(s) to search for.

    Returns:
        Matching result(s) corresponding to the result(s) attribute.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> results = get_results_with_result(result)
        >>> for result in results: # doctest: +ELLIPSIS
        ...     print(repr(result))
        Result(Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR), StateVector(...), 0, 0)
        Result(Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR), StateVector(...), 0, 0)

    """
    results = fetch_results_with_result(result)
    return results_local_storage_to_mpqp(results)


def get_results_with_id(result_id: int | list[int]) -> list[Result]:
    """Retrieve results with the given ID(s).

    Args:
        ID(s) to search for.

    Returns:
        Matching result(s) corresponding to the id(s).

    Example:
        >>> results1 = get_results_with_id(1)
        >>> for result in results1: # doctest: +ELLIPSIS
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
        >>> results2 = get_results_with_id([2, 3])
        >>> for result in results2: # doctest: +ELLIPSIS
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)

    """
    results = fetch_results_with_id(result_id)
    return results_local_storage_to_mpqp(results)


def get_jobs_with_id(job_id: int | list[int]) -> list[Job]:
    """Retrieve jobs with the given ID(s).

    Args:
        ID(s) to search for.

    Returns:
        Job(s) corresponding to the id(s).

    Example:
        >>> jobs = get_jobs_with_id([1, 2, 3])
        >>> for job in jobs: # doctest: +ELLIPSIS
        ...     print(job)
        Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
        Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
        Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))

    """
    jobs = fetch_jobs_with_id(job_id)
    return jobs_local_storage_to_mpqp(jobs)


def get_results_with_job_id(job_id: int | list[int]) -> list[Result]:
    """Retrieve results associated with the given job ID(s).

    Args:
        ID(s) to search for.

    Returns:
        Results corresponding to the job id(s).

    Example:
        >>> results = get_results_with_job_id(1)
        >>> for result in results: # doctest: +ELLIPSIS
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(...), Sample(...)], None, 1024)
    """
    results = fetch_results_with_job_id(job_id)
    return results_local_storage_to_mpqp(results)
