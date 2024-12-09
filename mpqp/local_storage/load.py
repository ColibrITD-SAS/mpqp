from __future__ import annotations

from typing import Optional

from mpqp.all import *
from mpqp.local_storage.queries import *
from mpqp.local_storage.setup import DictDB


def jobs_db_to_mpqp(jobs: Optional[list[DictDB] | DictDB]) -> list[Job]:
    """Convert a dictionary or list of dictionaries representing jobs into MPQP Job objects.

    Args:
        jobs: A dictionary or list of dictionaries retrieved from the database.

    Returns:
        A list of MPQP Job objects.

    Example:
        >>> job_db = fetch_jobs_with_id(1)
        >>> jobs_db_to_mpqp(job_db)
        [Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2)))]

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


def results_db_to_mpqp(results: Optional[list[DictDB] | DictDB]) -> list[Result]:
    """Convert a dictionary or list of dictionaries representing results into a
    :class:`~mpqp.execution.result.Result`.

    Args:
        results: The results retrieved from the database.

    Returns:
        The converted result(s).

    Example:
        >>> result_db = fetch_results_with_id([1, 2])
        >>> results = results_db_to_mpqp(result_db)
        >>> for result in results:  # doctest: +ELLIPSIS
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(2, index=0, count=532, probability=0.51953125), ...], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...), [Sample(2, index=0, count=489, probability=0.4775390625), ...], None, 1024)

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
                jobs_db_to_mpqp(job)[0],
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
                    jobs_db_to_mpqp(job)[0],
                    eval(eval(result['data'])),
                    error,
                    result['shots'],
                )
            )

    return results_mpqp


def get_all_jobs() -> list[Job]:
    """Retrieve all jobs from the local storage and convert them into
    :class:`mpqp.execution.job.Job`

    Returns:
        All locally stored jobs.

    Example:
        >>> jobs = get_all_jobs()
        >>> for job in jobs:
        ...     print(job)
        Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2)))
        Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2)))
        Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1))], nb_qubits=1, nb_cbits=1, label="H BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1)))
        Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1))], nb_qubits=1, nb_cbits=1, label="H BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1)))
        Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)
        Job(JobType.STATE_VECTOR, QCircuit([Id(0), Id(1)], nb_qubits=2, label="Id"), IBMDevice.AER_SIMULATOR)

    """
    jobs = fetch_all_jobs()
    return jobs_db_to_mpqp(jobs)


def get_all_results() -> list[Result]:
    """Retrieve all results from the local storage and convert them into
    :class:`mpqp.execution.result.Result`

    Returns:
        All locally stored results.

    Example:
        >>> results = get_all_results()
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))), [Sample(2, index=0, count=532, probability=0.51953125), Sample(2, index=3, count=492, probability=0.48046875)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))), [Sample(2, index=0, count=489, probability=0.4775390625), Sample(2, index=3, count=535, probability=0.5224609375)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))), [Sample(2, index=0, count=507, probability=0.4951171875), Sample(2, index=3, count=517, probability=0.5048828125)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1))], nb_qubits=1, nb_cbits=1, label="H BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1))), [Sample(1, index=0, count=502, probability=0.490234375), Sample(1, index=1, count=522, probability=0.509765625)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1))], nb_qubits=1, nb_cbits=1, label="H BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1))), [Sample(1, index=0, count=533, probability=0.5205078125), Sample(1, index=1, count=491, probability=0.4794921875)], None, 1024)
        Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        Result(Job(JobType.STATE_VECTOR, QCircuit([Id(0), Id(1)], nb_qubits=2, label="Id"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)

    """
    results = fetch_all_results()
    return results_db_to_mpqp(results)


def get_jobs_with_job(job: Job | list[Job]) -> list[Job]:
    """Retrieve jobs matching the given job(s) from the database.

    Args:
        job: Job(s) to search for.

    Returns:
        Matching job(s).

    Example:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)
        >>> print(get_jobs_with_job(job))
        [Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)]

    """
    jobs = fetch_jobs_with_job(job)
    return jobs_db_to_mpqp(jobs)


def get_jobs_with_result(result: Result | list[Result] | BatchResult) -> list[Job]:
    """Retrieve jobs associated with the given result(s) from the database.

    Args:
        result: Result(s) to find associated jobs for.

    Returns:
        Matching jobs.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> print(get_jobs_with_result(result))
        [Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)]

    """
    jobs = fetch_jobs_with_result(result)
    return jobs_db_to_mpqp(jobs)


def get_results_with_result_and_job(
    result: Result | list[Result] | BatchResult,
) -> list[Result]:
    """Retrieve results with matching result and job data.

    Args:
        result: Result(s) to search for.

    Returns:
        Matching result(s).

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> results = get_results_with_result_and_job(result)
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)

    """
    results = fetch_results_with_result_and_job(result)
    return results_db_to_mpqp(results)


def get_results_with_result(
    result: Result | list[Result] | BatchResult,
) -> list[Result]:
    """Retrieve results matching the given result(s) from the database.

    Args:
        result: Result(s) to search for.

    Returns:
        Matching results.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> results = get_results_with_result(result)
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        Result(Job(JobType.STATE_VECTOR, QCircuit([Id(0), Id(1)], nb_qubits=2, label="Id"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)

    """
    results = fetch_results_with_result(result)
    return results_db_to_mpqp(results)


def get_results_with_id(result_id: int | list[int]) -> list[Result]:
    """Retrieve results with the given IDs.

    Args:
        ID(s) to search for.

    Returns:
        Matching result(s).

    Example:
        >>> results1 = get_results_with_id(1)
        >>> for result in results1:
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))), [Sample(2, index=0, count=532, probability=0.51953125), Sample(2, index=3, count=492, probability=0.48046875)], None, 1024)
        >>> results2 = get_results_with_id([2, 3])
        >>> for result in results2:
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))), [Sample(2, index=0, count=489, probability=0.4775390625), Sample(2, index=3, count=535, probability=0.5224609375)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))), [Sample(2, index=0, count=507, probability=0.4951171875), Sample(2, index=3, count=517, probability=0.5048828125)], None, 1024)

    """
    results = fetch_results_with_id(result_id)
    return results_db_to_mpqp(results)


def get_jobs_with_id(job_id: int | list[int]) -> list[Job]:
    """Retrieve jobs with the given IDs.

    Args:
        ID(s) to search for.

    Returns:
        Matching jobs.

    Example:
        >>> jobs = get_jobs_with_id([1, 2, 3])
        >>> for job in jobs:
        ...     print(job)
        Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2)))
        Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2)))
        Job(JobType.SAMPLE, QCircuit([H(0), BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1))], nb_qubits=1, nb_cbits=1, label="H BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0], c_targets=[0], basis=ComputationalBasis(1)))

    """
    jobs = fetch_jobs_with_id(job_id)
    return jobs_db_to_mpqp(jobs)


def get_results_with_job_id(job_id: int | list[int]) -> list[Result]:
    """Retrieve results associated with the given job ID(s).

    Args:
        ID(s) to search for.

    Returns:
        Results attached to the matching jobs.

    Example:
        >>> results = get_results_with_job_id(1)
        >>> for result in results:
        ...     print(repr(result))
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))), [Sample(2, index=0, count=532, probability=0.51953125), Sample(2, index=3, count=492, probability=0.48046875)], None, 1024)
        Result(Job(JobType.SAMPLE, QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))], nb_qubits=2, nb_cbits=2, label="H CX BM"), IBMDevice.AER_SIMULATOR, BasisMeasure([0, 1], c_targets=[0, 1], basis=ComputationalBasis(2))), [Sample(2, index=0, count=489, probability=0.4775390625), Sample(2, index=3, count=535, probability=0.5224609375)], None, 1024)
    """
    results = fetch_results_with_job_id(job_id)
    return results_db_to_mpqp(results)
