from __future__ import annotations

from typing import Any, Optional
from mpqp.all import *
from mpqp.db.db_query import *
from mpqp.execution.runner import generate_job


def jobs_dict_to_mpqp(
    jobs: Optional[list[dict[Any, Any]] | dict[Any, Any]]
) -> list[Job]:
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


def results_dict_to_mpqp(
    results: Optional[list[dict[Any, Any]] | dict[Any, Any]]
) -> list[Result]:
    if results is None:
        return []
    results_mpqp = []
    if isinstance(results, dict):
        error = None if results['error'] is None else eval(results['error'])
        job = fetch_job_with_id(results['job_id'])
        results_mpqp.append(
            Result(
                jobs_dict_to_mpqp(job)[0],
                eval(eval(results['data'])),
                error,
                results['shots'],
            )
        )
    else:
        for result in results:
            error = None if result['error'] is None else eval(result['error'])
            job = fetch_job_with_id(result['job_id'])
            results_mpqp.append(
                Result(
                    jobs_dict_to_mpqp(job)[0],
                    eval(eval(result['data'])),
                    error,
                    result['shots'],
                )
            )

    return results_mpqp


def get_all_jobs():
    jobs = fetch_all_jobs()
    return jobs_dict_to_mpqp(jobs)


def get_all_results():
    results = fetch_all_results()
    return results_dict_to_mpqp(results)


def get_jobs_with_job(job: Job):
    jobs = fetch_jobs_with_job(job)
    return jobs_dict_to_mpqp(jobs)


def get_jobs_with_result(result: Result):
    jobs = fetch_jobs_with_result(result)
    return jobs_dict_to_mpqp(jobs)


def get_results_with_result_and_job(result: Result):
    results = fetch_results_with_result_and_job(result)
    return results_dict_to_mpqp(results)


def get_results_with_result(result: Result):
    results = fetch_results_with_result(result)
    return results_dict_to_mpqp(results)


def get_result_with_id(result_id: int):
    results = fetch_result_with_id(result_id)
    return results_dict_to_mpqp(results)


def get_job_with_id(job_id: int):
    jobs = fetch_job_with_id(job_id)
    return jobs_dict_to_mpqp(jobs)
