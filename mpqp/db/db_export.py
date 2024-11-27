from __future__ import annotations

from typing import Any, Optional
from mpqp.all import *
from mpqp.db.db_query import fetch_job_with_id
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


def results_dict_to_mpqp(results: Optional[list[dict[Any, Any]] | dict[Any, Any]]):
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
