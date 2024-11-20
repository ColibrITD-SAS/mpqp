"""For now, Google's access is very restricted, so this section is mostly here
for future proofing."""


def get_all_job_ids() -> list[str]:
    """Retrieves all job IDs associated with google jobs.

    Note:
        For now, no provider is exclusively attached to ``cirq``, so this will
        not return any job ID. For job IDs relative to ionQ, use
        :func:`~mpqp.execution.connection.ionq_connection.get_ionq_job_ids`.

    Returns:
        A list of job IDs.
    """
    return []
