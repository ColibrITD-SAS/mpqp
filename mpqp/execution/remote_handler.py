"""After the jobs are submitted, one can use the functions of this module to
retrieve the results from a job_id or the job directly, and list all job
attached to the configured accounts."""

from __future__ import annotations

from typeguard import typechecked

from mpqp.execution import Result
from mpqp.execution.connection.aws_connection import get_all_task_ids
from mpqp.execution.connection.qlm_connection import get_all_job_ids as qlm_ids
from mpqp.execution.connection.ibm_connection import get_all_job_ids as ibm_ids
from mpqp.execution.providers_execution.atos_execution import get_result_from_qlm_job_id
from mpqp.execution.providers_execution.aws_execution import (
    get_result_from_aws_task_arn,
)
from mpqp.execution.providers_execution.ibm_execution import get_result_from_ibm_job_id

from mpqp.execution.devices import IBMDevice, AWSDevice, ATOSDevice, AvailableDevice
from mpqp.execution.job import Job


@typechecked
def remote_result_from_id(job_id: str, device: AvailableDevice) -> Result:
    """
    Retrieve and parse a remote the result from a job_id and device. If the job is still running, it will wait until it
    is done.

    Examples:
        >>> print(remote_result_from_id('Job141933', ATOSDevice.QLM_LINALG))
        Result: ATOSDevice, QLM_LINALG
        Counts: [1017, 0, 0, 0, 983, 0, 0, 0]
        Probabilities: [0.5085 0.     0.     0.     0.4915 0.     0.     0.    ]
        State: 000, Index: 0, Count: 1017, Probability: 0.5085
        State: 100, Index: 4, Count: 983, Probability: 0.4915
        Error: 0.011181519941139355
        >>> print(remote_result_from_id(
        ...     'cm80pb1054sir2ck9i3g',
        ...     IBMDevice.IBMQ_QASM_SIMULATOR,
        ... ))
        Result: IBMDevice, IBMQ_QASM_SIMULATOR
        Expectation value: 1.6410799999999999
        Error/Variance: 1.24570724535
        >>> aws_task_id = (
        ...     'arn:aws:braket:us-east-1:752542621531:quantum-task/'
        ...     '6a46ae9a-d02f-4a23-b46f-eae43471bc22'
        ... )
        >>> print(remote_result_from_id(
        ...     aws_task_id,
        ...     AWSDevice.BRAKET_SV1_SIMULATOR,
        ... ))
        Result: AWSDevice, BRAKET_SV1_SIMULATOR
        Expectation value: 1.6635202030411578
        Error/Variance: None

    Args:
        job_id: Id used to identify the job on the remote device.
        device: Remote device on which the job was launched.

    Returns:
        A Result of the remote job identified by the job_id in parameter.
    """

    # if the device is not a remote one
    if not device.is_remote():
        raise ValueError(
            "Trying to retrieve a remote result while the device of the job was local."
        )

    # check the type of the device, and call the right job getter
    if isinstance(device, IBMDevice):
        return get_result_from_ibm_job_id(job_id)
    elif isinstance(device, ATOSDevice):
        return get_result_from_qlm_job_id(job_id)
    elif isinstance(device, AWSDevice):
        return get_result_from_aws_task_arn(job_id)
    else:
        raise NotImplementedError(
            f"The device {device.name} is not supported for remote features."
        )


@typechecked
def remote_result_from_job(job: Job) -> Result:
    """Retrieve and parse a remote the result from an mpqp job. If the job is
    still running, it will wait until it is done.

    Args:
        job: MPQP job for which we want to retrieve the result from the remote
            device.

    Returns:
        A Result of the remote job in parameter.
    """
    if job.id is None:
        raise ValueError("Can't retrieve remote result for a job whose id is None.")
    return remote_result_from_id(job.id, job.device)


def get_all_job_ids() -> dict[type[AvailableDevice], list[str]]:
    """Retrieve from the remote providers all the job-ids associated with this
    account.

    Returns:
        A dictionary of job-ids indexed by the correspond AvailableDevice
        (ATOSDevice, AWSDevice, IBMDevice, ...).
    """
    job_ids: dict[type[AvailableDevice], list[str]] = {
        AWSDevice: get_all_task_ids(),
        ATOSDevice: qlm_ids(),
        IBMDevice: ibm_ids(),
    }

    return job_ids
