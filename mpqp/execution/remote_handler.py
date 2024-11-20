"""After the jobs are submitted, one can use the functions of this module to
retrieve the results from a job_id or the job directly, and list all job
attached to the configured accounts."""

from __future__ import annotations

from typing import Optional

from typeguard import typechecked

from mpqp.execution import Result
from mpqp.execution.connection.aws_connection import get_all_task_ids as aws_ids
from mpqp.execution.connection.google_connection import get_all_job_ids as cirq_ids
from mpqp.execution.connection.ibm_connection import get_all_job_ids as ibm_ids
from mpqp.execution.connection.qlm_connection import get_all_job_ids as qlm_ids
from mpqp.execution.connection.google_connection import get_all_job_ids as cirq_ids
from mpqp.execution.connection.azure_connection import get_all_job_ids as azure_ids
from mpqp.execution.devices import (
    ATOSDevice,
    AvailableDevice,
    AWSDevice,
    GOOGLEDevice,
    AZUREDevice,
    IBMDevice,
)
from mpqp.execution.job import Job
from mpqp.execution.providers.atos import get_result_from_qlm_job_id
from mpqp.execution.providers.aws import get_result_from_aws_task_arn
from mpqp.execution.providers.ibm import get_result_from_ibm_job_id
from mpqp.execution.providers.azure import get_result_from_azure_job_id


@typechecked
def get_remote_result(
    job_data: str | Job, device: Optional[AvailableDevice] = None
) -> Result:
    """Retrieve and parse a remote the result from a job_id and device. If the
    job is still running, it will wait until it is done.

    Args:
        job_data: Either the :class:`~mpqp.execution.job.Job` object or the
            job id used to identify the job on the remote device.
        device: Remote device on which the job was launched, needed only if
            ``job_data`` is the identifier of the job.

    Returns:
        The ``Result`` of the desired remote job.

    Examples:
        >>> print(get_remote_result('Job141933', ATOSDevice.QLM_LINALG))
        Result: ATOSDevice, QLM_LINALG
         Counts: [1017, 0, 0, 0, 983, 0, 0, 0]
         Probabilities: [0.5085 0.     0.     0.     0.4915 0.     0.     0.    ]
          State: 000, Index: 0, Count: 1017, Probability: 0.5085
          State: 100, Index: 4, Count: 983, Probability: 0.4915
         Error: 0.011181519941139355
        >>> print(get_remote_result(
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
        >>> print(get_remote_result(
        ...     aws_task_id,
        ...     AWSDevice.BRAKET_SV1_SIMULATOR,
        ... ))
        Result: AWSDevice, BRAKET_SV1_SIMULATOR
         Expectation value: 1.6635202030411578
         Error/Variance: None

        >>> circ = QCircuit([H(0), CNOT(0,1)])
        >>> _, job = submit(circ, ATOSDevice.QLM_LINALG)
        >>> print(get_remote_result(job))
        Result: ATOSDevice, QLM_LINALG
         State vector: [0.7071068, 0, 0, 0.7071068]
         Probabilities: [0.5, 0, 0, 0.5]
         Number of qubits: 2

    """
    if isinstance(job_data, Job):
        if job_data.id is None:
            raise ValueError("Can't retrieve remote result for a job whose id is None.")

        device = job_data.device
        job_data = job_data.id
    else:
        if device is None:
            raise ValueError(
                "To get a remote result from a job it, please also provide the "
                "device to get the data from."
            )

    if not device.is_remote():
        raise ValueError(
            "Trying to retrieve a remote result while the device of the job was local."
        )

    if isinstance(device, IBMDevice):
        return get_result_from_ibm_job_id(job_data)
    elif isinstance(device, ATOSDevice):
        return get_result_from_qlm_job_id(job_data)
    elif isinstance(device, AWSDevice):
        return get_result_from_aws_task_arn(job_data)
    elif isinstance(device, AZUREDevice):
        return get_result_from_azure_job_id(job_data)
    else:
        raise NotImplementedError(
            f"The device {device.name} is not supported for remote features."
        )


def get_all_job_ids() -> dict[type[AvailableDevice], list[str]]:
    """Retrieve from the remote providers all the job-ids associated with this
    account.

    Returns:
        A dictionary of job-ids indexed by the correspond AvailableDevice
        (ATOSDevice, AWSDevice, IBMDevice, ...).
    """
    job_ids: dict[type[AvailableDevice], list[str]] = {
        AWSDevice: aws_ids(),
        ATOSDevice: qlm_ids(),
        IBMDevice: ibm_ids(),
        GOOGLEDevice: cirq_ids(),
        AZUREDevice: azure_ids(),
    }

    return job_ids
