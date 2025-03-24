"""When you call :func:`~mpqp.execution.runner.run` or
:func:`~mpqp.execution.runner.submit`, a :class:`Job` is created by
:func:`~mpqp.execution.runner.generate_job`. This job contains all
the needed information to configure the execution, and eventually retrieve
remote results.

A :class:`Job` can be of three types, given by the :class:`JobType` enum. In
addition, it has a status, given by the :class:`JobStatus` enum.

As described above, a :class:`Job` is generated on circuit submission so you
would in principle never need to instantiate one yourself.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from aenum import Enum, NoAlias, auto
from typeguard import typechecked

from mpqp.tools.generics import MessageEnum

# This is needed because for some reason pyright does not understand that Enum
# is a class (probably because Enum does weird things to the Enum class)
if TYPE_CHECKING:
    from enum import Enum

from mpqp.core.instruction.measurement import BasisMeasure, ExpectationMeasure, Measure

from ..core.circuit import QCircuit
from ..tools.errors import IBMRemoteExecutionError, QLMRemoteExecutionError
from .connection.azure_connection import get_jobs_by_id
from .connection.ibm_connection import get_QiskitRuntimeService
from .connection.qlm_connection import get_QLMaaSConnection
from .devices import ATOSDevice, AvailableDevice, AWSDevice, AZUREDevice, IBMDevice


class JobStatus(MessageEnum):
    """Possible states of a Job."""

    INIT = auto()
    """Initializing the job."""
    QUEUED = auto()
    """The job is in the queue."""
    RUNNING = auto()
    """The job is currently running."""
    CANCELLED = auto()
    """The job is cancelled."""
    ERROR = auto()
    """An error occurred with the job."""
    DONE = auto()
    """The job is successfully done."""


class JobType(Enum):
    """Possible types of Job to execute.

    Each type of job is restricted to some measures (and to some backends, but
    this is tackled by the backends themselves).
    """

    _settings_ = NoAlias
    STATE_VECTOR = {BasisMeasure, type(None)}
    """Retrieves the vector representing the quantum state, this type is *ideal*."""
    SAMPLE = {BasisMeasure}
    """Measures several times the quantum state in the basis, and retrieve the 
    counts. Contrarily to the ``STATE_VECTOR`` job type, this one is *realistic*."""
    OBSERVABLE = {ExpectationMeasure}
    """Computes the *expectation value* of an observable, using the state_vector 
    or the samples. This type is ideal too: it requires some trickery to 
    retrieve the expectation value in an optimal manner."""


@typechecked
class Job:
    """Representation of a job, an object regrouping all the information about
    the submission of a computation/measure of a quantum circuit on a
    specific hardware.

    A job has a type, and a status, and is attached to a specific device.
    Moreover, the job contains also the quantum circuit and the measure to be
    performed on the circuit.

    Args:
        job_type: Type of the job (sample, observable, ...).
        circuit: Circuit to execute. In addition of what the user input, this
            circuit may contain additional parts such as measure adjustment
            sections.
        device: Device (simulator, quantum computer) on which we want to execute
            the job.
        measure: Object representing the measure to perform.

    Examples:
        >>> circuit = QCircuit(3)
        >>> job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)

        >>> circuit.add(BasisMeasure([0, 1], shots=1000))
        >>> job2 = Job(
        ...     JobType.STATE_VECTOR,
        ...     circuit,
        ...     IBMDevice.AER_SIMULATOR,
        ...     circuit.measurements[0],
        ... )

    """

    # 3M-TODO: decide, when there are several measurements, if we define a
    #  multi-measure job, or if we need several jobs. For the moment, a Job can
    #  handle only one measurement

    def __init__(
        self,
        job_type: JobType,
        circuit: QCircuit,
        device: AvailableDevice,
        measure: Optional[Measure] = None,
    ):
        self._status = JobStatus.INIT

        self.job_type = job_type
        """See parameter description."""
        self.circuit = circuit
        """See parameter description."""
        self.device = device
        """See parameter description."""
        self.measure = deepcopy(measure)
        """See parameter description."""
        if self.measure is not None:
            self.measure._dynamic = False  # pyright: ignore[reportPrivateUsage]
            if isinstance(self.measure, BasisMeasure):
                self.measure._user_set_c_targets = (  # pyright: ignore[reportPrivateUsage]
                    True
                )
        self.id: Optional[str] = None
        """Contains the id of the remote job, used to retrieve the result from 
        the remote provider.  ``None`` if the job is local. It can take a little
        while before it is set to the right value (For instance, a job
        submission can require handshake protocols to conclude before
        attributing an id to the job)."""

    @property
    def status(self):
        """Update and return the current job status. Mainly relevant for remote jobs."""
        if self._status not in [
            JobStatus.DONE,
            JobStatus.ERROR,
            JobStatus.CANCELLED,
        ]:
            # in the remote case, we need to check the current status of the job.
            # in the local case, it is updated automatically after each step
            if self.device.is_remote():
                if TYPE_CHECKING:
                    assert isinstance(self.id, str)
                if isinstance(self.device, ATOSDevice):
                    self._status = get_qlm_job_status(self.id)
                elif isinstance(self.device, IBMDevice):
                    self._status = get_ibm_job_status(self.id)
                elif isinstance(self.device, AWSDevice):
                    self._status = get_aws_job_status(self.id)
                elif isinstance(self.device, AZUREDevice):
                    self._status = get_azure_job_status(self.id)
                else:
                    raise NotImplementedError(
                        f"Cannot update job status for the device {self.device} yet"
                    )
        return self._status

    @status.setter
    def status(self, job_status: JobStatus):
        self._status = job_status

    def __repr__(self) -> str:
        measure = ", " + repr(self.measure) if self.measure is not None else ""
        return f"{type(self).__name__}({self.job_type}, {repr(self.circuit)}, {self.device}{measure})"

    def __eq__(self, other):  # pyright: ignore[reportMissingParameterType]
        if not isinstance(other, Job):
            return False
        return (
            self.job_type == other.job_type
            and self.circuit == other.circuit
            and self.device == other.device
            and self.measure == other.measure
        )

    def to_dict(self):
        return {
            "job_type": self.job_type,
            "circuit": self.circuit,
            "device": self.device,
            "measure": self.measure,
            "id": self.id,
            "status": self.status,
        }

    @staticmethod
    def load_all():
        """Get all locally stored jobs.

        Uses :func:`~mpqp.local_storage.load.get_all_jobs`.

        Example:
            >>> for job in Job.load_all(): # doctest: +ELLIPSIS
            ...     print(job)
            Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
            Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
            Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
            Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
            Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)
            Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)

        """
        from mpqp.local_storage.load import get_all_jobs

        return get_all_jobs()

    @staticmethod
    def load_by_local_id(job_id: int):
        """Get the locally stored job associated with a local job id.

        Uses :func:`~mpqp.local_storage.load.get_jobs_with_id`.

        Args:
            job_id: Local id of the job you need.

        Example:
            >>> Job.load_by_local_id(1) # doctest: +ELLIPSIS
            Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
        """
        from mpqp.local_storage.load import get_jobs_with_id

        return get_jobs_with_id(job_id)[0]

    def load_similar(self):
        """Get the jobs similar to the target job.

        Uses :func:`~mpqp.local_storage.load.get_jobs_with_job`.

        Example:
            >>> job = Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)
            >>> print(job.load_similar()) # doctest: +ELLIPSIS
            [Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)]
        """
        from mpqp.local_storage.load import get_jobs_with_job

        return get_jobs_with_job(self)

    def save(self):
        """Saves a job and return the local id.

        Uses :func:`~mpqp.local_storage.save.insert_jobs`."""
        from mpqp.local_storage.save import insert_jobs

        return insert_jobs(self)[0]

    @staticmethod
    def delete_by_local_id(job_id: int):
        """Delete the locally stored job associated with a local job id.

        Uses :func:`~mpqp.local_storage.delete.remove_jobs_with_id`.

        Args:
            job_id: Local id of the job you want to delete.

        Example:
            >>> for job in Job.load_all(): # doctest: +ELLIPSIS
            ...     print(job)
            Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
            Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
            Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
            Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
            Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)
            Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)
            >>> Job.delete_by_local_id(1)
            >>> for job in Job.load_all(): # doctest: +ELLIPSIS
            ...     print(job)
            Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
            Job(JobType.SAMPLE, QCircuit(...), IBMDevice.AER_SIMULATOR, BasisMeasure(...))
            Job(JobType.SAMPLE, QCircuit(...), GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, BasisMeasure(...))
            Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)
            Job(JobType.STATE_VECTOR, QCircuit(...), IBMDevice.AER_SIMULATOR)

        """
        from mpqp.local_storage.delete import remove_jobs_with_id

        remove_jobs_with_id(job_id)

    def delete_associated_results(self):
        """Delete the results associated with the target job.

        Uses :func:`~mpqp.local_storage.delete.remove_results_with_job`."""
        from mpqp.local_storage.delete import remove_results_with_job

        remove_results_with_job(self)


@typechecked
def get_qlm_job_status(job_id: str) -> JobStatus:
    """Retrieves the status of a QLM job from the id in parameter, and returns
    the corresponding JobStatus of this library.

    Args:
        job_id: Id of the job for which we want to retrieve the status.
    """
    from qat.comm.qlmaas.ttypes import JobStatus as QLM_JobStatus
    from qat.comm.qlmaas.ttypes import QLMServiceException

    try:
        qlm_status = get_QLMaaSConnection().get_status(job_id)
    except QLMServiceException as e:
        raise QLMRemoteExecutionError(
            f"Something went wrong when trying to get the state of job {job_id}."
            f" Please make sure the job_id is correct.\n Trace: " + str(e)
        )

    if qlm_status in [QLM_JobStatus.WAITING, QLM_JobStatus.STOPPED]:
        return JobStatus.QUEUED
    elif qlm_status == QLM_JobStatus.RUNNING:
        return JobStatus.RUNNING
    elif qlm_status == QLM_JobStatus.CANCELLED:
        return JobStatus.CANCELLED
    elif qlm_status in [
        QLM_JobStatus.FAILED,
        QLM_JobStatus.UNKNOWN_JOB,
        QLM_JobStatus.IN_BUCKET,
        QLM_JobStatus.DELETED,
    ]:
        return JobStatus.ERROR
    else:
        return JobStatus.DONE


@typechecked
def get_ibm_job_status(job_id: str) -> JobStatus:
    """Retrieves the status of an IBM job from the id in parameter, and returns
    the corresponding JobStatus of this library.

    Args:
        job_id: Id of the job for which we want to retrieve the status.
    """
    if job_id in [e.job_id() for e in get_QiskitRuntimeService().jobs()]:
        ibm_job = get_QiskitRuntimeService().job(job_id)
    else:
        raise IBMRemoteExecutionError(
            f"Could not find job with id {job_id} on QiskitRuntime service."
        )

    status = ibm_job.status()
    if status == "ERROR":
        return JobStatus.ERROR
    elif status == "CANCELLED":
        return JobStatus.CANCELLED
    elif status == "QUEUED":
        return JobStatus.QUEUED
    elif status == "INITIALIZING":
        return JobStatus.INIT
    elif status == "RUNNING":
        return JobStatus.RUNNING
    elif status == "DONE":
        return JobStatus.DONE
    else:
        raise ValueError(f"Unexpected IBM job status: {status}")


@typechecked
def get_aws_job_status(job_id: str) -> JobStatus:
    """Retrieves the status of a AWS Braket from the id in parameter, and
    returns the corresponding JobStatus of this library.

    Args:
        job_id: Id of the job for which we want to retrieve the status.
    """
    from braket.aws import AwsQuantumTask

    task = AwsQuantumTask(job_id)
    state = task.state()
    if state == "FAILED":
        return JobStatus.ERROR
    elif state == "CANCELLED":
        return JobStatus.CANCELLED
    elif state == "CREATED":
        return JobStatus.INIT
    elif state == "QUEUED":
        return JobStatus.QUEUED
    elif state == "RUNNING":
        return JobStatus.RUNNING
    else:
        return JobStatus.DONE


@typechecked
def get_azure_job_status(job_id: str) -> JobStatus:
    """Retrieves the status of a azure from the id in parameter, and
    returns the corresponding JobStatus of this library.

    Args:
        job_id: Id of the job for which we want to retrieve the status.
    """

    job = get_jobs_by_id(job_id)

    status = job.details.status
    if status is None:
        raise ValueError(f"Unexpected azure job status: {status}")
    if status == "failed":
        return JobStatus.ERROR
    elif status == "cancelled":
        return JobStatus.CANCELLED
    elif status == "succeeded":
        return JobStatus.DONE
    elif status == "waiting":
        return JobStatus.QUEUED
    elif status == "executing":
        return JobStatus.RUNNING
    else:
        raise ValueError(f"Unexpected azure job status: {status}")
