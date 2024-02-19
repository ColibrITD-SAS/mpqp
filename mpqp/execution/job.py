from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from aenum import Enum, NoAlias

# This is needed because for some reason pyright does not understand that Enum
# is a class (probably because Enum does weird things to the Enum class)
from typeguard import typechecked

if TYPE_CHECKING:
    from enum import Enum

from mpqp.core.instruction.measurement import Measure, ExpectationMeasure, BasisMeasure
from .connection.ibm_connection import get_IBMProvider, get_QiskitRuntimeService
from .connection.qlm_connection import get_QLMaaSConnection
from ..core.circuit import QCircuit
from .devices import AvailableDevice, IBMDevice, AWSDevice, ATOSDevice

from qat.comm.qlmaas.ttypes import JobStatus as QLM_JobStatus, QLMServiceException
from qiskit.providers import JobStatus as IBM_JobStatus
from braket.aws import AwsQuantumTask

from ..tools.errors import IBMRemoteExecutionError, QLMRemoteExecutionError


class JobStatus(Enum):
    """Possible states of a Job."""

    INIT = "initializing the job"
    """Initializing the job."""
    QUEUED = "the job is in the queue"
    """The job is in the queue."""
    RUNNING = "the job is currently running"
    """The job is currently running."""
    CANCELLED = "the job is cancelled"
    """The job is cancelled."""
    ERROR = "an error occurred with the job"
    """An error occurred with the job."""
    DONE = "the job is successfully done"
    """The job is successfully done."""


class JobType(Enum):
    """
    Possible types of Job to execute.

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
    """
    Representation of a job, an object regrouping all the information about
    the submission of a computation/measure of a quantum circuit on a
    specific hardware.

    A job has a type, and a status, and is attached to a specific device.
    Moreover, the job contains also the quantum circuit and the measure to be
    performed on the circuit.

    Args:
        job_type: Type of the job (sample, observable, ...).
        circuit: Circuit to execute.
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
        ...     circuit.get_measurements()[0],
        ... )
    """

    # 6M-TODO: decide, when there are several measurements, if we define a
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
        self.measure = measure
        """See parameter description."""

        self.id: Optional[str] = None
        """Contains the id of the remote job, used to retrieve the result from 
        the remote provider.  ``None`` if the job is local. If the job is not 
        local, it will be set later on."""

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
                assert isinstance(self.id, str)
                if isinstance(self.device, ATOSDevice):
                    self._status = get_qlm_job_status(self.id)

                elif isinstance(self.device, IBMDevice):
                    self._status = get_ibm_job_status(self.id)

                elif isinstance(self.device, AWSDevice):
                    self._status = get_aws_job_status(self.id)

                else:
                    raise NotImplementedError(
                        f"Cannot update job status for the device {self.device} yet"
                    )
        return self._status

    @status.setter
    def status(self, job_status: JobStatus):
        self._status = job_status


def get_qlm_job_status(job_id: str) -> JobStatus:
    """Retrieves the status of a QLM job from the id in parameter, and returns
    the corresponding JobStatus of this library.

    Args:
        job_id: Id of the job for which we want to retrieve the status.
    """
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


def get_ibm_job_status(job_id: str) -> JobStatus:
    """Retrieves the status of an IBM job from the id in parameter, and returns
    the corresponding JobStatus of this library.

    Args:
        job_id: Id of the job for which we want to retrieve the status.
    """
    # test with QiskitRuntimeService
    if job_id in [e.job_id() for e in get_QiskitRuntimeService().jobs()]:
        ibm_job = get_QiskitRuntimeService().job(job_id)

    # if not, test with IBMProvider
    elif job_id in [e.job_id() for e in get_IBMProvider().jobs()]:
        ibm_job = get_IBMProvider().retrieve_job(job_id)

    else:
        raise IBMRemoteExecutionError(
            f"Could not find job with id {job_id} on IBM/QiskitRuntime provider"
        )
    status = ibm_job.status()
    if status == IBM_JobStatus.ERROR:
        return JobStatus.ERROR
    elif status == IBM_JobStatus.CANCELLED:
        return JobStatus.CANCELLED
    elif status in [IBM_JobStatus.QUEUED, IBM_JobStatus.VALIDATING]:
        return JobStatus.QUEUED
    elif status == IBM_JobStatus.INITIALIZING:
        return JobStatus.INIT
    elif status == IBM_JobStatus.RUNNING:
        return JobStatus.RUNNING
    else:
        return JobStatus.DONE


def get_aws_job_status(job_id: str) -> JobStatus:
    """Retrieves the status of a AWS Braket from the id in parameter, and
    returns the corresponding JobStatus of this library.

    Args:
        job_id: Id of the job for which we want to retrieve the status.
    """
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
