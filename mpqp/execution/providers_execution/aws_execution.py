import math
from typing import Optional
from typeguard import typechecked

import numpy as np
from braket.circuits.observables import Hermitian
from braket.tasks import GateModelQuantumTaskResult, QuantumTask
from braket.aws import AwsQuantumTask

from mpqp import QCircuit, Language
from mpqp.core.instruction.measurement import ExpectationMeasure, BasisMeasure, Observable
from mpqp.execution.connection.aws_connection import get_braket_device
from mpqp.execution.devices import AWSDevice
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.tools.errors import AWSBraketRemoteExecutionError
from mpqp.execution.job import Job, JobType, JobStatus
from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit
from mpqp.tools.errors import DeviceJobIncompatibleError


@typechecked
def run_braket(job: Job) -> Result:
    """
    Executes the job on the right AWS Braket device (local or remote) precised in the job in parameter and waits until
    the task is completed, then returns the Result.
    This function is not meant to be used directly, please use ``runner.run(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    """
    _, task = submit_job_braket(job)
    assert isinstance(job.device, AWSDevice)
    return extract_result(task.result(), job, job.device)


@typechecked
def submit_job_braket(job: Job) -> tuple[str, QuantumTask]:
    """
    Submits the job to the right local/remote device and returns the generated task.
    This function is not meant to be used directly, please use ``runner.submit(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        A string representing the task' id, and the Task itself.
    """

    # check some compatibility issues

    if job.job_type == JobType.STATE_VECTOR and job.device.is_remote():
        raise DeviceJobIncompatibleError("State vector cannot be computed using AWS Braket remote simulators and "
                                         "devices. Please use the LocalSimulator instead")

    # instantiate the device
    device = get_braket_device(job.device)  # type: ignore

    # convert job circuit into braket circuit
    brkt_circuit = job.circuit.to_other_language(Language.BRAKET)

    if job.job_type == JobType.STATE_VECTOR:
        brkt_circuit.state_vector()  # type: ignore
        job.status = JobStatus.RUNNING
        task = device.run(brkt_circuit, shots=0)

    elif job.job_type == JobType.SAMPLE:
        job.status = JobStatus.RUNNING
        task = device.run(brkt_circuit, shots=job.measure.shots)

    elif job.job_type == JobType.OBSERVABLE:
        if not isinstance(job.measure, ExpectationMeasure):
            raise ValueError(
                "Cannot compute expectation value if measure used in job is not of "
                "type ExpectationMeasure"
            )

        herm_op = Hermitian(job.measure.observable.matrix)
        brkt_circuit.expectation(observable=herm_op, target=job.measure.targets)  # type: ignore

        job.status = JobStatus.RUNNING
        task = device.run(brkt_circuit, shots=job.measure.shots)

    else:
        raise NotImplementedError(f"Job of type {job.job_type} not handled.")

    return task.id, task


@typechecked
def extract_result(braket_result: GateModelQuantumTaskResult, job: Optional[Job] = None,
                   device: Optional[AWSDevice] = AWSDevice.BRAKET_LOCAL_SIMULATOR) -> Result:
    """
    Constructs a Result from the result given by the run with Braket.

    Args:
        braket_result: Result returned by myQLM/QLM after running of the job.
        job: Original mpqp job used to generate the run. Used to retrieve more easily info to instantiate the result.
        device: AWSDevice on which the job was submitted.

    Returns:
        A Result containing the result info extracted from the Braket result.
    """
    if job is None:
        if len(braket_result.values) == 0:
            job_type = JobType.SAMPLE
            nb_qubits = len(list(braket_result.measurement_counts.keys())[0])
            shots = braket_result.task_metadata.shots
            measure = BasisMeasure(list(range(nb_qubits)), shots=shots)
        elif isinstance(braket_result.values[0], float):
            job_type = JobType.OBSERVABLE
            nb_qubits = braket_result.task_metadata.deviceParameters.paradigmParameters.qubitCount
            shots = braket_result.task_metadata.shots
            measure = ExpectationMeasure(list(range(nb_qubits)),
                                         Observable(np.zeros((2 ** nb_qubits, 2 ** nb_qubits))),
                                         shots)
        else:
            job_type = JobType.STATE_VECTOR
            nb_qubits = int(math.log2(len(braket_result.values[0])))
            measure = BasisMeasure(list(range(nb_qubits)), shots=0)
        job = Job(job_type, QCircuit(nb_qubits), device, measure)
    job.status = JobStatus.DONE

    if job.job_type == JobType.STATE_VECTOR:
        vector = braket_result.values[0]
        state_vector = StateVector(vector, nb_qubits=job.circuit.nb_qubits)
        return Result(job, state_vector, 0, 0)

    elif job.job_type == JobType.SAMPLE:
        counts = braket_result.measurement_counts
        sample_info = []
        for state in counts.keys():
            sample_info.append(
                Sample(job.circuit.nb_qubits, count=counts[state], bin_str=state)
            )
        return Result(job, sample_info, None, job.measure.shots)

    elif job.job_type == JobType.OBSERVABLE:
        exp_value = braket_result.values[0]
        return Result(job, exp_value, None, job.measure.shots)

    else:
        raise NotImplementedError(f"Job of type {job.job_type} not handled.")


@typechecked
def get_result_from_aws_task_arn(task_arn: str = None) -> Result:
    """
    Retrieves the result, described by the job_id in parameter, from the remote QLM and converts it into an mpqp result.
    If the job is still running, we wait (blocking) until it is DONE.

    Args:
        task_arn: Arn of the remote aws task.

    """
    task: QuantumTask = AwsQuantumTask(task_arn)
    # eventually catch an error if the id is not correct (wrong ID, wrong region/blablabla)

    status: str = task.state()  # get the status of the task, it is a string
    # depending on the status, either raise an error, wait for the task to finish, or get the result if done

    if status in ["FAILED", "CANCELLED"]:
        raise AWSBraketRemoteExecutionError("")
    elif status in ["CREATED", "QUEUED", "RUNNING", "COMPLETED"]:  #
        result = task.result()  # will get the result, and eventually wait for it
    else:
        raise AWSBraketRemoteExecutionError(
            f"Unknown status {status} for the task {task_arn}"
        )

    device_arn = task.metadata()["deviceArn"]
    device = AWSDevice.from_arn(device_arn)

    return extract_result(result, None, device)
