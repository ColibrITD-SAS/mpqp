import math
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from braket.circuits import Circuit
    from braket.tasks import GateModelQuantumTaskResult, QuantumTask

from typeguard import typechecked

from mpqp import Language, QCircuit
from mpqp.core.instruction.gates import CRk
from mpqp.core.instruction.measurement import (
    BasisMeasure,
    ExpectationMeasure,
    Observable,
)
from mpqp.execution.connection.aws_connection import get_braket_device
from mpqp.execution.devices import AWSDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.noise.noise_model import NoiseModel
from mpqp.tools.errors import AWSBraketRemoteExecutionError, DeviceJobIncompatibleError


@typechecked
def apply_noise_to_braket_circuit(
    braket_circuit: "Circuit",
    noises: list[NoiseModel],
    nb_qubits: int,
) -> "Circuit":
    """Apply noise models to a Braket circuit.

    This function applies noise models to a given Braket circuit based on the specified noise models and
    the number of qubits in the circuit. It modifies the original circuit by adding noise
    instructions and returns a new circuit with the noise applied.

    Args:
        braket_circuit: The Braket circuit to apply noise to.
        noises: A list of noise models to apply to the circuit.
        nb_qubits: The number of qubits in the circuit.

    Returns:
        A new circuit with the noise applied.
    """
    from braket.circuits import Circuit, Noise
    from braket.circuits.measure import Measure

    stored_measurements = []
    other_instructions = []

    for instr in braket_circuit.instructions:
        if isinstance(instr.operator, Measure):
            stored_measurements.append(instr)
        else:
            other_instructions.append(instr)

    noisy_circuit = Circuit(other_instructions)

    for noise in noises:
        braket_noise = noise.to_other_language(Language.BRAKET)
        assert isinstance(braket_noise, Noise)
        if CRk in noise.gates:
            raise NotImplementedError(
                "Cannot simulate noisy circuit with CRk gate due to an error on"
                " AWS Braket side."
            )
        noisy_circuit.apply_gate_noise(
            braket_noise,  # pyright: ignore[reportArgumentType]
            target_gates=(
                [
                    gate.braket_gate  # pyright: ignore[reportAttributeAccessIssue]
                    for gate in noise.gates
                    if hasattr(gate, "braket_gate")
                ]
                if len(noise.gates) != 0
                else None
            ),
            target_qubits=(
                noise.targets if set(noise.targets) != set(range(nb_qubits)) else None
            ),
        )

    return noisy_circuit


@typechecked
def run_braket(job: Job) -> Result:
    """Executes the job on the right AWS Braket device (local or remote)
    precised in the job in parameter and waits until the task is completed, then
    returns the Result.

    Args:
        job: Job to be executed.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    from braket.tasks import GateModelQuantumTaskResult

    _, task = submit_job_braket(job)
    assert isinstance(job.device, AWSDevice)
    res = task.result()
    assert isinstance(res, GateModelQuantumTaskResult)

    return extract_result(res, job, job.device)


@typechecked
def submit_job_braket(job: Job) -> tuple[str, "QuantumTask"]:
    """Submits the job to the right local/remote device and returns the
    generated task.

    Args:
        job: Job to be executed.

    Returns:
        The task's id and the Task itself.

    Raises:
        ValueError: If the job type is not supported for noisy simulations,
            or if it is of type ``OBSERVABLE`` but got no
            ``ExpectationMeasure``.
        NotImplementedError: If the job type is not ``STATE_VECTOR``, ``SAMPLE``
            or ``OBSERVABLE``.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    from braket.circuits import Circuit

    # check some compatibility issues
    if job.job_type == JobType.STATE_VECTOR and job.device.is_remote():
        raise DeviceJobIncompatibleError(
            "State vector cannot be computed using AWS Braket remote simulators"
            " and devices. Please use the LocalSimulator instead"
        )

    is_noisy = bool(job.circuit.noises)
    device = get_braket_device(job.device, is_noisy=is_noisy)  # type: ignore

    # convert job circuit into braket circuit, and apply the noise
    braket_circuit = job.circuit.to_other_language(Language.BRAKET)
    assert isinstance(braket_circuit, Circuit)

    if is_noisy and job.job_type not in [JobType.SAMPLE, JobType.OBSERVABLE]:
        raise ValueError(
            f"Job of type {job.job_type} is not supported for noisy circuits."
        )
    # excute the job based on its type
    if job.job_type == JobType.STATE_VECTOR:
        job.circuit = job.circuit.without_measurements()
        braket_circuit = job.circuit.to_other_language(Language.BRAKET)
        assert isinstance(braket_circuit, Circuit)
        braket_circuit.state_vector()  # type: ignore
        job.status = JobStatus.RUNNING
        task = device.run(braket_circuit, shots=0, inputs=None)

    elif job.job_type == JobType.SAMPLE:
        assert job.measure is not None
        job.status = JobStatus.RUNNING
        task = device.run(braket_circuit, shots=job.measure.shots, inputs=None)

    elif job.job_type == JobType.OBSERVABLE:
        if not isinstance(job.measure, ExpectationMeasure):
            raise ValueError(
                "Cannot compute expectation value if measure used in job is not of "
                "type ExpectationMeasure"
            )

        herm_op = job.measure.observable.to_other_language(Language.BRAKET)
        braket_circuit.expectation(observable=herm_op, target=job.measure.targets)  # type: ignore

        job.status = JobStatus.RUNNING
        task = device.run(braket_circuit, shots=job.measure.shots, inputs=None)

    else:
        raise NotImplementedError(f"Job of type {job.job_type} not handled.")

    return task.id, task


@typechecked
def extract_result(
    braket_result: "GateModelQuantumTaskResult",
    job: Optional[Job] = None,
    device: AWSDevice = AWSDevice.BRAKET_LOCAL_SIMULATOR,
) -> Result:
    """
    Constructs a Result from the result given by the run with Braket.

    Args:
        braket_result: Result returned by myQLM/QLM after running of the job.
        job: Original mpqp job used to generate the run. Used to retrieve more
            easily info to instantiate the result.
        device: AWSDevice on which the job was submitted.

    Returns:
        The ``braket`` result converted to our format.
    """
    from braket.device_schema.ionq import IonqDeviceParameters
    from braket.device_schema.oqc import OqcDeviceParameters
    from braket.device_schema.rigetti import RigettiDeviceParameters
    from braket.device_schema.simulators import GateModelSimulatorDeviceParameters

    if job is None:
        if len(braket_result.values) == 0:
            job_type = JobType.SAMPLE
            nb_qubits = len(list(braket_result.measurement_counts.keys())[0])
            shots = braket_result.task_metadata.shots
            measure = BasisMeasure(list(range(nb_qubits)), shots=shots)
        elif isinstance(braket_result.values[0], float):
            job_type = JobType.OBSERVABLE
            device_params = braket_result.task_metadata.deviceParameters
            assert (
                isinstance(device_params, IonqDeviceParameters)
                or isinstance(device_params, OqcDeviceParameters)
                or isinstance(device_params, RigettiDeviceParameters)
                or isinstance(device_params, GateModelSimulatorDeviceParameters)
            )
            nb_qubits = device_params.paradigmParameters.qubitCount
            shots = braket_result.task_metadata.shots
            measure = ExpectationMeasure(
                list(range(nb_qubits)),
                Observable(np.zeros((2**nb_qubits, 2**nb_qubits), dtype=np.complex64)),
                shots,
            )
        else:
            job_type = JobType.STATE_VECTOR
            nb_qubits = int(math.log2(len(braket_result.values[0])))
            measure = BasisMeasure(list(range(nb_qubits)), shots=0)
        job = Job(job_type, QCircuit(nb_qubits), device, measure)
    job.status = JobStatus.DONE

    if job.job_type == JobType.STATE_VECTOR:
        vector = braket_result.values[0]
        assert isinstance(vector, list) or isinstance(vector, np.ndarray)
        state_vector = StateVector(vector, nb_qubits=job.circuit.nb_qubits)
        return Result(job, state_vector, 0, 0)

    elif job.job_type == JobType.SAMPLE:
        assert job.measure is not None
        counts = braket_result.measurement_counts
        sample_info = []
        for state in counts.keys():
            sample_info.append(
                Sample(job.circuit.nb_qubits, count=counts[state], bin_str=state)
            )
        return Result(job, sample_info, None, job.measure.shots)

    elif job.job_type == JobType.OBSERVABLE:
        assert job.measure is not None
        exp_value = braket_result.values[0]
        return Result(job, exp_value, None, job.measure.shots)

    else:
        raise NotImplementedError(f"Job of type {job.job_type} not handled.")


@typechecked
def get_result_from_aws_task_arn(task_arn: str) -> Result:
    """Retrieves the result, described by the job_id in parameter, from the
    remote QLM and converts it into an mpqp result.

    If the job is still running, we wait (blocking) until it is DONE.

    Args:
        task_arn: Arn of the remote aws task.

    Raises:
        AWSBraketRemoteExecutionError: When the status of the task is unknown.
    """
    from braket.aws import AwsQuantumTask
    from braket.tasks import GateModelQuantumTaskResult, QuantumTask

    task: QuantumTask = AwsQuantumTask(task_arn)
    # catch an error if the id is not correct (wrong ID, wrong region, ...) ?

    status = task.state()

    if status in ["FAILED", "CANCELLED"]:
        raise AWSBraketRemoteExecutionError(f"Job status: {status}")
    elif status in ["CREATED", "QUEUED", "RUNNING", "COMPLETED"]:  #
        result = task.result()
        assert isinstance(result, GateModelQuantumTaskResult)
    else:
        raise AWSBraketRemoteExecutionError(
            f"Unknown status {status} for the task {task_arn}"
        )

    device_arn = task.metadata()["deviceArn"]
    device = AWSDevice.from_arn(device_arn)

    return extract_result(result, None, device)
