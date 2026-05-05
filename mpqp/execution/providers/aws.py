from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates import CRk
from mpqp.core.instruction.measurement import (
    BasisMeasure,
    ExpectationMeasure,
    Observable,
)
from mpqp.core.languages import Language
from mpqp.execution.connection.aws_connection import get_braket_device
from mpqp.execution.devices import AWSDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.noise.noise_model import NoiseModel
from mpqp.tools.errors import (
    AWSBraketRemoteExecutionError,
    DeviceJobIncompatibleError,
    DeviceJobIncompatibleWarning,
)

if TYPE_CHECKING:
    from braket.circuits import Circuit
    from braket.tasks import GateModelQuantumTaskResult, QuantumTask


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
        if TYPE_CHECKING:
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
                    gate.braket_gate
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


def run_braket(job: Job, reservation_arn: Optional[str] = None) -> Result:
    # TODO: check if we keep just reservation_arn, or if we provide change it to `provider_specific_options` dict,
    #  to be more generic
    """Executes the job on the right AWS Braket device (local or remote)
    precised in the job in parameter and waits until the task is completed, then
    returns the Result.

    Args:
        job: Job to be executed, it MUST be corresponding to a
            :class:`mpqp.execution.devices.AWSDevice`.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    # TODO : [multi-obs] update this to take into account the case when we have list of Observables
    # TODO : [multi-obs] check if Braket allows for a list of several observables
    if not isinstance(job.device, AWSDevice):
        raise ValueError(
            "`job` must correspond to an `AWSDevice`, but corresponds to a "
            f"{job.device} instead"
        )

    import warnings

    from braket.tasks import GateModelQuantumTaskResult

    
    try:
        if isinstance(job.measure, ExpectationMeasure):
            return run_braket_observable(job, reservation_arn)
        _, task = submit_job_braket(job, reservation_arn)
        res = task.result()
        if TYPE_CHECKING:
            assert isinstance(res, GateModelQuantumTaskResult)

        return extract_result(res, job, job.device)

    except DeviceJobIncompatibleError as e:
        warnings.warn(str(e), DeviceJobIncompatibleWarning, stacklevel=1)

        job.status = JobStatus.ERROR
        job.status_message = "Job execution failed. See warning for details."

        return Result(
            job,
            data=None,
            shots=0,
        )


def run_braket_observable(job: Job, reservation_arn: Optional[str] = None):
    # TODO: check if we keep just reservation_arn, or if we provide change it to `provider_specific_options` dict,
    #  to be more generic
    """Returns the result of an ``OBSERVABLE`` job.

    TODO: check that the link bellow is correctly generated.
    If :attr:`~mpqp.execution.job.Job.measure.optimize_measurement`, this
    function will run based on the grouping of the pauli monomials (Read
    :ref:`TODO here` for more information).

    Otherwise each observable will be ran one by one.

    Args:
        job: Job to be executed.

    Returns:
        A result containing the expectation values of the observables.
    """
    from braket.circuits import Circuit
    from braket.tasks import GateModelQuantumTaskResult

    assert isinstance(job.device, AWSDevice)

    transpiled_circuit = job.circuit.transpiled_for_device(job.device)
    assert isinstance(transpiled_circuit, Circuit)

    device = get_braket_device(
        job.device,
        is_noisy=bool(job.circuit.noises),
    )

    if job.measure is None:
        raise NotImplementedError("job.measure is None")
    assert isinstance(job.measure, ExpectationMeasure)

    results, errors = {}, {}

    if job.measure.optimize_measurement:
        # TODO: review this if it can be rewritten or regrouped. What if the observable were already transpiled?
        #  we still call it?
        job.measure.pre_transpile_observables(job.device)
        eigenvalues_by_group, transpiled_pre_measures = (
            job.measure.translated_pre_measures[job.device]
        )
        # TODO: check that the transpilation of observables/premeasure is done/required correctly

        expectation_values: dict[str, float] = {}

        for group_eigenvalues, pre_measure in zip(
            eigenvalues_by_group, transpiled_pre_measures
        ):
            job.status = JobStatus.RUNNING
            if job.measure.shots == 0:
                from copy import deepcopy

                cirq = deepcopy(transpiled_circuit + pre_measure)
                cirq.state_vector()  # pyright: ignore[reportAttributeAccessIssue]
                local_result = device.run(cirq, shots=0, inputs=None).result()

                assert isinstance(local_result, GateModelQuantumTaskResult)
                values = local_result.values[0]
                sorted_values = []
                for i in range(len(values)):
                    sorted_values.append(float(np.abs(values[i]) ** 2))
            else:
                local_result = device.run(
                    transpiled_circuit + pre_measure,
                    shots=job.measure.shots,
                    inputs=None,
                )
                result = local_result.result()
                assert isinstance(result, GateModelQuantumTaskResult)

                length = 2**job.circuit.nb_qubits
                sorted_values: list[float] = []
                for i in range(length):
                    binary_state = f"{bin(i)[2:].zfill(len(bin(length))- 3)}"
                    if binary_state in result.measurement_probabilities:
                        sorted_values.append(
                            result.measurement_probabilities[binary_state].real
                        )
                    else:
                        sorted_values.append(0)
            for name, eigenvalue in group_eigenvalues.items():
                expectation_value: float = np.dot(
                    eigenvalue,
                    np.array(sorted_values, dtype=np.float64),
                )
                expectation_values[name] = expectation_value
        for i, obs in enumerate(job.measure.observables):
            string = obs.pauli_string
            local: float = 0
            for monoms in string.monomials:
                if TYPE_CHECKING:
                    assert isinstance(monoms.coef, (int, float))
                local += expectation_values[monoms.name] * monoms.coef
            results.update({f"observable_{i}": local})
            errors.update({f"observable_{len(errors)}": None})

        if len(results) == 1:
            return Result(job, results["observable_0"], shots=job.measure.shots)
        return Result(job, results, errors, shots=job.measure.shots)

    else:
        from copy import deepcopy

        braket_sum = None
        index = []
        for i, obs in enumerate(job.measure.observables):
            from braket.circuits.observables import Hermitian

            if job.measure.shots == 0:
                # TODO: Remove this when Braket will have fixed PauliString with coeff
                # force the conversion to matrix to avoid issues with pauli_string with coeff
                if obs._pauli_string is not None:  # pyright: ignore[reportPrivateUsage]
                    obs = deepcopy(obs)
                    obs.matrix
                    obs._pauli_string = None  # pyright: ignore[reportPrivateUsage]
            braket_obs = obs.to_other_language(Language.BRAKET)

            if isinstance(braket_obs, Hermitian):
                copy = deepcopy(transpiled_circuit)
                copy.expectation(  # pyright: ignore[reportAttributeAccessIssue]
                    observable=braket_obs, target=job.measure.targets
                )
                job.status = JobStatus.RUNNING
                local_result = device.run(
                    copy, shots=job.measure.shots, inputs=None
                ).result()
                assert isinstance(local_result, GateModelQuantumTaskResult)
                results.update({f"observable_{i}": local_result.values[0].real})
                errors.update({f"observable_{i}": None})
            else:
                index.append(i)
                results.update({f"observable_{i}": None})
                errors.update({f"observable_{i}": None})
                braket_sum = (
                    braket_sum + braket_obs if braket_sum is not None else braket_obs
                )

        if braket_sum is not None:
            from braket.program_sets import CircuitBinding, ProgramSet
            from braket.tasks.program_set_quantum_task_result import (
                ProgramSetQuantumTaskResult,
            )

            copy = deepcopy(transpiled_circuit)
            program_set = ProgramSet(
                CircuitBinding(
                    copy,
                    observables=braket_sum,
                )
            )
            job.status = JobStatus.RUNNING

            if TYPE_CHECKING:
                assert isinstance(device, AWSDevice)

            local_result = device.run(
                program_set,
                shots=program_set.total_executables * job.measure.shots,
                inputs=None,
            ).result()
            assert isinstance(local_result, ProgramSetQuantumTaskResult)
            for res in local_result:
                for i, value in enumerate(res.entries):
                    results.update({f"observable_{index[i]}": value.expectation})
                    errors.update({f"observable_{index[i]}": None})

        if len(results) == 1:
            return Result(job, results["observable_0"], None, job.measure.shots)

    return Result(job, results, errors, job.measure.shots)


def submit_job_braket(
    job: Job, reservation_arn: Optional[str] = None
) -> tuple[str, "QuantumTask"]:
    # TODO: change reservation_arn to more generic parameter for provider_options
    """Submits the job to the right local/remote device and returns the
    generated task.

    Args:
        job: Job to be executed, it MUST be corresponding to a
            :class:`mpqp.execution.devices.AWSDevice`.

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
        :func:`~mpqp.execution.runner.run` instead.
    """
    if not isinstance(job.device, AWSDevice):
        raise ValueError(
            "`job` must correspond to an `AWSDevice`, but corresponds to a "
            f"{job.device} instead"
        )

    if job.job_type == JobType.STATE_VECTOR and job.device.is_remote():
        raise DeviceJobIncompatibleError(
            "State vector cannot be computed using AWS Braket remote simulators"
            " and devices. Please use the LocalSimulator instead"
        )
    if job.job_type == JobType.SAMPLE and job.measure is None:
        raise ValueError("`SAMPLE` jobs must have a measure.")
    if job.job_type == JobType.OBSERVABLE and not isinstance(
        job.measure, ExpectationMeasure
    ):
        raise ValueError("`OBSERVABLE` jobs must have an `ExpectationMeasure`.")
    is_noisy = bool(job.circuit.noises)
    if is_noisy and job.job_type not in [JobType.SAMPLE, JobType.OBSERVABLE]:
        raise ValueError(
            f"Job of type {job.job_type} is not supported for noisy circuits."
        )

    from braket.circuits import Circuit

    device = get_braket_device(job.device, is_noisy=is_noisy)
    braket_circuit = job.circuit.transpiled_for_device(job.device)
    if TYPE_CHECKING:
        assert isinstance(braket_circuit, Circuit)

    if job.job_type == JobType.STATE_VECTOR:
        # rebind safe_retrieve_samples from braket to Normalize the probability
        # because the bracket does not do so and this causes a crash.
        from braket.default_simulator.state_vector_simulation import (
            StateVectorSimulation,
        )

        def safe_retrieve_samples(self):  # pyright: ignore[reportMissingParameterType]
            probs = self.probabilities
            probs = probs / np.sum(probs)
            return np.random.choice(len(self._state_vector), p=probs, size=self._shots)

        StateVectorSimulation.retrieve_samples = safe_retrieve_samples
        # ----

        braket_circuit.state_vector()  # pyright: ignore[reportAttributeAccessIssue]
        job.status = JobStatus.RUNNING

        if TYPE_CHECKING:
            assert isinstance(device, AWSDevice)
        task = device.run(braket_circuit, shots=0, inputs=None)

    elif job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert job.measure is not None
        job.status = JobStatus.RUNNING
        if TYPE_CHECKING:
            assert isinstance(device, AWSDevice)
        task = device.run(braket_circuit, shots=job.measure.shots, inputs=None)

    elif job.job_type == JobType.OBSERVABLE:
        if TYPE_CHECKING:
            assert isinstance(job.measure, ExpectationMeasure)

        # TODO : [multi-obs] update this to take into account the case when we have list of Observables
        #  and in this case looping over the list of observables is not sufficient, we have to take into account the
        #  grouping and do it manually if job.measure.optimize_measurement is true otherwise do it sequentially
        if job.measure.observables[0].pre_transpiled is None:
            herm_op = job.measure.observables[0].to_other_language(Language.BRAKET)
        else:
            herm_op = job.measure.observables[0].pre_transpiled
        braket_circuit.expectation(  # pyright: ignore[reportAttributeAccessIssue]
            observable=herm_op, target=job.measure.targets
        )

        job.status = JobStatus.RUNNING

        if TYPE_CHECKING:
            assert isinstance(device, AWSDevice)
        task = device.run(braket_circuit, shots=job.measure.shots, inputs=None)

    else:
        raise NotImplementedError(f"Job of type {job.job_type} not handled.")

    job.id = task.id

    return (
        task.id,
        task,
    )  # TODO : [multi-obs] update this to take into account the case when we have list of Observables


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
            if TYPE_CHECKING:
                assert isinstance(
                    device_params,
                    (
                        IonqDeviceParameters,
                        OqcDeviceParameters,
                        RigettiDeviceParameters,
                        GateModelSimulatorDeviceParameters,
                    ),
                )
            nb_qubits = device_params.paradigmParameters.qubitCount
            shots = braket_result.task_metadata.shots
            measure = ExpectationMeasure(
                Observable(np.zeros((2**nb_qubits, 2**nb_qubits), dtype=np.complex128)),
                list(range(nb_qubits)),
                shots,
            )
        else:
            job_type = JobType.STATE_VECTOR
            nb_qubits = int(math.log2(len(braket_result.values[0])))
            measure = BasisMeasure(list(range(nb_qubits)), shots=0)
        job = Job(job_type, QCircuit([measure], nb_qubits=nb_qubits), device)
    job.status = JobStatus.DONE

    if job.job_type in (JobType.SAMPLE, JobType.OBSERVABLE) and job.measure is None:
        raise ValueError("`SAMPLE` or `OBSERVABLE` jobs must have a measure.")

    if job.job_type == JobType.STATE_VECTOR:
        vector = braket_result.values[0]
        if TYPE_CHECKING:
            assert isinstance(vector, (list, np.ndarray))
        state_vector = StateVector(vector, nb_qubits=job.circuit.nb_qubits)
        return Result(job, state_vector, 0, 0)

    elif job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert job.measure is not None
        counts = braket_result.measurement_counts
        sample_info = []
        for state in counts.keys():
            sample_info.append(
                Sample(job.circuit.nb_qubits, count=counts[state], bin_str=state)
            )
        return Result(job, sample_info, None, job.measure.shots)

    elif job.job_type == JobType.OBSERVABLE:
        if TYPE_CHECKING:
            assert job.measure is not None
        exp_value = braket_result.values[0]
        return Result(job, exp_value, None, job.measure.shots)

    else:
        raise NotImplementedError(f"Job of type {job.job_type} not handled.")


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
    from braket.tasks import GateModelQuantumTaskResult

    task: QuantumTask = AwsQuantumTask(task_arn)
    # catch an error if the id is not correct (wrong ID, wrong region, ...) ?

    status = task.state()

    if status in ["FAILED", "CANCELLED"]:
        raise AWSBraketRemoteExecutionError(f"Job status: {status}")
    elif status in ["CREATED", "QUEUED", "RUNNING", "COMPLETED"]:  #
        result = task.result()
        if TYPE_CHECKING:
            assert isinstance(result, GateModelQuantumTaskResult)
    else:
        raise AWSBraketRemoteExecutionError(
            f"Unknown status {status} for the task {task_arn}"
        )

    device_arn = task.metadata()["deviceArn"]
    device = AWSDevice.from_arn(device_arn)

    return extract_result(result, None, device)


def estimate_cost_single_job(
    job: Job, hybrid_iterations: int = 1, estimated_time_seconds: int = 3
) -> float:
    """
    Estimates the cost of executing a :class:`~mpqp.execution.job.Job` on a remote AWS Braket device.

    Args:
        job: :class:`~mpqp.execution.job.Job` for which we want to estimate the cost. The job's device must be an :class:`~mpqp.execution.devices.AWSDevice`.
        hybrid_iterations: Number of iteration in a case of a hybrid (quantum-classical) job.
        estimated_time_seconds: Estimated runtime for simulator jobs (in seconds). The minimum duration billing is 3 seconds.

    Returns:
        The estimated price (in USD) for the execution of the job in parameter.

    Example:
        >>> circuit = QCircuit([H(0), CNOT(0, 1), CNOT(1, 2), BasisMeasure(shots=245)])
        >>> job = generate_job(circuit, AWSDevice.IONQ_ARIA_1)
        >>> estimate_cost_single_job(job, hybrid_iterations=150)
        1147.5

    """

    if not isinstance(job.device, AWSDevice):
        raise ValueError(
            f"This function was expecting a job with an AWSDevice but got a {type(job.device).__name__}."
        )

    if job.device.is_remote():
        if job.device.is_simulator():
            if "sv1" in job.device.value or "dm1" in job.device.value:
                minute_cost = 0.075
            elif "tn1" in job.device.value:
                minute_cost = 0.275
            else:
                raise ValueError
            return minute_cost * max(estimated_time_seconds / 60, 3 / 60)
        else:
            if job.measure is None:
                raise DeviceJobIncompatibleError(
                    "An AWS remote job on a quantum computer requires to have a measure."
                )

            if "ionq" in job.device.value:
                task_cost = 0.3
                shot_cost = 0.03

            elif "iqm" in job.device.value:
                task_cost = 0.3
                shot_cost = 0.00145

            elif "rigetti" in job.device.value:
                task_cost = 0.3
                shot_cost = 0.0009

            elif "quera" in job.device.value:
                task_cost = 0.3
                shot_cost = 0.01

            else:
                raise NotImplementedError(
                    f"Cost estimation not implemented yet for {job.device.name} device."
                )

            return (task_cost + job.measure.shots * shot_cost) * hybrid_iterations

    else:
        return 0


# TODO: finish this implementation, why did we need context manager ?
# @contextmanager
def optional_reservation_arn(device: AWSDevice, reservation_arn: Optional[str] = None):
    from braket.aws import DirectReservation

    if reservation_arn:
        with DirectReservation(device.get_arn(), reservation_arn=reservation_arn):
            yield
    else:
        yield
