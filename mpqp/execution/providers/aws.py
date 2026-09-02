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
    from braket.devices.device import Device as BraketDevice
    from braket.program_sets import ProgramSet
    from braket.tasks import GateModelQuantumTaskResult, QuantumTask
    from braket.tasks.program_set_quantum_task_result import (
        ProgramSetQuantumTaskResult,
    )


def _ordered_measurement_probabilities(
    probabilities: dict[str, float],
    measured_qubits: list[int],
    targets: list[int],
) -> np.ndarray:
    """Return probabilities with bits ordered like the observable targets."""
    measured_positions = {
        qubit: position for position, qubit in enumerate(measured_qubits)
    }
    try:
        target_positions = [measured_positions[target] for target in targets]
    except KeyError as error:
        raise ValueError(
            f"Braket did not measure observable target {error.args[0]}."
        ) from error

    ordered_probabilities = np.zeros(2 ** len(targets), dtype=np.float64)
    for measured_state, probability in probabilities.items():
        target_state = "".join(
            measured_state[position] for position in target_positions
        )
        ordered_probabilities[int(target_state, 2)] += probability
    return ordered_probabilities


def _embed_pauli_observable_for_braket(
    observable: Observable,
    targets: list[int],
    nb_qubits: int,
) -> Observable:
    """Embed a target-local Pauli observable in the complete circuit."""
    from mpqp.core.instruction.measurement.pauli_string import (
        PauliString,
        PauliStringMonomial,
        pI,
    )

    embedded_pauli_string = PauliString()
    for monomial in observable.pauli_string.monomials:
        embedded_atoms = [pI] * nb_qubits
        for local_index, target in enumerate(targets):
            embedded_atoms[target] = monomial.atoms[local_index]
        embedded_pauli_string += PauliStringMonomial(
            monomial.coef,
            embedded_atoms,
        )
    return Observable(embedded_pauli_string.simplify(), label=observable.label)


def _run_braket_program_set(
    job: Job,
    device: "BraketDevice",
    program_set: "ProgramSet",
) -> "ProgramSetQuantumTaskResult":
    """Run a program set and bind its single Braket task id to the job."""
    from braket.tasks.program_set_quantum_task_result import (
        ProgramSetQuantumTaskResult,
    )

    job.status = JobStatus.RUNNING
    task = device.run(
        program_set,
        shots=program_set.total_shots,
        inputs=None,
    )
    job.id = task.id
    result = task.result()
    assert isinstance(result, ProgramSetQuantumTaskResult)
    return result


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

    for noise in reversed(noises):
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


def run_braket(job: Job) -> Result:
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
            return run_braket_observable(job)

        job.id, task = submit_job_braket(job)
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


def _run_exact_braket_observables(
    job: Job,
    transpiled_circuit: "Circuit",
    device: "BraketDevice",
) -> Result:
    """Evaluate all exact observables as result types of one circuit."""
    from copy import deepcopy

    from braket.circuits import ResultType
    from braket.circuits.observables import Hermitian, I as BraketIdentity
    from braket.tasks import GateModelQuantumTaskResult

    assert isinstance(job.measure, ExpectationMeasure)
    exact_circuit = deepcopy(transpiled_circuit)
    braket_targets = job.measure.targets
    result_specs: list[list[tuple[int, float]]] = []

    # Braket mishandles coefficients on Pauli strings. Evaluate unique,
    # coefficient-free monomials and recombine them locally instead.
    for observable in job.measure.observables:
        observable_specs: list[tuple[int, float]] = []
        if observable._pauli_string is not None:  # pyright: ignore[reportPrivateUsage]
            for monomial in observable.pauli_string.monomials:
                coefficient = float(monomial.coef)
                if coefficient == 0:
                    continue
                braket_observable = (monomial / coefficient).to_other_language(
                    Language.BRAKET
                )
                result_type = ResultType.Expectation(
                    observable=braket_observable,
                    target=braket_targets,
                )
                if result_type not in exact_circuit.result_types:
                    exact_circuit.add_result_type(result_type)
                observable_specs.append(
                    (exact_circuit.result_types.index(result_type), coefficient)
                )

            if not observable_specs:
                result_type = ResultType.Expectation(
                    observable=BraketIdentity(),
                    target=[braket_targets[0]],
                )
                if result_type not in exact_circuit.result_types:
                    exact_circuit.add_result_type(result_type)
                observable_specs.append(
                    (exact_circuit.result_types.index(result_type), 0.0)
                )
        else:
            braket_observable = Hermitian(
                observable.matrix,
                display_name=(
                    observable.label if observable.label is not None else "Hermitian"
                ),
            )
            result_type = ResultType.Expectation(
                observable=braket_observable,
                target=braket_targets,
            )
            if result_type not in exact_circuit.result_types:
                exact_circuit.add_result_type(result_type)
            observable_specs.append(
                (exact_circuit.result_types.index(result_type), 1.0)
            )
        result_specs.append(observable_specs)

    job.status = JobStatus.RUNNING
    task = device.run(exact_circuit, shots=0, inputs=None)
    job.id = task.id
    exact_result = task.result()
    assert isinstance(exact_result, GateModelQuantumTaskResult)

    results: dict[str, float] = {}
    errors: dict[str, None] = {}
    for index, observable_specs in enumerate(result_specs):
        results[f"observable_{index}"] = sum(
            coefficient * float(exact_result.values[result_index].real)
            for result_index, coefficient in observable_specs
        )
        errors[f"observable_{index}"] = None
    job.status = JobStatus.DONE
    if len(results) == 1:
        return Result(job, results["observable_0"], None, job.measure.shots)
    return Result(job, results, errors, job.measure.shots)


def _run_optimized_braket_observables(
    job: Job,
    transpiled_circuit: "Circuit",
    device: "BraketDevice",
) -> Result:
    """Evaluate grouped Pauli observables in one Braket program set."""
    from copy import deepcopy

    from braket.program_sets import ProgramSet

    from mpqp.tools.pauli_grouping import (
        find_qubitwise_rotations,
        pauli_monomial_eigenvalues,
    )

    assert isinstance(job.measure, ExpectationMeasure)
    if job.measure.pre_transpiled is None:
        grouping = job.measure.get_pauli_grouping()
        pre_measures = [QCircuit(find_qubitwise_rotations(group)) for group in grouping]
        for pre_measure in pre_measures:
            for instruction in pre_measure.instructions:
                instruction.targets = [
                    job.measure.targets[target] for target in instruction.targets
                ]
        transpiled_pre_measures = [
            pre_measure.to_other_language(Language.BRAKET)
            for pre_measure in pre_measures
        ]
        eigenvalues = [
            {monomial.name: pauli_monomial_eigenvalues(monomial) for monomial in group}
            for group in grouping
        ]
    else:
        eigenvalues, transpiled_pre_measures = (
            job.measure.pre_transpiled
        )  # pyright: ignore[reportGeneralTypeIssues]

    programs = []
    for pre_measure in transpiled_pre_measures:
        circuit = deepcopy(transpiled_circuit + pre_measure)
        circuit.measure(job.measure.targets)
        programs.append(circuit)

    program_set = ProgramSet(programs, shots_per_executable=job.measure.shots)
    program_set_result = _run_braket_program_set(job, device, program_set)

    expectation_values: dict[str, float] = {}
    for eigenvalues_by_name, program_result in zip(
        eigenvalues, program_set_result, strict=True
    ):
        measured_entry = program_result.entries[0]
        probabilities = _ordered_measurement_probabilities(
            measured_entry.probabilities,
            measured_entry.measured_qubits,
            job.measure.targets,
        )
        for name, monomial_eigenvalues in eigenvalues_by_name.items():
            expectation_values[name] = float(
                np.dot(monomial_eigenvalues, probabilities)
            )

    results: dict[str, float] = {}
    errors: dict[str, None] = {}
    for index, observable in enumerate(job.measure.observables):
        expectation = sum(
            expectation_values[monomial.name] * float(monomial.coef)
            for monomial in observable.pauli_string.monomials
        )
        results[f"observable_{index}"] = expectation
        errors[f"observable_{index}"] = None
    job.status = JobStatus.DONE
    if len(results) == 1:
        return Result(job, results["observable_0"], None, job.measure.shots)
    return Result(job, results, errors, job.measure.shots)


def _run_sampled_braket_observables(
    job: Job,
    transpiled_circuit: "Circuit",
    device: "BraketDevice",
) -> Result:
    """Evaluate ungrouped sampled observables in one Braket program set."""
    from copy import deepcopy

    from braket.circuits import Instruction
    from braket.circuits.observables import Hermitian
    from braket.program_sets import CircuitBinding, ProgramSet

    assert isinstance(job.measure, ExpectationMeasure)
    programs = []
    result_specs: list[tuple[int, np.ndarray | None, list[int] | None]] = []

    for index, observable in enumerate(job.measure.observables):
        if observable._pauli_string is None:  # pyright: ignore[reportPrivateUsage]
            braket_observable = Hermitian(
                observable.matrix,
                display_name=(
                    observable.label if observable.label is not None else "Hermitian"
                ),
            )
            circuit = deepcopy(transpiled_circuit)
            for gate in braket_observable.basis_rotation_gates:
                circuit.add_instruction(Instruction(gate, target=job.measure.targets))
            circuit.measure(job.measure.targets)
            programs.append(circuit)
            result_specs.append(
                (index, braket_observable.eigenvalues, job.measure.targets)
            )
        else:
            embedded_observable = _embed_pauli_observable_for_braket(
                observable,
                job.measure.targets,
                job.circuit.nb_qubits,
            )
            programs.append(
                CircuitBinding(
                    deepcopy(transpiled_circuit),
                    observables=embedded_observable.to_other_language(Language.BRAKET),
                )
            )
            result_specs.append((index, None, None))

    program_set = ProgramSet(programs, shots_per_executable=job.measure.shots)
    program_set_result = _run_braket_program_set(job, device, program_set)

    results: dict[str, float] = {}
    errors: dict[str, None] = {}
    for (observable_index, eigenvalues, targets), program_result in zip(
        result_specs, program_set_result, strict=True
    ):
        if eigenvalues is None:
            expectation = program_result.expectation()
            if expectation is None:
                raise ValueError(
                    "Braket did not return an expectation value for "
                    f"observable_{observable_index}."
                )
        else:
            assert targets is not None
            measured_entry = program_result.entries[0]
            probabilities = _ordered_measurement_probabilities(
                measured_entry.probabilities,
                measured_entry.measured_qubits,
                targets,
            )
            expectation = np.dot(eigenvalues, probabilities)

        results[f"observable_{observable_index}"] = float(expectation)
        errors[f"observable_{observable_index}"] = None
    job.status = JobStatus.DONE
    if len(results) == 1:
        return Result(job, results["observable_0"], None, job.measure.shots)
    return Result(job, results, errors, job.measure.shots)


def run_braket_observable(job: Job) -> Result:
    """Run an ``OBSERVABLE`` job as one Braket task.

    TODO: check that the link bellow is correctly generated.
    If :attr:`~mpqp.execution.job.Job.measure.optimize_measurement`, this
    function will run based on the grouping of the pauli monomials (Read
    :ref:`TODO here` for more information).

    Exact simulations attach all expectation result types to one circuit.
    Sampled simulations submit all required circuits in one program set. Both
    cases therefore create a single Braket task and bind its id to ``job.id``.

    Args:
        job: Job to be executed.

    Returns:
        The result containing one expectation value when the measure contains
        one observable, or a dictionary indexed by observable when it contains
        several observables.

    Raises:
        NotImplementedError: If the job does not contain a measurement.

    Note:
        This function is not meant to be used directly. Use
        :func:`mpqp.execution.runner.run` instead.
    """
    from braket.circuits import Circuit

    assert isinstance(job.device, AWSDevice)
    if job.circuit.transpiled_circuit is None:
        transpiled_circuit = job.circuit.to_other_device(job.device)
    else:
        transpiled_circuit = job.circuit.transpiled_circuit
        assert isinstance(transpiled_circuit, Circuit)

    if job.measure is None:
        raise NotImplementedError("job.measure is None")
    assert isinstance(job.measure, ExpectationMeasure)

    device = get_braket_device(job.device, is_noisy=bool(job.circuit.noises))
    if job.measure.shots == 0:
        return _run_exact_braket_observables(job, transpiled_circuit, device)

    all_observables_are_pauli = all(
        observable._pauli_string is not None  # pyright: ignore[reportPrivateUsage]
        for observable in job.measure.observables
    )
    if job.measure.optimize_measurement and all_observables_are_pauli:
        return _run_optimized_braket_observables(job, transpiled_circuit, device)
    return _run_sampled_braket_observables(job, transpiled_circuit, device)


def submit_job_braket(job: Job) -> tuple[str, "QuantumTask"]:
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

    if job.circuit.transpiled_circuit is None:
        braket_circuit = job.circuit.to_other_device(job.device)
    else:
        braket_circuit = job.circuit.transpiled_circuit

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
        # TODO : [multi-obs] update this to take into account the case when we have list of Observables
        if TYPE_CHECKING:
            assert isinstance(job.measure, ExpectationMeasure)
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

    parsed_result = extract_result(result, None, device)
    parsed_result.job.id = task_arn
    parsed_result.job.status = JobStatus.DONE
    return parsed_result


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
