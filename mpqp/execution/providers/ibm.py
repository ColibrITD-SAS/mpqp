from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union

from mpqp import QCircuit
from mpqp.core.circuitbinding import CircuitBinding
from mpqp.core.instruction.gates import ControlledGate, Gate, Id
from mpqp.core.instruction.gates.native_gates import NativeGate
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.core.languages import Language
from mpqp.execution.connection.ibm_connection import (
    get_backend,
    get_QiskitRuntimeService,
)
from mpqp.execution.devices import AZUREDevice, IBMDevice
from mpqp.execution.job import ExecutionMode, Job, JobStatus, JobType
from mpqp.execution.result import BatchResult, Result, Sample, StateVector
from mpqp.execution.providers.providers_params import QiskitParams
from mpqp.noise import DimensionalNoiseModel
from mpqp.tools.errors import (
    DeviceJobIncompatibleError,
    IBMRemoteExecutionError,
    InstructionParsingError,
)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.primitives import (
        EstimatorResult,
        PubResult,
        SamplerPubResult,
    )
    from qiskit.providers.backend import BackendV2
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.result import Result as QiskitResult
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit_ibm_runtime import RuntimeJobV2, Session
    from qiskit.primitives.containers import EstimatorPubLike

    from mpqp.execution.simulated_devices import StaticIBMSimulatedDevice


def run_ibm(
    job: Job, qiskit_params: Optional[QiskitParams] = None
) -> Result | BatchResult:
    """Executes the job on the right IBM Q device precised in the job in
    parameter.

    Args:
        job: Job to be executed.
        qiskit_params: IBM Quantum Cloud specific parameters, mainly for remote jobs.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    if not job.device.is_remote():
        return run_aer(job)

    if job.mode == ExecutionMode.SESSION:
        return run_remote_ibm_session(job)

    return run_remote_ibm(job, qiskit_params)


def compute_expectation_value(
    job: "Job",
    simulator: Optional["AerSimulator"],
    ibm_circuit: Optional["QuantumCircuit"] = None,
    pubs: Optional[list["EstimatorPubLike"]] = None,
    pubs_contexts: Optional[list[list["Job"]]] = None,
    shots: Optional[int] = None,
) -> Result | BatchResult:
    """Configures observable job and run it locally, and returns the
    corresponding Result. Supports both single circuits and batched PUBs.

    Each batched PUB has an ordered list of context jobs, one for each
    parameter/measurement pair resolved by the binding mode.
    """
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import PubResult
    from qiskit.primitives.containers import DataBin
    from mpqp.execution.simulated_devices import StaticIBMSimulatedDevice

    pubs_to_run = []
    context_jobs_to_run = []

    if pubs is not None:
        if ibm_circuit is not None:
            raise ValueError(
                "Cannot provide both pubs and ibm_circuit. Please provide only one."
            )
        if shots is None:
            raise ValueError("Shots must be provided when using pubs.")
        if pubs_contexts is None or len(pubs) != len(pubs_contexts):
            raise ValueError("pubs_contexts must perfectly map 1:1 to pubs.")

        nb_shots = shots
        context_jobs_to_run = pubs_contexts

        for pub in pubs:
            circ = pub[0]  # pyright: ignore[reportIndexIssue]
            obs_array = pub[1] if len(pub) > 1 else None  # pyright: ignore
            params = pub[2] if len(pub) > 2 else None  # pyright: ignore

            if obs_array is not None:
                if params is not None:
                    params = [
                        values
                        for observables, values in zip(
                            obs_array, params  # pyright: ignore[reportArgumentType]
                        )
                        for _ in observables
                    ]
                obs_array = [
                    obs
                    for observables in obs_array  # pyright: ignore[reportGeneralTypeIssues]
                    for obs in observables
                ]

            if obs_array is not None and circ.layout is not None:
                obs_array = [obs.apply_layout(circ.layout) for obs in obs_array]

            if params is not None:
                pubs_to_run.append((circ, obs_array, params))
            elif obs_array is not None:
                pubs_to_run.append((circ, obs_array))
            else:
                pubs_to_run.append((circ,))

    else:
        # Exécution classique 1:1 (sans CircuitBinding)
        if ibm_circuit is None:
            raise ValueError("Either pubs or ibm_circuit must be provided.")
        if not isinstance(job.measure, ExpectationMeasure):
            raise ValueError(
                "Cannot compute expectation value if measure used in job is not of "
                f"type ExpectationMeasure, {job.measure}"
            )
        if shots is not None:
            raise ValueError(
                "Shots is already specified in the job.measure. Please do not provide it separately."
            )

        nb_shots = job.measure.shots

        job.measure.pre_transpile_observables(job.device)
        qiskit_observables: list[SparsePauliOp] = []
        for obs in job.measure.observables:
            translated = obs.pre_transpiled[job.device]
            if TYPE_CHECKING:
                assert isinstance(translated, SparsePauliOp)

            if ibm_circuit.layout is not None:
                translated = translated.apply_layout(ibm_circuit.layout)

            qiskit_observables.append(translated)

        pubs_to_run = [(ibm_circuit, qiskit_observables)]
        context_jobs_to_run = [[job]]

    if isinstance(job.device, StaticIBMSimulatedDevice) or nb_shots != 0:
        from qiskit_ibm_runtime import EstimatorV2 as Runtime_Estimator

        backend = (
            job.device.value()
            if isinstance(job.device, StaticIBMSimulatedDevice)
            else simulator
        )
        options = {"default_shots": nb_shots}
        estimator = Runtime_Estimator(mode=backend, options=options)

    else:
        from qiskit_aer.primitives import EstimatorV2 as Estimator

        if simulator is None:
            raise ValueError("Simulator is required for noisy simulations.")

        simulator.set_options(shots=nb_shots)
        options = {"backend_options": simulator.options}
        estimator = Estimator(options=options)

    job.status = JobStatus.RUNNING
    job_expectation = estimator.run(pubs_to_run)
    estimator_result = job_expectation.result()

    if TYPE_CHECKING:
        assert isinstance(job.device, (IBMDevice, StaticIBMSimulatedDevice))
        assert isinstance(estimator_result, list)

    extracted_items = []

    for i, contexts in enumerate(context_jobs_to_run):
        offset = 0
        for context_job in contexts:
            pub_result = estimator_result[i]
            if pubs is not None:
                assert isinstance(context_job.measure, ExpectationMeasure)
                count = len(context_job.measure.observables)
                data = pub_result.data
                pub_result = PubResult(
                    DataBin(
                        evs=data.evs[offset : offset + count],
                        stds=data.stds[offset : offset + count],
                        shape=(count,),
                    ),
                    metadata=pub_result.metadata,
                )
                offset += count
            extracted_items.append(
                extract_result(pub_result, context_job, job.device, experiment_index=0)
            )

    final_flat_results = []
    for item in extracted_items:
        if isinstance(item, BatchResult):
            final_flat_results.extend(item.results)
        else:
            final_flat_results.append(item)
    if isinstance(job.circuit, QCircuit):
        return final_flat_results[0]
    return BatchResult(final_flat_results)


def check_job_compatibility(job: Job):
    """Checks whether the job in parameter has coherent and compatible
    attributes.

    Args:
        job: Job for which we want to check compatibility.

    Raises:
        DeviceJobIncompatibleError: If there is a mismatch between information
            contained in the job (measure and job_type, device and job_type,
            etc...).
    """
    from mpqp.execution.simulated_devices import StaticIBMSimulatedDevice

    if TYPE_CHECKING:
        assert isinstance(job.device, (IBMDevice, StaticIBMSimulatedDevice))

    if job.job_type == JobType.STATE_VECTOR and not job.device.supports_state_vector():
        raise DeviceJobIncompatibleError(
            "Cannot reconstruct state vector with this device. Please use "
            "a local device supporting state vector jobs instead (or change the job "
            "type, for example by giving a number of shots to a BasisMeasure)."
        )

    if job.job_type == JobType.OBSERVABLE and not (
        job.device.supports_observable_ideal() or job.device.supports_observable()
    ):
        raise DeviceJobIncompatibleError(
            f"Expectation values cannot be computed with {job.device.name} device"
        )

    if isinstance(job.circuit, CircuitBinding):
        return

    if type(job.measure) not in job.job_type.value:
        raise DeviceJobIncompatibleError(
            f"An {job.job_type.name} job is valid only if the corresponding circuit has an measure in "
            f"{list(map(lambda cls: cls.__name__, job.job_type.value))}. "
            f"{type(job.measure).__name__} was given instead."
        )

    if (
        job.job_type == JobType.OBSERVABLE
        and job.device.is_remote()
        and job.measure is not None
        and job.measure.shots == 0
    ):
        raise DeviceJobIncompatibleError(
            "Expectation values cannot be computed exactly using IBM remote"
            " simulators and devices. Please use a local simulator instead."
        )


def generate_qiskit_noise_model(
    circuit: QCircuit,
    multiple_noise_warning: bool = True,
) -> tuple["Qiskit_NoiseModel", QCircuit]:
    """Generate a ``qiskit`` noise model packing all the
    :class:`~mpqp.noise.noise_model.NoiseModel` attached to the given QCircuit.

    In ``qiskit``, the noise cannot be applied to qubits unaffected by any
    operations. For this reason, this function also returns a copy of the
    circuit padded with identities on "naked" qubits.

    Args:
        circuit: Circuit containing the noise models to pack.
        multiple_noise_warning: Boolean to enable/disable warnings about
            multiple noise on the same gate. Default True, warnings will be raised.

    Returns:
        A ``qiskit`` noise model combining the provided noise models and the
        modified circuit, padded with identities on the "naked" qubits.

    """

    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel

    noise_model = Qiskit_NoiseModel()

    modified_circuit = deepcopy(circuit)

    used_qubits = set().union(
        *(
            inst.connections()
            for inst in modified_circuit.instructions
            if isinstance(inst, Gate)
        )
    )
    modified_circuit.instructions.extend(
        [
            Id(qubit)
            for qubit in range(modified_circuit.nb_qubits)
            if qubit not in used_qubits
        ]
    )

    gate_instructions = modified_circuit.gates

    noisy_identity_counter = 0

    for noise in modified_circuit.noises:
        qiskit_error = noise.to_other_language(Language.QISKIT)
        if TYPE_CHECKING:
            from qiskit_aer.noise.errors.quantum_error import QuantumError

            assert isinstance(qiskit_error, QuantumError)

        # If all qubits are affected
        if len(noise.targets) == modified_circuit.nb_qubits:
            if len(noise.gates) != 0:
                for gate in noise.gates:
                    size = gate.nb_qubits
                    if TYPE_CHECKING:
                        assert isinstance(size, int)

                    if isinstance(noise, DimensionalNoiseModel):
                        if size == noise.dimension:
                            noise_model.add_all_qubit_quantum_error(
                                qiskit_error, [gate.qiskit_string], warnings=False
                            )
                    else:
                        tensor_error = qiskit_error
                        for _ in range(1, size):
                            tensor_error = tensor_error.tensor(qiskit_error)
                        noise_model.add_all_qubit_quantum_error(
                            tensor_error, [gate.qiskit_string], warnings=False
                        )
            else:
                for gate in gate_instructions:

                    if not isinstance(gate, NativeGate):
                        warnings.warn(
                            f"Ignoring gate '{type(gate)}' as it's not a native gate. "
                            "Noise is only applied to native gates."
                        )
                        continue

                    connections = gate.connections()
                    size = len(connections)

                    qiskit_error_qubits = (
                        gate.controls + gate.targets
                        if isinstance(gate, ControlledGate)
                        else gate.targets
                    )

                    if (
                        isinstance(noise, DimensionalNoiseModel)
                        and noise.dimension > size
                    ):
                        continue
                    elif (
                        isinstance(noise, DimensionalNoiseModel)
                        and 1 < noise.dimension == size
                    ):
                        noise_model.add_quantum_error(
                            qiskit_error,
                            [gate.qiskit_string],
                            qiskit_error_qubits,
                            warnings=False,
                        )
                    else:
                        tensor_error = qiskit_error
                        for _ in range(1, size):
                            tensor_error = tensor_error.tensor(qiskit_error)
                        noise_model.add_quantum_error(
                            tensor_error,
                            [gate.qiskit_string],
                            qiskit_error_qubits,
                            warnings=False,
                        )

        else:
            gates_str = [gate.qiskit_string for gate in noise.gates]

            for gate in gate_instructions:

                if not isinstance(gate, NativeGate):
                    warnings.warn(
                        f"Ignoring gate '{type(gate)}' as it's not a native gate. "
                        "Noise is only applied to native gates."
                    )
                    continue

                # If gates are specified in the noise and the current gate is not in the list, we move to the next one
                if len(gates_str) != 0 and gate.qiskit_string not in gates_str:
                    continue

                connections = gate.connections()
                intersection = connections.intersection(set(noise.targets))

                # Gate targets are included in the noise targets
                if intersection == connections:
                    qiskit_error_qubits = (
                        gate.controls + gate.targets
                        if isinstance(gate, ControlledGate)
                        else gate.targets
                    )

                    # Noise model is multi-dimensional
                    if isinstance(
                        noise, DimensionalNoiseModel
                    ) and noise.dimension > len(connections):
                        continue
                    elif isinstance(
                        noise, DimensionalNoiseModel
                    ) and 1 < noise.dimension == len(connections):
                        noise_model.add_quantum_error(
                            qiskit_error,
                            [gate.qiskit_string],
                            qiskit_error_qubits,
                            warnings=False,
                        )
                    else:
                        tensor_error = qiskit_error
                        for _ in range(1, len(connections)):
                            tensor_error = tensor_error.tensor(qiskit_error)
                        noise_model.add_quantum_error(
                            tensor_error,
                            [gate.qiskit_string],
                            qiskit_error_qubits,
                            warnings=False,
                        )

                # Only some targets of the gate are included in the noise targets
                elif len(intersection) != 0:
                    if (not isinstance(noise, DimensionalNoiseModel)) or (
                        noise.dimension == 1
                    ):
                        for qubit in intersection:
                            # We add a custom identity gate on the relevant
                            # qubits to apply noise after the gate
                            labeled_identity = Id(
                                target=qubit,
                                label=f"noisy_identity_{noisy_identity_counter}",
                            )
                            noise_model.add_quantum_error(
                                qiskit_error,
                                [labeled_identity.label],
                                [qubit],
                                warnings=False,
                            )
                            gate_index = modified_circuit.instructions.index(gate)
                            modified_circuit.instructions.insert(
                                gate_index + 1, labeled_identity
                            )
                            noisy_identity_counter += 1

    return noise_model, modified_circuit


def run_aer(job: Job) -> Result | BatchResult:
    """Executes the job on the right AER local simulator precised in the job in
    parameter.

    Args:
        job: Job to be executed.

    Returns:
        the result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    check_job_compatibility(job)

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    from mpqp.execution.simulated_devices import StaticIBMSimulatedDevice

    if TYPE_CHECKING:
        assert isinstance(job.device, (IBMDevice, StaticIBMSimulatedDevice))

    if isinstance(job.circuit, CircuitBinding):
        binding = job.circuit

        if isinstance(job.device, StaticIBMSimulatedDevice):
            if binding.is_noisy:
                warnings.warn(
                    "NoiseModel are ignored when running the circuit on a SimulatedDevice"
                )
            backend_sim = job.device.to_noisy_simulator()
        else:
            backend_sim = AerSimulator(method=job.device.value)

        binding.transpiled_circuits(job.device, backend_sim=backend_sim)
        if binding.transpiled_noise_model is not None:
            backend_sim = AerSimulator(
                method=job.device.value,
                noise_model=binding.transpiled_noise_model,
            )

        if job.job_type == JobType.OBSERVABLE:
            pubs_with_context = binding.to_other_device(job.device)
            pubs = [item[0] for item in pubs_with_context]
            contexts = [item[1] for item in pubs_with_context]
            return compute_expectation_value(
                job, backend_sim, pubs=pubs, pubs_contexts=contexts, shots=binding.shots
            )

        if job.job_type in (JobType.SAMPLE, JobType.STATE_VECTOR):
            unrolled_items = binding.unroll()
            bound_circuits: list[QuantumCircuit] = []
            context_jobs: list[Job] = []

            for c, v, m in unrolled_items:
                q_c = c.transpiled_circuit[job.device]
                if TYPE_CHECKING:
                    assert isinstance(q_c, QuantumCircuit)

                if v:
                    normalized_values = {str(key): val for key, val in v.items()}
                    b_c = q_c.assign_parameters(
                        {p: normalized_values[p.name] for p in q_c.parameters}
                    )
                else:
                    b_c = q_c.copy()

                c_context = c.without_measurements(deep_copy=False)
                if m is not None:
                    c_context.add(deepcopy(m))
                if job.job_type == JobType.STATE_VECTOR:
                    b_c.save_statevector()  # pyright: ignore[reportAttributeAccessIssue]
                elif job.job_type == JobType.SAMPLE:
                    measure = c_context.measurements[0]
                    assert isinstance(measure, BasisMeasure)
                    assert measure.c_targets is not None
                    from qiskit.circuit import ClassicalRegister

                    if b_c.num_clbits < c_context.nb_cbits:
                        b_c.add_register(
                            ClassicalRegister(c_context.nb_cbits - b_c.num_clbits)
                        )
                    for pre_measure in measure.pre_measure:
                        qiskit_pre_measure = pre_measure.to_other_language(
                            Language.QISKIT
                        )
                        b_c.append(
                            qiskit_pre_measure,
                            list(reversed(pre_measure.targets)),
                            cargs=[],
                        )
                    b_c.append(
                        measure.to_other_language(Language.QISKIT),
                        [measure.targets],
                        [measure.c_targets],
                    )
                bound_circuits.append(b_c)

                context_jobs.append(Job(job.job_type, c_context, job.device, values=v))

            job.status = JobStatus.RUNNING
            if job.job_type == JobType.STATE_VECTOR:
                job_sim = backend_sim.run(bound_circuits, shots=0)
            else:
                shots = binding.shots if binding.shots is not None else 1024
                job_sim = backend_sim.run(bound_circuits, shots=shots)
            result_sim = job_sim.result()
            extracted_items: list[Result] = []
            for i, context_job in enumerate(context_jobs):
                extracted = extract_result(
                    result=result_sim,
                    job=context_job,
                    device=job.device,
                    experiment_index=i,
                )
                if not isinstance(extracted, Result):
                    raise TypeError("A single IBM experiment must return a Result.")
                extracted_items.append(extracted)
            job.status = JobStatus.DONE
            return BatchResult(extracted_items)

        raise ValueError(f"Job type {job.job_type} not handled in CircuitBinding.")

    if isinstance(job.device, StaticIBMSimulatedDevice):
        if job.circuit.noises:
            warnings.warn(
                "NoiseModel are ignored when running the circuit on a SimulatedDevice"
            )
        backend_sim = job.device.to_noisy_simulator()
    elif job.circuit.noises:
        qiskit_circuit = job.circuit.transpiled_for_device(job.device)
        if job.circuit.transpiled_noise_model is None:
            raise InstructionParsingError("transpiled_noise_model is not initialized")
        backend_sim = AerSimulator(
            method=job.device.value,
            noise_model=job.circuit.transpiled_noise_model,
        )
    else:
        backend_sim = AerSimulator(method=job.device.value)

    qiskit_circuit = job.circuit.transpiled_for_device(job.device)
    if TYPE_CHECKING:
        assert isinstance(qiskit_circuit, QuantumCircuit)

    if job.job_type == JobType.STATE_VECTOR:
        qiskit_circuit.save_statevector()  # pyright: ignore[reportAttributeAccessIssue]
        job.status = JobStatus.RUNNING
        result_sim = backend_sim.run(qiskit_circuit, shots=0).result()
        result = extract_result(result_sim, job, job.device)
    elif job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert job.measure is not None
        job.status = JobStatus.RUNNING
        result_sim = backend_sim.run(qiskit_circuit, shots=job.measure.shots).result()
        result = extract_result(result_sim, job, job.device)
    elif job.job_type == JobType.OBSERVABLE:
        result = compute_expectation_value(job, backend_sim, qiskit_circuit)
    else:
        raise ValueError(f"Job type {job.job_type} not handled.")

    job.status = JobStatus.DONE
    return result


def _submit_remote_ibm(
    job: Job,
    qiskit_params: Optional[QiskitParams] = None,
    *,
    runtime_target: Union[BackendV2, Session],
) -> tuple[str, "RuntimeJobV2"]:
    """Submits the job on the remote IBM device (quantum computer or simulator).

    Args:
        job: Job to be executed.
        qiskit_params: IBM Quantum Cloud specific parameters, mainly for remote submissions.

    Returns:
        IBM's job id and the ``qiskit`` job itself.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_ibm_runtime import EstimatorV2 as Runtime_Estimator
    from qiskit_ibm_runtime import SamplerV2 as Runtime_Sampler

    meas = job.measure
    check_job_compatibility(job)

    if TYPE_CHECKING:
        assert isinstance(job.device, IBMDevice)
        assert isinstance(job.circuit, QCircuit)

    instance = qiskit_params.instance if qiskit_params is not None else None

    backend = get_backend(job.device, instance)
    job.device = IBMDevice(backend.name)

    qiskit_circ = job.circuit.transpiled_for_device(job.device)

    if TYPE_CHECKING:
        assert isinstance(qiskit_circ, QuantumCircuit)

    if job.job_type == JobType.OBSERVABLE:
        if TYPE_CHECKING:
            assert isinstance(meas, ExpectationMeasure)
        estimator = Runtime_Estimator(mode=backend)
        # estimator = Runtime_Estimator(mode=runtime_target)
        qiskit_observables = [
            (
                obs.to_other_language(Language.QISKIT)
                if job.device not in obs.pre_transpiled
                else obs.pre_transpiled[job.device]
            )
            for obs in meas.observables
        ]
        if TYPE_CHECKING:
            assert all(isinstance(obs, SparsePauliOp) for obs in qiskit_observables)

        # We have to disable all the twirling options and set manually the number of circuits and shots per circuits
        twirling = getattr(estimator.options, "twirling", None)
        if twirling is not None:
            twirling.enable_gates = False
            twirling.enable_measure = False
            twirling.num_randomizations = 1
            twirling.shots_per_randomization = meas.shots

        setattr(estimator.options, "default_shots", meas.shots)
        ibm_job = estimator.run([(qiskit_circ, qiskit_observables)])

    elif job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert isinstance(meas, BasisMeasure)
        # sampler = Runtime_Sampler(mode=runtime_target)
        sampler = Runtime_Sampler(mode=backend)
        ibm_job = sampler.run([qiskit_circ], shots=meas.shots)

    else:
        raise NotImplementedError(
            f"{job.job_type} not handled by remote remote IBM devices."
        )

    job.id = ibm_job.job_id()

    return job.id, ibm_job


def submit_remote_ibm(
    job: Job, qiskit_params: Optional[QiskitParams] = None
) -> tuple[str, "RuntimeJobV2"]:
    # TODO: docs
    if TYPE_CHECKING:
        assert isinstance(job.device, IBMDevice)
    backend = get_backend(job.device)

    try:
        job.device = IBMDevice(backend.name)
    except Exception:
        pass

    return _submit_remote_ibm(job, qiskit_params, runtime_target=backend)


def submit_remote_ibm_batch(jobs: list[Job]) -> tuple[list[str], "RuntimeJobV2"]:
    # TODO: docs
    if len(jobs) == 0:
        raise ValueError(
            "Can't submit an IBM batch: job list is empty. "
            "Batch execution requires at leat one Job object"
        )

    reference_job = jobs[0]
    if TYPE_CHECKING:
        assert isinstance(reference_job.device, IBMDevice)
    backend = get_backend(reference_job.device)

    from qiskit.quantum_info import SparsePauliOp
    from qiskit_ibm_runtime import EstimatorV2 as Runtime_Estimator

    from mpqp.execution.connection.ibm_connection import get_or_create_ibm_session

    execution_target = (
        get_or_create_ibm_session(backend)
        if reference_job.mode == ExecutionMode.SESSION
        else backend
    )
    estimator = Runtime_Estimator(mode=execution_target)

    per_job_circuits: list[QuantumCircuit] = []
    per_job_observables: list[list[SparsePauliOp]] = []

    for job in jobs:
        meas = job.measure
        check_job_compatibility(job)

        circuit = job.circuit
        if not isinstance(circuit, QCircuit):
            raise TypeError("IBM batch jobs must contain QCircuit instances.")
        if job.values is not None:
            circuit.bind_parameters(job.device, job.values)

        qc = circuit.transpiled_for_device(job.device)
        if TYPE_CHECKING:
            assert isinstance(qc, QuantumCircuit)

        per_job_circuits.append(qc)

        if TYPE_CHECKING:
            assert isinstance(meas, ExpectationMeasure)
        meas.pre_transpile_observables(job.device)

        obs_list: list[SparsePauliOp] = []

        for obs in meas.observables:
            translated = obs.pre_transpiled[job.device]
            if TYPE_CHECKING:
                assert isinstance(translated, SparsePauliOp)
            obs_list.append(translated.apply_layout(qc.layout))

        if TYPE_CHECKING:
            assert all(isinstance(obs, SparsePauliOp) for obs in obs_list)

        per_job_observables.append(obs_list)

    estimator_input = list(zip(per_job_circuits, per_job_observables))
    ibm_job = estimator.run(estimator_input)

    job_ids = [ibm_job.job_id()] * len(jobs)
    for job, job_id in zip(jobs, job_ids):
        job.id = job_id

    return job_ids, ibm_job


def submit_remote_ibm_session(
    job: Job, session: "Session"
) -> tuple[str, "RuntimeJobV2"]:
    # TODO: docs
    return _submit_remote_ibm(job, runtime_target=session)


def run_remote_ibm(job: Job, qiskit_params: Optional[QiskitParams] = None) -> Result:
    """Submits the job on the right IBM remote device, precised in the job in
    parameter, and waits until the job is completed.

    Args:
        job: Job to be executed.
        qiskit_params: IBM Quantum Cloud specific parameters, mainly for remote jobs.


    Returns:
        A Result after submission and execution of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    _, remote_job = submit_remote_ibm(job, qiskit_params)
    ibm_result = remote_job.result()
    if TYPE_CHECKING:
        assert isinstance(job.device, IBMDevice)

    result = extract_result(ibm_result, job, job.device)
    if not isinstance(result, Result):
        raise TypeError("A single IBM job must return a Result.")
    return result


def run_remote_ibm_session(job: Job) -> Result:
    # TODO: docs
    from mpqp.execution.connection.ibm_connection import get_or_create_ibm_session

    if TYPE_CHECKING:
        assert isinstance(job.device, IBMDevice)
    backend = get_backend(job.device)
    session = get_or_create_ibm_session(backend)

    _, remote_job = submit_remote_ibm_session(job, session)
    ibm_result = remote_job.result()
    result = extract_result(ibm_result, job, job.device)
    if not isinstance(result, Result):
        raise TypeError("A single IBM session job must return a Result.")
    return result


def run_remote_ibm_batch(jobs: list[Job]) -> BatchResult:
    _, remote_job = submit_remote_ibm_batch(jobs)
    ibm_batch_results = remote_job.result()

    mpqp_batch_results = []
    for job, res in zip(jobs, ibm_batch_results):
        if TYPE_CHECKING:
            assert isinstance(job.device, IBMDevice)
        mpqp_batch_results.append(extract_result(res, job, job.device))

    return BatchResult(mpqp_batch_results)


def extract_result(
    result: "QiskitResult | EstimatorResult | PubResult | SamplerPubResult",
    job: Optional[Job],
    device: "IBMDevice | StaticIBMSimulatedDevice | AZUREDevice",
    experiment_index: int = 0,
) -> Result | BatchResult:
    """Parses a result from ``IBM`` execution (remote or local) in a ``MPQP``
    :class:`~mpqp.execution.result.Result`.

    Args:
        result: Result returned by IBM after running of the job.
        job: ``MPQP`` job used to generate the run. Enables a more complete
            result.
        device: IBMDevice on which the job was submitted. Used to know if the
            run was remote or local

    Returns:
        The ``qiskit`` result converted to our format.
    """
    from qiskit.result import Result as QiskitResult
    from qiskit.primitives import PubResult, SamplerPubResult, EstimatorResult
    import numpy as np

    # If this is a PubResult from primitives V2
    if isinstance(result, (PubResult | SamplerPubResult)):
        res_data = result.data
        if hasattr(res_data, "evs"):
            if job is None:
                job = Job(JobType.OBSERVABLE, QCircuit(0), device)

            exp_values = np.array(res_data.evs)  # type: ignore
            stds = (
                np.array(res_data.stds)  # type: ignore
                if hasattr(res_data, "stds")
                else np.zeros_like(exp_values)
            )

            shots = (
                job.measure.shots
                if job.device.is_simulator() and job.measure is not None
                else result.metadata["shots"]
            )

            measures: list[ExpectationMeasure] = (
                job.circuit.measurements if job.circuit.measurements else []
            )  # pyright: ignore[reportAssignmentType]

            if exp_values.ndim == 0:
                val = float(exp_values)
                std = float(stds) if stds.size > 0 else 0.0
                return Result(job, val, std, shots)
            elif exp_values.ndim == 2:
                N_obs, M_params = exp_values.shape
                batch_results = []
                for j in range(M_params):
                    exp_dict, err_dict = {}, {}
                    obs_flat_idx = 0
                    for m in measures:
                        for obs in m.observables:
                            label = obs.label
                            if obs_flat_idx < N_obs:
                                exp_dict[label] = float(exp_values[obs_flat_idx, j])
                                err_dict[label] = float(stds[obs_flat_idx, j])
                            obs_flat_idx += 1
                    batch_results.append(Result(job, exp_dict, err_dict, shots))
                return BatchResult(batch_results)

            elif exp_values.ndim == 1:
                total_obs = sum(len(m.observables) for m in measures)

                if len(exp_values) == total_obs and len(measures) == 1:
                    exp_dict, err_dict = {}, {}
                    if len(measures[0].observables) == 1:
                        return Result(job, exp_values[0], stds[0], shots)
                    for idx, obs in enumerate(measures[0].observables):
                        label = obs.label
                        exp_dict[label] = float(exp_values[idx])
                        err_dict[label] = float(stds[idx])
                    return Result(job, exp_dict, err_dict, shots)

                else:
                    batch_results = []
                    for idx, val in enumerate(exp_values):
                        std_val = (
                            float(stds[idx])
                            if stds.size.item() > 0 and stds.size > idx  # type: ignore
                            else 0.0
                        )
                        m_idx = idx % len(measures) if len(measures) > 0 else 0
                        m = measures[m_idx] if m_idx < len(measures) else measures[0]
                        obs = m.observables[0] if m.observables else None
                        label = obs.label if obs else f"ibm_obs_{idx}"
                        batch_results.append(Result(job, float(val), std_val, shots))
                    return BatchResult(batch_results)
            else:
                batch_results = []
                observables = measures[0].observables
                nb_observables = len(observables)
                for idx, val in np.ndenumerate(exp_values):
                    std_val = float(stds[idx]) if stds.size > 0 else 0.0
                    obs_idx = (
                        idx[-1] % nb_observables
                        if len(idx) > 0 and nb_observables > 0
                        else 0
                    )

                    if nb_observables <= 1:
                        batch_results.append(Result(job, float(val), std_val, shots))
                    else:
                        label = (
                            observables[obs_idx].label
                            if obs_idx < len(observables)
                            else f"ibm_obs_{obs_idx}"
                        )
                        if TYPE_CHECKING:
                            assert label
                        batch_results.append(
                            Result(job, {label: float(val)}, {label: std_val}, shots)
                        )

                return BatchResult(batch_results)

        else:
            if job is None:
                shots = (
                    res_data.c.num_shots  # pyright: ignore[reportAttributeAccessIssue]
                )
                nb_qubits = (
                    res_data.c.num_bits  # pyright: ignore[reportAttributeAccessIssue]
                )
                job = Job(
                    JobType.SAMPLE,
                    QCircuit(
                        [BasisMeasure(list(range(nb_qubits)), shots=shots)],
                        nb_qubits=nb_qubits,
                    ),
                    device,
                )
            if TYPE_CHECKING:
                assert job.measure is not None

            bit_array = None
            for key in dir(res_data):
                if not key.startswith("_"):
                    val = getattr(res_data, key)
                    if hasattr(val, "get_counts"):
                        bit_array = val
                        break

            if bit_array is None:
                raise ValueError("No valid BitArray found in SamplerPubResult data.")

            counts_data = bit_array.get_counts()
            shots = bit_array.num_shots

            counts_array = np.atleast_1d(counts_data)
            batch_results = []

            for count_dict in counts_array:
                data = [
                    Sample(bin_str=k[::-1], count=v, nb_qubits=job.circuit.nb_qubits)
                    for k, v in count_dict.items()
                ]
                batch_results.append(Result(job, data, None, shots))

            if len(batch_results) == 1:
                return batch_results[0]
            return BatchResult(batch_results)

    else:

        if job is not None and (
            isinstance(result, EstimatorResult) != (job.job_type == JobType.OBSERVABLE)
        ):
            raise ValueError(
                "Mismatch between job type and result type: if the result is an"
                " `EstimatorResult` the job must be of type `OBSERVABLE` but here was not."
            )

        if isinstance(result, EstimatorResult):
            if job is None:
                job = Job(JobType.OBSERVABLE, QCircuit(0), device)

            if len(result.values) == 1:
                return Result(
                    job,
                    result.values[0],
                    (
                        result.metadata[0]["variance"]
                        if "variance" in result.metadata[0]
                        else None
                    ),
                    result.metadata[0]["shots"] if "shots" in result.metadata[0] else 0,
                )

            exp_values_dict = dict()
            errors_dict = dict()

            shots = result.metadata[0]["shots"] if "shots" in result.metadata[0] else 0

            for i in range(len(result.values)):
                qiskit_order = len(result.values) - i - 1
                label = (
                    job.measure.observables[i].label
                    if isinstance(job.measure, ExpectationMeasure)
                    else f"ibm_obs_{i}"
                )
                variance = (
                    result.metadata[qiskit_order]["variance"]
                    if "variance" in result.metadata[qiskit_order]
                    else None
                )
                exp_values_dict[label] = result.values[qiskit_order]
                errors_dict[label] = variance

            return Result(job, exp_values_dict, errors_dict, shots)

        elif isinstance(
            result, QiskitResult
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
            if job is None:
                job_data = result.data()
                if "statevector" in job_data:
                    job_type = JobType.STATE_VECTOR
                    nb_qubits = int(math.log(len(result.get_statevector()), 2))
                    job = Job(job_type, QCircuit(nb_qubits), device)
                elif "counts" in job_data:
                    job_type = JobType.SAMPLE
                    nb_qubits = len(list(result.get_counts())[0])
                    assert result.results is not None
                    shots = result.results[0].shots
                    job = Job(
                        job_type,
                        QCircuit(
                            [BasisMeasure(list(range(nb_qubits)), shots=shots)],
                            nb_qubits=nb_qubits,
                        ),
                        device,
                    )
                else:
                    if len(result.data()) == 0:
                        raise ValueError(
                            "Result data is empty, cannot extract anything. Check "
                            "if the associated job was successfully completed."
                        )
                    else:
                        raise ValueError(
                            f"Data with keys {result.data().keys()} in result not handled."
                        )

            if job.job_type == JobType.STATE_VECTOR:
                vector = np.array(result.get_statevector(experiment_index).reverse_qargs())  # type: ignore[reportUnnecessaryIsInstance]
                state_vector = StateVector(
                    vector,
                    job.circuit.nb_qubits,
                )
                return Result(job, state_vector, 0, 0, False)
            elif job.job_type == JobType.SAMPLE:
                if TYPE_CHECKING:
                    assert job.measure is not None
                if isinstance(device, AZUREDevice):
                    from mpqp.execution.providers.azure import (
                        extract_samples as extract_samples_azure,
                    )

                    data = extract_samples_azure(job, result)
                else:
                    data = extract_samples(job, result, experiment_index)
                return Result(job, data, None, job.measure.shots)
            else:
                raise NotImplementedError(f"{job.job_type} not handled.")

        else:
            raise NotImplementedError(f"Result type {type(result)} not handled")


def get_result_from_ibm_job_id(job_id: str) -> Result:
    """Retrieves from IBM remote platform and parse the result of the job_id
    given in parameter. If the job is still running, we wait (blocking) until it
    is ``DONE``.

    Args:
        job_id: Id of the remote IBM job.

    Returns:
        The result (or batch of result) converted to our format.
    """
    from qiskit.providers import BackendV2

    connector = get_QiskitRuntimeService()
    ibm_job = (
        connector.job(job_id)
        if job_id in [job.job_id() for job in connector.jobs()]
        else None
    )

    if ibm_job is None:
        raise IBMRemoteExecutionError(
            f"Job with id {job_id} was not found on this account."
        )

    status = ibm_job.status()
    if status in ["CANCELLED", "ERROR"]:
        raise IBMRemoteExecutionError(
            f"Trying to retrieve an IBM result for a job in status {status}"
        )

    # If the job is finished, it will get the result, if still running it is block until it finishes
    result = ibm_job.result()
    backend = ibm_job.backend()
    if TYPE_CHECKING:
        assert isinstance(backend, BackendV2)
    ibm_device = IBMDevice(backend.name)

    result = extract_result(result, None, ibm_device)
    if TYPE_CHECKING:
        assert isinstance(result, Result)
    return result


def extract_samples(
    job: Job, result: QiskitResult, experiment_index: int = 0
) -> list[Sample]:
    """Extracts measurement samples from the execution results.

    Args:
        job: ``MPQP`` job used to generate the run. Enables a more complete result.
        result: Result returned by IBM after running of the job.
        experiment_index: Index of the experiment/circuit in the batch.

    Returns:
        A list of sample objects representing measurement outcomes.

    """
    counts = result.get_counts(experiment_index)
    job_data = result.data(experiment_index)
    return [
        Sample(
            bin_str=item[::-1],
            count=counts[item],
            nb_qubits=job.circuit.nb_qubits,
            probability=(
                job_data.get("probabilities").get(item)
                if "probabilities" in job_data
                else None
            ),
        )
        for item in counts
    ]
