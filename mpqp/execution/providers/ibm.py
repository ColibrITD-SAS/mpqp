from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, overload

import numpy as np

from mpqp.core.circuit import CircuitBinding, QCircuit
from mpqp.core.instruction.gates import Gate, Id
from mpqp.core.instruction.gates.native_gates import NativeGate
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.core.languages import Language
from mpqp.execution.connection.ibm_connection import (
    get_backend,
    get_QiskitRuntimeService,
)
from mpqp.execution.devices import AZUREDevice, IBMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector, BatchResult
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
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.result import Result as QiskitResult
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit_ibm_runtime import RuntimeJobV2
    from qiskit.primitives.containers import EstimatorPubLike

    from mpqp.execution.simulated_devices import StaticIBMSimulatedDevice


def run_ibm(job: Job) -> Result | BatchResult:
    """Executes the job on the right IBM Q device precised in the job in
    parameter.

    Args:
        job: Job to be executed.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    return run_aer(job) if not job.device.is_remote() else run_remote_ibm(job)


def compute_expectation_value(
    job: "Job",
    simulator: Optional["AerSimulator"],
    ibm_circuit: Optional["QuantumCircuit"] = None,
    pubs: Optional[list["EstimatorPubLike"]] = None,
    pubs_contexts: Optional[list["Job"]] = None,
    shots: Optional[int] = None,
) -> Result | BatchResult:
    """Configures observable job and run it locally, and returns the
    corresponding Result. Supports both single circuits and batched PUBs.
    """
    from qiskit.quantum_info import SparsePauliOp
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
        context_jobs_to_run = (
            pubs_contexts  # On récupère les contextes générés par Broadcasting
        )

        for pub in pubs:
            circ = pub[0]
            obs_array = pub[1] if len(pub) > 1 else None
            params = pub[2] if len(pub) > 2 else None

            if obs_array is not None and circ.layout is not None:

                def _apply_layout(obs_item):
                    if isinstance(obs_item, list):
                        return [_apply_layout(o) for o in obs_item]
                    return obs_item.apply_layout(circ.layout)

                obs_array = _apply_layout(obs_array)

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

        qiskit_observables: list[SparsePauliOp] = []
        for obs in job.measure.observables:
            if obs.pre_transpiled is None:
                translated = obs.to_other_language(Language.QISKIT)
            else:
                translated = obs.pre_transpiled
            if TYPE_CHECKING:
                assert isinstance(translated, SparsePauliOp)

            if ibm_circuit.layout is not None:
                translated = translated.apply_layout(ibm_circuit.layout)

            qiskit_observables.append(translated)

        pubs_to_run = [(ibm_circuit, qiskit_observables)]
        context_jobs_to_run = [job]

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
    
    for i, context_job in enumerate(context_jobs_to_run):
        pub = pubs_to_run[i]
        
        extracted = extract_result(
            result=estimator_result[i], 
            job=context_job, 
            device=job.device,
            experiment_index=0,
        )
        extracted_items.append(extracted)

    final_flat_results = []
    for item in extracted_items:
        if isinstance(item, BatchResult):
            final_flat_results.extend(item.results)
        else:
            final_flat_results.append(item)

    if len(final_flat_results) == 1:
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

    if not type(job.measure) in job.job_type.value:
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

    Note:
        The qubit order in the returned noise model is reversed to match
        ``qiskit``'s qubit ordering conventions.
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

                    reversed_qubits = [
                        modified_circuit.nb_qubits - 1 - qubit for qubit in connections
                    ]

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
                            reversed_qubits,
                            warnings=False,
                        )
                    else:
                        tensor_error = qiskit_error
                        for _ in range(1, size):
                            tensor_error = tensor_error.tensor(qiskit_error)
                        noise_model.add_quantum_error(
                            tensor_error,
                            [gate.qiskit_string],
                            reversed_qubits,
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

                    reversed_qubits = [
                        modified_circuit.nb_qubits - 1 - qubit for qubit in connections
                    ]

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
                            reversed_qubits,
                            warnings=False,
                        )
                    else:
                        tensor_error = qiskit_error
                        for _ in range(1, len(connections)):
                            tensor_error = tensor_error.tensor(qiskit_error)
                        noise_model.add_quantum_error(
                            tensor_error,
                            [gate.qiskit_string],
                            reversed_qubits,
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
                                [modified_circuit.nb_qubits - 1 - qubit],
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
        backend_sim = None

        if isinstance(job.device, StaticIBMSimulatedDevice):
            if binding.is_noisy:
                import warnings

                warnings.warn(
                    "NoiseModel are ignored when running the circuit on a SimulatedDevice"
                )
            backend_sim = job.device.to_noisy_simulator()

            binding.transpiled_circuits(job.device, backend_sim=backend_sim)
        else:
            binding.transpiled_circuits(job.device, backend_sim=backend_sim)
            if binding.transpiled_noise_model is not None:
                backend_sim = AerSimulator(
                    method=job.device.value, noise_model=binding.transpiled_noise_model
                )
            else:
                backend_sim = AerSimulator(method=job.device.value)

        if job.job_type == JobType.OBSERVABLE:

            if binding.measurements is None:
                raise ValueError(
                    "CircuitBinding requires measurements for OBSERVABLE job types."
                )

            pubs_with_context = binding.Broadcasting(job.device)

            pubs = [item[0] for item in pubs_with_context]
            contexts = [item[1] for item in pubs_with_context]

            return compute_expectation_value(
                job, backend_sim, pubs=pubs, pubs_contexts=contexts, shots=binding.shots
            )
        elif job.job_type in (JobType.SAMPLE, JobType.STATE_VECTOR):
            unrolled_items = binding.unroll()
            bound_circuits = []
            context_jobs = []
            
            for c, v, m in unrolled_items:
                q_c = c.transpiled_circuit
                if TYPE_CHECKING:
                    assert isinstance(q_c, QuantumCircuit)

                b_c = q_c.assign_parameters(v) if v else q_c.copy()

                if job.job_type == JobType.STATE_VECTOR:
                    b_c.save_statevector() # pyright: ignore[reportAttributeAccessIssue]
                elif job.job_type == JobType.SAMPLE:
                    assert isinstance(m, BasisMeasure)
                    for pre_measure in m.pre_measure:
                        cargs = []
                        qiskit_pre_measure = pre_measure.to_other_language(
                            Language.QISKIT
                        )
                        b_c.append(
                            qiskit_pre_measure,
                            list(reversed(pre_measure.targets)),
                            cargs=cargs,
                        )
                    if m._dynamic :
                        tagrets = list(range(c.nb_qubits))
                        c_targets = list(range(c.nb_qubits))
                        from qiskit.circuit import ClassicalRegister

                        creg = ClassicalRegister(c.nb_qubits, "c")
                        b_c.add_register(creg)
                    else:
                        tagrets = list(m.targets)
                        c_targets =  list(m.c_targets)

                    b_c.append(
                        m.to_other_language(Language.QISKIT),
                        [tagrets],
                        [c_targets],
                    )
                bound_circuits.append(b_c)
                
                c_context = c.without_measurements(deep_copy=False)
                if m is not None:
                    c_context.add(m)
                else:
                    c_context.add(c.measurements)
                context_jobs.append(Job(job.job_type, c_context, job.device))
             
            job.status = JobStatus.RUNNING

            if job.job_type == JobType.STATE_VECTOR:
                job_sim = backend_sim.run(bound_circuits, shots=0)
            else:
                shots = binding.shots if binding.shots is not None else 1024
                print(shots)
                job_sim = backend_sim.run(bound_circuits, shots=shots)

            result_sim = job_sim.result()

            extracted_items = []
            for i, context_job in enumerate(context_jobs):
                extracted = extract_result(
                    result=result_sim, 
                    job=context_job, 
                    device=job.device,
                    experiment_index=i
                )
                extracted_items.append(extracted)

            if len(extracted_items) == 1:
                result = extracted_items[0]
            else:
                result = BatchResult(extracted_items)
                
            job.status = JobStatus.DONE
            return result
        else:
            raise ValueError(f"Job type {job.job_type} not handled in CircuitBinding.")
    else:
        job_circuit = job.circuit
        if isinstance(job.device, StaticIBMSimulatedDevice):
            if len(job.circuit.noises) != 0:
                import warnings

                warnings.warn(
                    "NoiseModel are ignored when running the circuit on a "
                    "SimulatedDevice"
                )
                # 3M-TODO: handle case when we put NoiseModel + IBMSimulatedDevice
                # (grab qiskit NoiseModel from AerSimulator generated below, and add
                # to it directly)
            backend_sim = job.device.to_noisy_simulator()
        elif len(job.circuit.noises) != 0:
            if job.circuit.transpiled_circuit is not None:
                if job.circuit.transpiled_noise_model is None:
                    raise InstructionParsingError(
                        "transpiled_noise_model is not initialized"
                    )
                backend_sim = AerSimulator(
                    method=job.device.value,
                    noise_model=job.circuit.transpiled_noise_model,
                )
            else:
                noise_model, modified_circuit = generate_qiskit_noise_model(job.circuit)
                job_circuit = modified_circuit
                backend_sim = AerSimulator(
                    method=job.device.value, noise_model=noise_model
                )
        else:
            backend_sim = AerSimulator(method=job.device.value)

        if job.circuit.transpiled_circuit is None:
            qiskit_circuit = job_circuit.to_other_device(
                job.device, backend_sim=backend_sim
            )
        else:
            qiskit_circuit = job.circuit.transpiled_circuit
            if TYPE_CHECKING:
                assert isinstance(qiskit_circuit, QuantumCircuit)
        if job.job_type == JobType.STATE_VECTOR:
            # the save_statevector method is patched on qiskit_aer load, meaning
            # the type checker can't find it. I hate it but it is what it is.
            # this explains the `type: ignore`. This method is needed to get a
            # statevector out of the statevector simulator.
            qiskit_circuit.save_statevector()  # pyright: ignore[reportAttributeAccessIssue]
            job.status = JobStatus.RUNNING
            job_sim = backend_sim.run(qiskit_circuit, shots=0)
            result_sim = job_sim.result()
            if TYPE_CHECKING:
                assert isinstance(job.device, IBMDevice)
            result = extract_result(result_sim, job, job.device)

        elif job.job_type == JobType.SAMPLE:
            if TYPE_CHECKING:
                assert job.measure is not None

            job.status = JobStatus.RUNNING

            job_sim = backend_sim.run(qiskit_circuit, shots=job.measure.shots)
            result_sim = job_sim.result()
            if TYPE_CHECKING:
                assert isinstance(job.device, (IBMDevice, StaticIBMSimulatedDevice))
            result = extract_result(result_sim, job, job.device)

        elif job.job_type == JobType.OBSERVABLE:
            result = compute_expectation_value(job, backend_sim, qiskit_circuit)

        else:
            raise ValueError(f"Job type {job.job_type} not handled.")

        job.status = JobStatus.DONE
        return result


def submit_remote_ibm(job: Job) -> tuple[str, "RuntimeJobV2"]:
    """Submits the job on the remote IBM device (quantum computer or simulator).

    Args:
        job: Job to be executed.

    Returns:
        IBM's job id and the ``qiskit`` job itself.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import EstimatorV2 as Runtime_Estimator
    from qiskit_ibm_runtime import SamplerV2 as Runtime_Sampler
    from qiskit_ibm_runtime import Session

    meas = job.measure

    check_job_compatibility(job)

    if TYPE_CHECKING:
        assert isinstance(job.device, IBMDevice)
    backend = get_backend(job.device)
    job.device = IBMDevice(backend.name)
    session = Session(backend=backend)

    if job.circuit.transpiled_circuit is None:
        qiskit_circ = job.circuit.to_other_device(job.device)
    else:
        qiskit_circ = job.circuit.transpiled_circuit

    if TYPE_CHECKING:
        assert isinstance(qiskit_circ, QuantumCircuit)

    if job.job_type == JobType.OBSERVABLE:
        if TYPE_CHECKING:
            assert isinstance(meas, ExpectationMeasure)
        estimator = Runtime_Estimator(mode=session)
        qiskit_observables = [
            (
                obs.to_other_language(Language.QISKIT)
                if obs.pre_transpiled is None
                else obs.pre_transpiled
            )
            for obs in meas.observables
        ]
        if TYPE_CHECKING:
            assert all(isinstance(obs, SparsePauliOp) for obs in qiskit_observables)

        qiskit_observables = [
            obs.apply_layout(qiskit_circ.layout) for obs in qiskit_observables
        ]

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
        sampler = Runtime_Sampler(mode=session)
        ibm_job = sampler.run([qiskit_circ], shots=meas.shots)
    else:
        raise NotImplementedError(
            f"{job.job_type} not handled by remote remote IBM devices."
        )

    job.id = ibm_job.job_id()

    return job.id, ibm_job


def submit_remote_ibm_pubs(jobs: list[Job]):
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import EstimatorV2 as Runtime_Estimator
    from qiskit_ibm_runtime import Session
    from qiskit.primitives.containers import EstimatorPubLike

    pubs: list[EstimatorPubLike] = []
    for job in jobs:
        meas = job.measure

        check_job_compatibility(job)

        if TYPE_CHECKING:
            assert isinstance(job.device, IBMDevice)

        if job.circuit.transpiled_circuit is None:
            qiskit_circ = job.circuit.to_other_device(job.device)
        else:
            qiskit_circ = job.circuit.transpiled_circuit

        if TYPE_CHECKING:
            assert isinstance(qiskit_circ, QuantumCircuit)

        if job.job_type == JobType.OBSERVABLE:
            if TYPE_CHECKING:
                assert isinstance(meas, ExpectationMeasure)

            qiskit_observables = [
                (
                    obs.to_other_language(Language.QISKIT)
                    if obs.pre_transpiled is None
                    else obs.pre_transpiled
                )
                for obs in meas.observables
            ]
            if TYPE_CHECKING:
                assert all(isinstance(obs, SparsePauliOp) for obs in qiskit_observables)

            qiskit_observables = [
                obs.apply_layout(qiskit_circ.layout) for obs in qiskit_observables
            ]

            pubs.append((qiskit_circ, qiskit_observables))

        else:
            raise NotImplementedError(
                f"{job.job_type} not handled by remote remote IBM devices."
            )

        backend = get_backend(job.device)
        job.device = IBMDevice(backend.name)

        session = Session(backend=backend)

        estimator = Runtime_Estimator(mode=session)

        # We have to disable all the twirling options and set manually the number of circuits and shots per circuits
        twirling = getattr(estimator.options, "twirling", None)
        if twirling is not None:
            twirling.enable_gates = False
            twirling.enable_measure = False
            twirling.num_randomizations = 1
            twirling.shots_per_randomization = meas.shots

        setattr(estimator.options, "default_shots", meas.shots)

        ibm_job = estimator.run(pubs)

        job.id = ibm_job.job_id()

        return job.id, ibm_job


def run_remote_ibm(job: Job) -> Result:
    """Submits the job on the right IBM remote device, precised in the job in
    parameter, and waits until the job is completed.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    _, remote_job = submit_remote_ibm(job)
    ibm_result = remote_job.result()
    if TYPE_CHECKING:
        assert isinstance(job.device, IBMDevice)

    return extract_result(ibm_result, job, job.device)


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

            exp_values = np.array(res_data.evs)
            stds = (
                np.array(res_data.stds)
                if hasattr(res_data, "stds")
                else np.zeros_like(exp_values)
            )

            shots = (
                job.measure.shots
                if job.device.is_simulator() and job.measure is not None
                else result.metadata["shots"]
            )
            
            measures = job.circuit.measurements if job.circuit.measurements else []

            if exp_values.ndim == 0:
                val = float(exp_values)
                std = float(stds) if stds.size > 0 else 0.0
                m = measures[0] if measures else None
                obs = m.observables[0] if m.observables else None
                label = obs.label if obs else "ibm_obs_0"
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
                        std_val = float(stds[idx]) if stds.size.item() > 0 and stds.size > idx else 0.0
                        m_idx = idx % len(measures) if len(measures) > 0 else 0
                        m = measures[m_idx] if m_idx < len(measures) else measures[0]
                        obs = m.observables[0] if m.observables else None
                        label = obs.label if obs else f"ibm_obs_{idx}"
                        batch_results.append(Result(job, float(val), std_val, shots))
                    return BatchResult(batch_results)
            else:
                batch_results = []
                observables = job.measure.observables
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
                if type(device) == AZUREDevice:
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

    return extract_result(result, None, ibm_device)


def extract_samples(job: Job, result: QiskitResult, experiment_index: int = 0) -> list[Sample]:
    """Extracts measurement samples from the execution results.

    Args:
        job: ``MPQP`` job used to generate the run. Enables a more complete result.
        result: Result returned by IBM after running of the job.
        experiment_index: Index of the experiment/circuit in the batch.

    Returns:
        A list of sample objects representing measurement outcomes.

    """
    counts = result.get_counts(0)
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
