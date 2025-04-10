from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates import Gate, Id
from mpqp.core.instruction.gates.native_gates import NativeGate
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.core.languages import Language
from mpqp.execution.connection.ibm_connection import (
    get_backend,
    get_QiskitRuntimeService,
)
from mpqp.execution.devices import AZUREDevice, IBMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.noise import DimensionalNoiseModel
from mpqp.tools.errors import (
    DeviceJobIncompatibleError,
    IBMNoiseModelGeneration,
    IBMRemoteExecutionError,
    InstructionParsingError,
)
from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.primitives import (
        EstimatorResult,
        PrimitiveResult,
        PubResult,
        SamplerPubResult,
    )
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.result import Result as QiskitResult
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit_ibm_runtime import RuntimeJobV2

    from mpqp.execution.simulated_devices import IBMSimulatedDevice


@typechecked
def run_ibm(job: Job, warnings: bool = True) -> Result:
    """Executes the job on the right IBM Q device precised in the job in
    parameter.

    Args:
        job: Job to be executed.
        warnings:

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    return run_aer(job) if not job.device.is_remote() else run_remote_ibm(job)


@typechecked
def compute_expectation_value(
    ibm_circuit: QuantumCircuit, job: Job, simulator: Optional["AerSimulator"]
) -> Result:
    """Configures observable job and run it locally, and returns the
    corresponding Result.

    Args:
        ibm_circuit: QuantumCircuit (with its qubits already reversed) for which we want
            to estimate the expectation value.
        job: Job containing the execution input data.
        simulator: AerSimulator to be used to set the EstimatorV2 options.

    Returns:
        The Result of the job.

    Raises:
        ValueError: If the job's device is not a
            :class:`~mpqp.execution.simulated_devices.IBMSimulatedDevice`
            and ``simulator`` is ``None``.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    from qiskit.quantum_info import SparsePauliOp

    from mpqp.execution.simulated_devices import IBMSimulatedDevice

    if not isinstance(job.measure, ExpectationMeasure):
        raise ValueError(
            "Cannot compute expectation value if measure used in job is not of "
            "type ExpectationMeasure"
        )

    nb_shots = job.measure.shots

    qiskit_observables: list[SparsePauliOp] = []
    for obs in job.measure.observables:
        translated = obs.to_other_language(Language.QISKIT)
        if TYPE_CHECKING:
            assert isinstance(translated, SparsePauliOp)
        qiskit_observables.append(translated)

    if isinstance(job.device, IBMSimulatedDevice) or nb_shots != 0:
        from qiskit_ibm_runtime import EstimatorV2 as Runtime_Estimator

        backend = (
            job.device.value()
            if isinstance(job.device, IBMSimulatedDevice)
            else simulator
        )

        if TYPE_CHECKING:
            assert isinstance(ibm_circuit, QuantumCircuit)

        qiskit_observables = [
            obs.apply_layout(ibm_circuit.layout) for obs in qiskit_observables
        ]
        options = {"default_shots": nb_shots}
        estimator = Runtime_Estimator(mode=backend, options=options)

    else:
        from qiskit_aer.primitives import EstimatorV2 as Estimator

        if simulator is None:
            raise ValueError("Simulator is required for noisy simulations.")

        simulator.set_options(shots=nb_shots)
        options = {
            "backend_options": simulator.options,
        }
        estimator = Estimator(options=options)

    job.status = JobStatus.RUNNING
    job_expectation = estimator.run([(ibm_circuit, qiskit_observables)])
    estimator_result = job_expectation.result()

    if TYPE_CHECKING:
        assert isinstance(job.device, (IBMDevice, IBMSimulatedDevice))

    return extract_result(estimator_result, job, job.device)


@typechecked
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
    from mpqp.execution.simulated_devices import IBMSimulatedDevice

    if TYPE_CHECKING:
        assert isinstance(job.device, (IBMDevice, IBMSimulatedDevice))

    if not type(job.measure) in job.job_type.value:
        raise DeviceJobIncompatibleError(
            f"An {job.job_type.name} job is valid only if the corresponding circuit has an measure in "
            f"{list(map(lambda cls: cls.__name__, job.job_type.value))}. "
            f"{type(job.measure).__name__} was given instead."
        )

    if job.job_type == JobType.STATE_VECTOR and not job.device.supports_state_vector():
        raise DeviceJobIncompatibleError(
            "Cannot reconstruct state vector with this device. Please use "
            "a local device supporting state vector jobs instead (or change the job "
            "type, for example by giving a number of shots to a BasisMeasure)."
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

    if job.job_type == JobType.OBSERVABLE and not (
        job.device.supports_observable_ideal() or job.device.supports_observable()
    ):
        raise DeviceJobIncompatibleError(
            f"Expectation values cannot be computed with {job.device.name} device"
        )


@typechecked
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
    import io
    import logging

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

    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    log_handler.setLevel(logging.WARNING)

    logger = logging.getLogger()
    logger.addHandler(log_handler)
    try:
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
                                    qiskit_error, [gate.qiskit_string]
                                )
                        else:
                            tensor_error = qiskit_error
                            for _ in range(1, size):
                                tensor_error = tensor_error.tensor(qiskit_error)
                            noise_model.add_all_qubit_quantum_error(
                                tensor_error, [gate.qiskit_string]
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
                            modified_circuit.nb_qubits - 1 - qubit
                            for qubit in connections
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
                            )
                        else:
                            tensor_error = qiskit_error
                            for _ in range(1, size):
                                tensor_error = tensor_error.tensor(qiskit_error)
                            noise_model.add_quantum_error(
                                tensor_error,
                                [gate.qiskit_string],
                                reversed_qubits,
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
                            modified_circuit.nb_qubits - 1 - qubit
                            for qubit in connections
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
                            )
                        else:
                            tensor_error = qiskit_error
                            for _ in range(1, len(connections)):
                                tensor_error = tensor_error.tensor(qiskit_error)
                            noise_model.add_quantum_error(
                                tensor_error,
                                [gate.qiskit_string],
                                reversed_qubits,
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

        log_handler.flush()
        captured_logs = log_stream.getvalue()

        if (
            multiple_noise_warning is False
            and "WARNING: quantum error already exists" in captured_logs
        ):
            pass
        elif len(captured_logs) != 0:
            warnings.warn(
                "Some warnings were raised during the generation of the noise model:\n"
                f"{captured_logs}",
                IBMNoiseModelGeneration,
            )

    finally:
        logger.removeHandler(log_handler)

    return noise_model, modified_circuit


@typechecked
def run_aer(job: Job):
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

    from mpqp.execution.simulated_devices import IBMSimulatedDevice

    if job.circuit.transpiled_circuit is None:
        qiskit_circuit = (
            (
                # 3M-TODO: careful, if we ever support several measurements, the
                # line bellow will have to changer
                job.circuit.without_measurements()
                + job.circuit.pre_measure()
            ).to_other_device(job.device)
            if (job.job_type == JobType.STATE_VECTOR)
            else job.circuit.to_other_device(job.device)
        )
    else:
        qiskit_circuit = job.circuit.transpiled_circuit

    if TYPE_CHECKING:
        assert isinstance(qiskit_circuit, QuantumCircuit)

    if isinstance(job.device, IBMSimulatedDevice):
        if len(job.circuit.noises) != 0:
            warnings.warn(
                "NoiseModel are ignored when running the circuit on a "
                "SimulatedDevice"
            )
            # 3M-TODO: handle case when we put NoiseModel + IBMSimulatedDevice
            # (grab qiskit NoiseModel from AerSimulator generated below, and add
            # to it directly)
        backend_sim = job.device.to_noisy_simulator()
    elif len(job.circuit.noises) != 0:
        if job.circuit.transpiled_noise_model is None:
            raise InstructionParsingError("transpiled_noise_model is not initialized")
        backend_sim = AerSimulator(
            method=job.device.value, noise_model=job.circuit.transpiled_noise_model
        )
    else:
        backend_sim = AerSimulator(method=job.device.value)

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

        if isinstance(job.device, IBMSimulatedDevice):
            from qiskit import transpile

            # TODO I don't know why we need to retranspile here, it is supposed to be done in to_other_device,
            #  but without it, it doesn't woghk
            qiskit_circuit = transpile(qiskit_circuit, backend_sim)

        job_sim = backend_sim.run(qiskit_circuit, shots=job.measure.shots)
        result_sim = job_sim.result()
        if TYPE_CHECKING:
            assert isinstance(job.device, (IBMDevice, IBMSimulatedDevice))
        result = extract_result(result_sim, job, job.device)

    elif job.job_type == JobType.OBSERVABLE:
        result = compute_expectation_value(qiskit_circuit, job, backend_sim)

    else:
        raise ValueError(f"Job type {job.job_type} not handled.")

    job.status = JobStatus.DONE
    return result


@typechecked
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

    service = get_QiskitRuntimeService()
    if TYPE_CHECKING:
        assert isinstance(job.device, IBMDevice)
    backend = get_backend(job.device)
    job.device = IBMDevice(backend.name)
    session = Session(service=service, backend=backend)

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
            obs.to_other_language(Language.QISKIT) for obs in meas.observables
        ]
        if TYPE_CHECKING:
            assert all(isinstance(obs, SparsePauliOp) for obs in qiskit_observables)

        qiskit_observables = [
            obs.apply_layout(  # pyright: ignore[reportAttributeAccessIssue]
                qiskit_circ.layout
            )
            for obs in qiskit_observables
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


@typechecked
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


@typechecked
def extract_result(
    result: "QiskitResult | EstimatorResult | PrimitiveResult[PubResult | SamplerPubResult]",
    job: Optional[Job],
    device: "IBMDevice | IBMSimulatedDevice | AZUREDevice",
) -> Result:
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
    from qiskit.primitives import EstimatorResult, PrimitiveResult
    from qiskit.result import Result as QiskitResult

    # If this is a PubResult from primitives V2
    if isinstance(result, PrimitiveResult):
        # res_data is a DataBin, which means all typechecking is out of the
        # windows for this specific object
        res_data = result[0].data

        if hasattr(res_data, "evs"):
            if job is None:
                job = Job(JobType.OBSERVABLE, QCircuit(0), device, None)

            exp_values = res_data.evs  # pyright: ignore[reportAttributeAccessIssue]
            exp_values = np.atleast_1d(exp_values)

            stds = res_data.stds  # pyright: ignore[reportAttributeAccessIssue]
            stds = np.atleast_1d(stds)
            shots = (
                job.measure.shots
                if job.device.is_simulator() and job.measure is not None
                else result[0].metadata["shots"]
            )

            # If only one result, we directly return the expectation value
            if len(exp_values) == 1:
                return Result(job, float(exp_values[0]), float(stds[0]), shots)

            # If several results, we construct the dictionary with observable labels
            exp_values_dict = dict()
            errors_dict = dict()
            for i in range(len(exp_values)):
                label = (
                    job.measure.observables[i].label
                    if isinstance(job.measure, ExpectationMeasure)
                    else f"ibm_obs_{i}"
                )
                exp_values_dict[label] = float(exp_values[i])
                errors_dict[label] = float(stds[i])

            return Result(job, exp_values_dict, errors_dict, shots)

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
                    BasisMeasure(list(range(nb_qubits)), shots=shots),
                )
            if TYPE_CHECKING:
                assert job.measure is not None

            counts = getattr(res_data, 'c', None)
            counts = counts.get_counts() if counts else {}
            data = [
                Sample(
                    bin_str=item,
                    count=counts[item],
                    nb_qubits=job.circuit.nb_qubits,
                )
                for item in counts
            ]

            return Result(job, data, None, job.measure.shots)

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
                job = Job(JobType.OBSERVABLE, QCircuit(0), device, None)

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
                label = (
                    job.measure.observables[i].label
                    if isinstance(job.measure, ExpectationMeasure)
                    else f"ibm_obs_{i}"
                )
                variance = (
                    result.metadata[i]["variance"]
                    if "variance" in result.metadata[i]
                    else None
                )
                exp_values_dict[label] = result.values[i]
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
                    shots = result.results[0].shots
                    job = Job(
                        job_type,
                        QCircuit(nb_qubits),
                        device,
                        BasisMeasure(list(range(nb_qubits)), shots=shots),
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
                vector = np.array(result.get_statevector())
                state_vector = StateVector(
                    vector,  # pyright: ignore[reportArgumentType]
                    job.circuit.nb_qubits,
                )
                return Result(job, state_vector, 0, 0)
            elif job.job_type == JobType.SAMPLE:
                if TYPE_CHECKING:
                    assert job.measure is not None
                if type(device) == AZUREDevice:
                    from mpqp.execution.providers.azure import (
                        extract_samples as extract_samples_azure,
                    )

                    data = extract_samples_azure(job, result)
                else:
                    data = extract_samples(job, result)
                return Result(job, data, None, job.measure.shots)
            else:
                raise NotImplementedError(f"{job.job_type} not handled.")

        else:
            raise NotImplementedError(f"Result type {type(result)} not handled")


@typechecked
def get_result_from_ibm_job_id(job_id: str) -> Result:
    """Retrieves from IBM remote platform and parse the result of the job_id
    given in parameter. If the job is still running, we wait (blocking) until it
    is ``DONE``.

    Args:
        job_id: Id of the remote IBM job.

    Returns:
        The result (or batch of result) converted to our format.
    """
    from qiskit.providers import BackendV1, BackendV2

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
        assert isinstance(backend, (BackendV1, BackendV2))
    ibm_device = IBMDevice(backend.name)

    return extract_result(result, None, ibm_device)


def extract_samples(job: Job, result: QiskitResult) -> list[Sample]:
    """Extracts measurement samples from the execution results.

    Args:
        job: ``MPQP`` job used to generate the run. Enables a more complete result.
        result: Result returned by IBM after running of the job.

    Returns:
        A list of sample objects representing measurement outcomes.

    """
    counts = result.get_counts(0)
    job_data = result.data()
    return [
        Sample(
            bin_str=item,
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
