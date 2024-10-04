from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np
from typeguard import typechecked

from mpqp.noise import DimensionalNoiseModel

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.primitives import (
        EstimatorResult,
        PrimitiveResult,
        PubResult,
        SamplerPubResult,
    )
    from qiskit.result import Result as QiskitResult
    from qiskit_ibm_runtime import RuntimeJobV2
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates import TOF, CRk, Gate, Id, P, Rk, Rx, Ry, Rz, T, U, NativeGate
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.core.languages import Language
from mpqp.execution.connection.ibm_connection import (
    get_backend,
    get_QiskitRuntimeService,
)
from mpqp.execution.devices import IBMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.tools.errors import DeviceJobIncompatibleError, IBMRemoteExecutionError


@typechecked
def run_ibm(job: Job) -> Result:
    """Executes the job on the right IBM Q device precised in the job in
    parameter.

    Args:
        job: Job to be executed.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    return run_aer(job) if not job.device.is_remote() else run_remote_ibm(job)


@typechecked
def compute_expectation_value(ibm_circuit: QuantumCircuit, job: Job) -> Result:
    """Configures observable job and run it locally, and returns the
    corresponding Result.

    Args:
        ibm_circuit: QuantumCircuit (already reversed bits)
        job: Mpqp job describing the observable job to run.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    from qiskit.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp

    if not isinstance(job.measure, ExpectationMeasure):
        raise ValueError(
            "Cannot compute expectation value if measure used in job is not of "
            "type ExpectationMeasure"
        )
    nb_shots = job.measure.shots
    qiskit_observable = job.measure.observable.to_other_language(Language.QISKIT)
    assert isinstance(qiskit_observable, SparsePauliOp)

    estimator = Estimator()

    # 3M-TODO: think of the possibility to compute several expectation values at
    #  the same time when the circuit is the same apparently the estimator.run()
    #  can take several circuits and observables at the same time, to verify if
    #  putting them all together increases the performance

    job.status = JobStatus.RUNNING
    job_expectation = estimator.run(
        [ibm_circuit], [qiskit_observable], shots=nb_shots if nb_shots != 0 else None
    )
    estimator_result = job_expectation.result()
    assert isinstance(job.device, IBMDevice)
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
    assert isinstance(job.device, IBMDevice)
    if not type(job.measure) in job.job_type.value:
        raise DeviceJobIncompatibleError(
            f"An {job.job_type.name} job is valid only if the corresponding circuit has an measure in "
            f"{list(map(lambda cls: cls.__name__, job.job_type.value))}. "
            f"{type(job.measure).__name__} was given instead."
        )
    if job.job_type == JobType.STATE_VECTOR and not job.device.supports_statevector():
        raise DeviceJobIncompatibleError(
            "Cannot reconstruct state vector with this device. Please use "
            f"{IBMDevice.AER_SIMULATOR_STATEVECTOR} instead (or change the job "
            "type, for example by giving a number of shots to a BasisMeasure)."
        )
    incompatibilities = {
        IBMDevice.AER_SIMULATOR_STABILIZER: {CRk, P, Rk, Rx, Ry, Rz, T, TOF, U},
        IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER: {Rx, Rz},
    }
    if job.device in incompatibilities:
        circ_gates = {type(i) for i in job.circuit.instructions}
        incompatible_gates = circ_gates.intersection(incompatibilities[job.device])
        if len(incompatible_gates) != 0:
            raise ValueError(
                f"Gate(s) {incompatible_gates} cannot be simulated on {job.device}."
            )


@typechecked
def generate_qiskit_noise_model(
    circuit: QCircuit,
) -> tuple["Qiskit_NoiseModel", QCircuit]:
    """
    Generate a Qiskit Aer NoiseModel from the list of MPQP NoiseModels attached to the QCircuit in parameter.

    Args:
        circuit: QCircuit used to generate the Qiskit NoiseModel.

    Returns:
        Qiskit_NoiseModel: A Qiskit NoiseModel combining the provided noise models.

    Note:
        The qubit order is reversed to match Qiskit's qubit ordering conventions.

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

    gate_instructions = modified_circuit.get_gates()

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
                        if size != noise.dimension:
                            continue
                        else:
                            noise_model.add_all_qubit_quantum_error(qiskit_error, [gate.qiskit_string])
                    else:
                        tensor_error = qiskit_error
                        for _ in range(1, size):
                            tensor_error = tensor_error.tensor(qiskit_error)
                        noise_model.add_all_qubit_quantum_error(tensor_error, [gate.qiskit_string])
            else:
                for gate in gate_instructions:

                    if not isinstance(gate, NativeGate):
                        # TODO put a warning to say we ignored this gate, because we don't handle for noise other
                        #  gates than native ones
                        continue

                    connections = gate.connections()
                    size = len(connections)

                    reversed_qubits = [
                        modified_circuit.nb_qubits - 1 - qubit for qubit in connections
                    ]

                    if isinstance(noise, DimensionalNoiseModel) and noise.dimension > size:
                        continue
                    elif (
                        isinstance(noise, DimensionalNoiseModel) and 1 < noise.dimension == size
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
                    # TODO put a warning to say we ignored this gate, because we don't handle for noise other
                    #  gates than native ones
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
                    if isinstance(noise, DimensionalNoiseModel) and noise.dimension > len(connections):
                        continue
                    elif isinstance(noise, DimensionalNoiseModel) and 1 < noise.dimension == len(connections):
                        noise_model.add_quantum_error(
                            qiskit_error,
                            [gate.qiskit_string],
                            reversed_qubits,
                        )
                    else:
                        size = len(connections)
                        tensor_error = qiskit_error
                        for _ in range(1, size):
                            tensor_error = tensor_error.tensor(qiskit_error)
                        noise_model.add_quantum_error(
                            tensor_error,
                            [gate.qiskit_string],
                            reversed_qubits,
                        )

                # Only some targets of the gate are included in the noise targets
                elif len(intersection) != 0:
                    if (not isinstance(noise, DimensionalNoiseModel)) or (noise.dimension == 1):
                        for qubit in intersection:
                            # We add a custom identity gate on the relevant qubits to apply noise after the gate
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
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    check_job_compatibility(job)

    from qiskit import QuantumCircuit
    from qiskit.compiler import transpile
    from qiskit_aer import AerSimulator

    job_circuit = job.circuit

    if len(job.circuit.noises) != 0:
        noise_model, modified_circuit = generate_qiskit_noise_model(job.circuit)
        job_circuit = modified_circuit
        backend_sim = AerSimulator(method=job.device.value, noise_model=noise_model)
    else:
        backend_sim = AerSimulator(method=job.device.value)

    qiskit_circuit = (
        job_circuit.without_measurements().to_other_language(Language.QISKIT)
        if (job.job_type == JobType.STATE_VECTOR)
        else job_circuit.to_other_language(Language.QISKIT)
    )
    if TYPE_CHECKING:
        assert isinstance(qiskit_circuit, QuantumCircuit)

    qiskit_circuit = qiskit_circuit.reverse_bits()
    qiskit_circuit = transpile(
        qiskit_circuit,
        backend_sim,
        layout_method="trivial",
        optimization_level=0,
    )

    if job.job_type == JobType.STATE_VECTOR:
        # the save_statevector method is patched on qiskit_aer load, meaning
        # the type checker can't find it. I hate it but it is what it is.
        # this explains the `type: ignore`. This method is needed to get a
        # statevector our of the statevector simulator...
        qiskit_circuit.save_statevector()  # pyright: ignore[reportAttributeAccessIssue]
        job.status = JobStatus.RUNNING
        job_sim = backend_sim.run(qiskit_circuit, shots=0)
        result_sim = job_sim.result()
        assert isinstance(job.device, IBMDevice)
        result = extract_result(result_sim, job, job.device)

    elif job.job_type == JobType.SAMPLE:
        assert job.measure is not None

        job.status = JobStatus.RUNNING
        job_sim = backend_sim.run(qiskit_circuit, shots=job.measure.shots)
        result_sim = job_sim.result()
        assert isinstance(job.device, IBMDevice)
        result = extract_result(result_sim, job, job.device)

    elif job.job_type == JobType.OBSERVABLE:
        result = compute_expectation_value(qiskit_circuit, job)

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
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    from qiskit import QuantumCircuit
    from qiskit.compiler import transpile
    from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
    from qiskit_ibm_runtime import EstimatorV2 as Runtime_Estimator
    from qiskit_ibm_runtime import SamplerV2 as Runtime_Sampler
    from qiskit_ibm_runtime import Session

    if job.job_type == JobType.STATE_VECTOR:
        raise DeviceJobIncompatibleError(
            "State vector cannot be computed using IBM remote simulators and"
            " devices. Please use a local simulator instead."
        )

    meas = job.measure

    if job.job_type == JobType.OBSERVABLE:
        if not isinstance(meas, ExpectationMeasure):
            raise ValueError(
                "An observable job must is valid only if the corresponding "
                "circuit has an expectation measure."
            )
        if meas.shots == 0:
            raise DeviceJobIncompatibleError(
                "Expectation values cannot be computed exactly using IBM remote"
                " simulators and devices. Please use a local simulator instead."
            )
    check_job_compatibility(job)

    qiskit_circ = job.circuit.to_other_language(Language.QISKIT)
    if TYPE_CHECKING:
        assert isinstance(qiskit_circ, QuantumCircuit)

    qiskit_circ = qiskit_circ.reverse_bits()
    service = get_QiskitRuntimeService()
    assert isinstance(job.device, IBMDevice)
    backend = get_backend(job.device)
    session = Session(service=service, backend=backend)
    qiskit_circ = transpile(qiskit_circ, backend)

    if job.job_type == JobType.OBSERVABLE:
        tot_size = qiskit_circ.num_qubits
        if TYPE_CHECKING:
            assert isinstance(meas, ExpectationMeasure)
        estimator = Runtime_Estimator(session=session)
        qiskit_observable = meas.observable.to_other_language(Language.QISKIT)
        if TYPE_CHECKING:
            assert isinstance(qiskit_observable, SparsePauliOp)

        # Fills the Pauli strings with identities to make the observable size
        # match the circuit size
        qiskit_observable = SparsePauliOp(
            [
                # for some reason, the type checker gets the type of
                # pauli.paulis[0] wrong, and as such the wrong tensor is
                # inferred. Because of this, the type inside it are not guessed
                # properly either, forcing us to "ignore" this problem.
                pauli.paulis[0].tensor(
                    PauliList(  # pyright: ignore[reportArgumentType]
                        Pauli("I" * (tot_size - meas.observable.nb_qubits))
                    )
                )
                for pauli in qiskit_observable
            ],
            coeffs=qiskit_observable.coeffs,
        )

        # FIXME: when we precise the target precision like this, it does not give the right number of shots at the end.
        #  https://github.com/Qiskit/qiskit-ibm-runtime/blob/ed71c5bf8d4fa23c26a0a26c6d45373263e5ecde/qiskit_ibm_runtime/qiskit/primitives/backend_estimator_v2.py#L154
        #  Tried once with shots=1234, but got shots=1280 with the real experiment, looks like the decimal part of
        #  precision is truncated. The problem is on the IBM side, an issue has been published :
        #  https://github.com/Qiskit/qiskit-ibm-runtime/issues/1749
        precision = 1 / np.sqrt(meas.shots)
        ibm_job = estimator.run([(qiskit_circ, qiskit_observable)], precision=precision)
    elif job.job_type == JobType.SAMPLE:
        assert isinstance(meas, BasisMeasure)
        sampler = Runtime_Sampler(session=session)
        ibm_job = sampler.run([qiskit_circ], shots=meas.shots)
    else:
        raise NotImplementedError(f"{job.job_type} not handled.")

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
        :func:``run<mpqp.execution.runner.run>`` instead.
    """
    _, remote_job = submit_remote_ibm(job)
    ibm_result = remote_job.result()

    assert isinstance(job.device, IBMDevice)
    return extract_result(ibm_result, job, job.device)


@typechecked
def extract_result(
    result: "QiskitResult | EstimatorResult | PrimitiveResult[PubResult | SamplerPubResult]",
    job: Optional[Job],
    device: IBMDevice,
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
        res_data = result[0].data
        # res_data is a DataBin, which means all typechecking is out of the
        # windows for this specific object

        # If we are in observable mode
        if hasattr(res_data, "evs"):
            if job is None:
                job = Job(JobType.OBSERVABLE, QCircuit(0), device)
            mean = float(res_data.evs)  # pyright: ignore[reportAttributeAccessIssue]
            error = float(res_data.stds)  # pyright: ignore[reportAttributeAccessIssue]
            shots = result[0].metadata["shots"]
            return Result(job, mean, error, shots)
        # If we are in sample mode
        else:
            if job is None:
                shots = (
                    res_data.meas.num_shots  # pyright: ignore[reportAttributeAccessIssue]
                )
                nb_qubits = (
                    res_data.meas.num_bits  # pyright: ignore[reportAttributeAccessIssue]
                )
                job = Job(
                    JobType.SAMPLE,
                    QCircuit(nb_qubits),
                    device,
                    BasisMeasure(list(range(nb_qubits)), shots=shots),
                )
            assert job.measure is not None

            counts = (
                res_data.meas.get_counts()  # pyright: ignore[reportAttributeAccessIssue]
            )
            data = [
                Sample(
                    bin_str=item, count=counts[item], nb_qubits=job.circuit.nb_qubits
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
                job = Job(JobType.OBSERVABLE, QCircuit(0), device)
            shots = result.metadata[0]["shots"] if "shots" in result.metadata[0] else 0
            variance = (
                result.metadata[0]["variance"]
                if "variance" in result.metadata[0]
                else None
            )
            return Result(job, result.values[0], variance, shots)

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
                assert job.measure is not None
                counts = result.get_counts(0)
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
        The result converted to our format.
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
    assert isinstance(backend, (BackendV1, BackendV2))
    ibm_device = IBMDevice(backend.name)

    return extract_result(result, None, ibm_device)
