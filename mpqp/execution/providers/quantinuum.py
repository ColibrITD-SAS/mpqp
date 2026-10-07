from __future__ import annotations

import math
from collections import Counter
from numbers import Complex
from typing import TYPE_CHECKING, Optional
from warnings import warn

import numpy as np

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure, ExpectationMeasure
from mpqp.core.instruction.measurement.pauli_string import CommutingTypes
from mpqp.core.languages import Language
from mpqp.execution.connection.quantinuum_connection import get_quantinuum_config
from mpqp.execution.devices import QUANTINUUMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.providers.providers_params import QuantinuumParams
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.tools.errors import (
    DeviceJobIncompatibleError,
    ModifiedShotsNumberWarning,
    NumberQubitsWarning,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from pytket.backends.backend import Backend
    from pytket.backends.backendresult import BackendResult
    from pytket.circuit import Circuit as tket_Circuit
    from qnexus.models.references import (
        CircuitRef,
        CompilationResultRef,
        ExecuteJobRef,
    )


def run_quantinuum(
    job: Job, quantinuum_params: Optional[QuantinuumParams] = None
) -> Result:
    """Execute a job on a local or remote Quantinuum device, wait for it to
    complete, and return the result.

    Args:
        job: Job to execute. It must target a
            :class:`mpqp.execution.devices.QUANTINUUMDevice`.
        quantinuum_params: Quantinuum specific parameters used to configure
            circuit compilation and observable grouping.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    if not isinstance(job.device, QUANTINUUMDevice):
        raise DeviceJobIncompatibleError(
            "`job` must correspond to a `QUANTINUUMDevice`, but corresponds "
            f"to a {job.device} instead."
        )

    if not job.device.is_remote():
        return run_tket_local(job, quantinuum_params)
    else:
        return run_nexus_remote(job, quantinuum_params)


def check_job_compatibility(job: Job) -> None:
    """Check whether a job is compatible with Quantinuum execution.

    Args:
        job: Job to validate.

    Raises:
        DeviceJobIncompatibleError: If the job type, measurement, or target
            device are incompatible.
    """
    if not isinstance(job.device, QUANTINUUMDevice):
        raise DeviceJobIncompatibleError(
            "`job` must correspond to a `QUANTINUUMDevice`, but corresponds "
            f"to a {job.device} instead."
        )

    if type(job.measure) not in job.job_type.value:
        expected_measures = ", ".join(
            measure_type.__name__ for measure_type in job.job_type.value
        )
        raise DeviceJobIncompatibleError(
            f"A {job.job_type.name} job requires a measurement of type "
            f"{expected_measures}, but {type(job.measure).__name__} was given."
        )

    if TYPE_CHECKING:
        assert isinstance(job.device, QUANTINUUMDevice)

    if job.job_type == JobType.SAMPLE and not job.device.supports_samples():
        raise DeviceJobIncompatibleError(
            f"{job.device} does not support `SAMPLE` jobs."
        )

    if job.job_type == JobType.STATE_VECTOR and not job.device.supports_state_vector():
        raise DeviceJobIncompatibleError(
            f"{job.device} does not support `STATE_VECTOR` jobs."
        )

    if job.job_type == JobType.OBSERVABLE:
        if TYPE_CHECKING:
            assert isinstance(job.measure, ExpectationMeasure)
        if job.measure.shots == 0:
            if job.device.is_remote():
                raise DeviceJobIncompatibleError(
                    "Quantinuum Nexus does not handle exact observable jobs. Submit a "
                    "state-vector job instead and compute the expectation value locally."
                )
        else:
            if not job.device.supports_samples() and job.measure.optimize_measurement:
                raise DeviceJobIncompatibleError(
                    f"{job.device} does not support sampled observable jobs."
                )
            if (
                not job.device.supports_observable()
                or not job.device.supports_samples()
            ):
                raise DeviceJobIncompatibleError(
                    f"{job.device} does not support sampled or observable jobs."
                )
            if (
                job.measure.optimize_measurement
                and job.measure.commuting_type != CommutingTypes.QUBITWISE
            ):
                raise DeviceJobIncompatibleError(
                    "Optimized sampled observable jobs on Quantinuum currently support only "
                    "qubit-wise commuting Pauli grouping."
                )


def run_tket_local(
    job: Job, quantinuum_params: Optional[QuantinuumParams] = None
) -> Result:
    """Execute a job using a local TKET backend.

    Args:
        job: Job targeting a local TKET device.
        quantinuum_params: Quantinuum specific parameters used to configure
            circuit compilation and observable grouping.

    Returns:
        The result after local compilation and execution.
    """
    check_job_compatibility(job)

    if TYPE_CHECKING:
        assert isinstance(job.device, QUANTINUUMDevice)

    if job.circuit.transpiled_circuit is None:
        tket_circuit = job.circuit.to_other_device(job.device)
    else:
        tket_circuit = job.circuit.transpiled_circuit
        if TYPE_CHECKING:
            assert isinstance(tket_circuit, tket_Circuit)

    if job.device == QUANTINUUMDevice.TKET_AER_SIMULATOR:
        from pytket.extensions.qiskit.backends.aer import AerBackend

        backend = AerBackend()
    elif job.device == QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR:
        from pytket.extensions.qiskit.backends.aer import AerStateBackend

        backend = AerStateBackend()
    elif job.device == QUANTINUUMDevice.TKET_QULACS_SIMULATOR:
        from pytket.extensions.qulacs.backends.qulacs_backend import QulacsBackend

        backend = QulacsBackend()
    else:
        raise ValueError(f"Local TKET device {job.device} is not handled.")

    optim_level = (
        0 if quantinuum_params is None else quantinuum_params.optimisation_level
    )
    compiled_circuit = backend.get_compiled_circuit(
        tket_circuit, optimisation_level=optim_level
    )

    if job.job_type == JobType.OBSERVABLE:
        return run_tket_observable(compiled_circuit, backend, job, quantinuum_params)

    n_shots = None if job.measure is None else job.measure.shots

    job.status = JobStatus.RUNNING
    backend_result = backend.run_circuit(compiled_circuit, n_shots=n_shots)
    return extract_result([backend_result], job)


def run_tket_observable(
    compiled_circuit: "tket_Circuit",
    backend: "Backend",
    job: Job,
    quantinuum_params: Optional[QuantinuumParams] = None,
) -> Result:
    """Execute an observable job using a local TKET backend.

    Args:
        compiled_circuit: TKET circuit compiled for the selected backend.
        backend: Local TKET backend used to execute the circuit.
        job: Observable job to execute.
        quantinuum_params: Quantinuum specific parameters used to configure
            observable grouping.

    Returns:
        An MPQP result containing the observable expectation values.
    """
    if TYPE_CHECKING:
        assert isinstance(job.measure, ExpectationMeasure)

    nb_shots = job.measure.shots

    if nb_shots == 0 or not job.measure.optimize_measurement:
        optimisation_strat = None
    else:
        if (
            quantinuum_params is not None
            and quantinuum_params.commutation_strategy is not None
        ):
            optimisation_strat = quantinuum_params.commutation_strategy
        else:
            from pytket.partition import PauliPartitionStrat

            optimisation_strat = (
                PauliPartitionStrat.NonConflictingSets
                if job.measure.commuting_type == CommutingTypes.QUBITWISE
                else PauliPartitionStrat.CommutingSets
            )

    from pytket.utils.expectations import get_operator_expectation_value

    expectation_values = {}
    errors = {}
    for i, o in enumerate(job.measure.observables):
        translated_obs = o.to_other_language(Language.TKET, targets=job.measure.targets)

        exp_value = get_operator_expectation_value(
            compiled_circuit, translated_obs, backend, nb_shots, optimisation_strat
        ).real

        expectation_values.update(
            {f"observable_{i}" if o.label is None else o.label: exp_value}
        )

        variance = (1.0 - exp_value**2) / job.measure.shots if nb_shots != 0 else 0.0

        errors.update({f"observable_{i}" if o.label is None else o.label: variance})

    if len(expectation_values) == 1:
        return Result(
            job,
            next(iter(expectation_values.values())),
            next(iter(errors.values())),
            shots=job.measure.shots,
        )

    return Result(job, expectation_values, errors, shots=job.measure.shots)


def run_nexus_remote(job: Job, quantinuum_params: Optional[QuantinuumParams] = None):
    """Submit a job to Quantinuum Nexus, wait for it to complete, and return
    its result.

    Args:
        job: Job to submit to a remote Quantinuum Nexus device.
        quantinuum_params: Quantinuum specific parameters used to configure
            circuit compilation.

    Returns:
        The result after submission and execution of the job.
    """

    try:
        _, execute_job_ref = submit_job_nexus(job, quantinuum_params)

        backend_results = fetch_nexus_results(execute_job_ref)

        return extract_result(backend_results, job)

    except Exception as error:
        job.status = JobStatus.ERROR
        job.status_message = str(error)
        raise error


def fetch_nexus_results(execute_job_ref: "ExecuteJobRef") -> list["BackendResult"]:
    """Wait for a Nexus execution job and retrieve its backend results.

    Args:
        execute_job_ref: Reference to the Nexus execution job.

    Returns:
        The TKET backend results returned by Nexus.

    Raises:
        RuntimeError: If the execution finishes without returning a result.
    """
    import qnexus as qnx

    execution_status = qnx.jobs.wait_for(execute_job_ref)
    result_refs = qnx.jobs.results(execute_job_ref)

    if not result_refs:
        status = execution_status.status.value
        raise RuntimeError(
            f"Quantinuum Nexus execution job '{execute_job_ref.id}' finished "
            f"with status '{status}', but no result was returned."
        )

    return [  # pyright: ignore[reportReturnType]
        ref.download_result()  # pyright: ignore[reportAttributeAccessIssue]
        for ref in result_refs
    ]


def submit_job_nexus(
    job: Job, provider_params: Optional[QuantinuumParams] = None
) -> tuple[str, "ExecuteJobRef"]:
    """Submit a job to a supported Quantinuum Nexus backend.

    Args:
        job: Job to submit.
        provider_params: Quantinuum specific parameters used to configure
            circuit compilation.

    Returns:
        The Nexus execution job ID and its reference.
    """
    check_job_compatibility(job)

    if job.job_type == JobType.OBSERVABLE:
        return submit_nexus_observable(job, provider_params)

    n_shots: int | list[None]
    if job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert job.measure is not None
        n_shots = job.measure.shots
    else:
        n_shots = [None]

    execute_job_ref = submit_circuits_to_nexus(
        job,
        [job.circuit],
        n_shots,
        name=f"mpqp-{job.job_type.name.lower()}-{job.device.value}",
        provider_params=provider_params,
    )

    if TYPE_CHECKING:
        assert job.id is not None

    return job.id, execute_job_ref


def submit_nexus_observable(
    job: Job, provider_params: Optional[QuantinuumParams] = None
) -> tuple[str, "ExecuteJobRef"]:
    """Submit a sampled observable as a single Nexus execution job.

    Exact observable jobs cannot be submitted directly to Nexus. For sampled
    observables, each qubit-wise commuting Pauli group is submitted as a circuit
    containing the required basis change followed by a measurement. The original
    MPQP ``Job`` is required to reconstruct the expectation values from the
    returned counts.

    Args:
        job: Observable job to submit.
        provider_params: Quantinuum specific parameters used to configure
            circuit compilation.

    Returns:
        The Nexus execution job ID and its reference.
    """
    if TYPE_CHECKING:
        assert isinstance(job.measure, ExpectationMeasure)

    circuit = job.circuit.without_measurements()
    n_shots: int | list[None]

    if job.measure.optimize_measurement:
        from warnings import warn

        warn(
            "Enabling `optimize_measurement` submits the observable as one or more "
            "`SAMPLE` circuits, with one circuit per Pauli group."
        )

        from mpqp.tools.pauli_grouping import find_qubitwise_rotations

        circuits = []
        grouping = job.measure.get_pauli_grouping()
        for group in grouping:
            sample_circuit = circuit + QCircuit(
                find_qubitwise_rotations(group, job.measure.targets)
            )
            sample_circuit.add(
                BasisMeasure(
                    job.measure.targets,
                    shots=job.measure.shots,
                )
            )
            circuits.append(sample_circuit)
        n_shots = job.measure.shots
    else:
        raise ValueError(
            "Cannot submit remote OBSERVABLE jobs on Nexus. Enable optimize_measurement=True in the "
            "ExpectationMeasure to let MPQP handle the Pauli grouping through a sampling job."
        )

    execute_job_ref = submit_circuits_to_nexus(  # TODO check that the ordering of the group is the same as circuits
        job,
        circuits,
        n_shots,
        name=f"mpqp-observable-{job.device.value}",
        description="mpqp:observable",
        provider_params=provider_params,
    )

    if TYPE_CHECKING:
        assert job.id is not None

    return job.id, execute_job_ref


def submit_circuits_to_nexus(
    job: Job,
    circuits: list[QCircuit],
    n_shots: int | list[None],
    name: str,
    description: str = "",
    provider_params: Optional[QuantinuumParams] = None,
) -> "ExecuteJobRef":
    """Prepare and submit one or more circuits to Quantinuum Nexus.

    The circuits are converted to TKET, uploaded to Nexus, and compiled for the
    selected backend. After compilation finishes, the compiled circuits are
    submitted together as a single Nexus execution job. Its ID is stored in
    the original MPQP job, which is then marked as running.

    For a sampled observable job, each circuit represents one Pauli group. The
    circuits are uploaded in grouping order, and the description of each
    uploaded circuit contains its group index.

    Args:
        job: Job to submit.
        circuits: Circuits to compile and execute, in submission order.
        n_shots: Number of shots to use, or ``[None]`` for a state-vector
            execution.
        name: Name to use for the Nexus job.
        description: Description of the Nexus execute job.
        provider_params: Quantinuum specific parameters used to configure
            circuit compilation.

    Returns:
        A reference to the Nexus execution job.

    Raises:
        RuntimeError: If compilation returns no circuit or a different number
            of circuits than expected.
    """

    import qnexus as qnx

    if TYPE_CHECKING:
        assert isinstance(job.device, QUANTINUUMDevice)
    tket_circuits = []
    for circuit in circuits:
        if job.circuit.transpiled_circuit is None:
            tket_circuits.append(circuit.to_other_device(job.device))
        else:
            tket_circuits.append(circuit.transpiled_circuit)

    backend_config = get_quantinuum_config(job.device)
    uploaded_circuit_refs = [
        qnx.circuits.upload(
            circuit=tket_circuit,
            name=f"{name}-circuit-{index}",
            description=(
                f"observable-group-{index}"
                if isinstance(job.measure, ExpectationMeasure)
                and job.measure.optimize_measurement
                else None
            ),
        )
        for index, tket_circuit in enumerate(tket_circuits)
    ]

    compile_job_ref = qnx.start_compile_job(
        programs=uploaded_circuit_refs,
        backend_config=backend_config,
        optimisation_level=(
            0 if provider_params is None else provider_params.optimisation_level
        ),
        name=f"{name}-compilation-job",
    )

    compilation_status = qnx.jobs.wait_for(compile_job_ref)

    compilation_result_refs = qnx.jobs.results(compile_job_ref)
    if not compilation_result_refs:
        status = compilation_status.status.value
        raise RuntimeError(
            f"Quantinuum Nexus compilation job '{compile_job_ref.id}' finished "
            f"with status '{status}', but no compiled circuit was returned."
        )

    compiled_circuit_refs = []
    for compilation_result_ref in compilation_result_refs:
        if TYPE_CHECKING:
            assert isinstance(compilation_result_ref, CompilationResultRef)
        compiled_circuit_ref = compilation_result_ref.get_output()
        if TYPE_CHECKING:
            assert isinstance(compiled_circuit_ref, CircuitRef)
        compiled_circuit_refs.append(compiled_circuit_ref)

    if len(compiled_circuit_refs) != len(circuits):
        raise RuntimeError(
            f"Quantinuum Nexus compiled {len(compiled_circuit_refs)} circuits, "
            f"but {len(circuits)} were submitted."
        )

    job.status = JobStatus.RUNNING
    execute_job_ref = qnx.start_execute_job(
        programs=compiled_circuit_refs,
        backend_config=backend_config,
        n_shots=n_shots,
        name=f"{name}-execution-job",
        description=description,
    )
    job.id = str(execute_job_ref.id)

    return execute_job_ref


def extract_remote_observable_grouped_result(
    backend_results: list["BackendResult"], job: Job
) -> Result:
    """Build an MPQP observable result from the counts returned for each Pauli
    group.

    Args:
        backend_results: TKET backend results returned by Nexus for the
            measurement circuit of each Pauli group.
        job: Original MPQP observable job.

    Returns:
        The reconstructed MPQP observable result.

    Raises:
        ValueError: If the number of backend results does not match the number
            of Pauli groups.
    """

    from mpqp.tools.pauli_grouping import pauli_monomial_eigenvalues

    if TYPE_CHECKING:
        assert isinstance(job.measure, ExpectationMeasure)

    job.status = JobStatus.DONE

    grouping = job.measure.get_pauli_grouping()

    eigenvalues = [
        {monomial.name: pauli_monomial_eigenvalues(monomial) for monomial in group}
        for group in grouping
    ]
    # TODO: improve this, 1. compute eigenvalues with a method in the monomial ?
    #  2. store it in an attribute so we don't recompute that at each iteration ?

    if len(backend_results) != len(grouping):
        raise ValueError(
            "The number of results returned for an `OBSERVABLE` job must match "
            "the number of Pauli groups. Ensure that submission and retrieval use "
            "the same observables and grouping method."
        )

    exp_values, errors = {}, {}

    if job.measure.nb_qubits != job.circuit.nb_qubits:
        warn(
            "Observable result extraction for partial targets is not fully "
            "supported. Counts currently cover the complete circuit register, "
            "which may produce an incorrect mapping between measured bits and "
            "circuit qubits.",
            NumberQubitsWarning,
        )
        # TODO: implement when we precise the targets, the mapping of the counts and basis state indices can be wrong.

    # TODO: Keep the received shot count for each Pauli group instead of
    # keeping only the count from the last group.
    received_shots = job.measure.shots
    # TODO: Test that circuits and results are ordered in the same way,
    # as this is critical for Pauli groups.
    for index, backend_result in enumerate(backend_results):
        raw_counts = backend_result.get_counts()
        received_shots = sum(raw_counts.values())
        if received_shots != job.measure.shots:
            warn(
                f"Received number of shots is different {received_shots} from given number of shots {job.measure.shots}. "
                f"We will proceed with the received number of shots instead.",
                ModifiedShotsNumberWarning,
            )
        length = 2**job.measure.nb_qubits
        sorted_values: list[float] = []
        for i in range(length):
            binary_state = f"{bin(i)[2:].zfill(len(bin(length)) - 3)}"
            tket_binary = tuple(int(b) for b in binary_state)
            if tket_binary in raw_counts:
                sorted_values.append(raw_counts[tket_binary].real / received_shots)
            else:
                sorted_values.append(0)
        for name, eigenvalue in eigenvalues[index].items():
            expectation_value: float = np.dot(
                eigenvalue,
                np.array(sorted_values, dtype=np.float64),
            )
            exp_values[name] = expectation_value

    result_dict = {}
    for i, obs in enumerate(job.measure.observables):
        string = obs.pauli_string
        local: float = 0
        for monoms in string.monomials:
            if TYPE_CHECKING:
                assert isinstance(monoms.coef, (int, float))
            local += exp_values[monoms.name] * monoms.coef
        result_dict.update(
            {f"observable_{i}" if obs.label is None else obs.label: local}
        )
        variance = (1.0 - local**2) / received_shots
        # FIXME the variance of an observable is not really the variance of a single monomial, coefs play a role
        errors.update({f"observable_{i}" if obs.label is None else obs.label: variance})

    if len(result_dict) == 1:
        return Result(
            job,
            next(iter(result_dict.values())),
            next(iter(errors.values())),
            shots=received_shots,
        )

    return Result(job, result_dict, errors, received_shots)


def extract_state_vector_result(
    amplitudes: list[Complex] | npt.NDArray[np.complex128],
    job: Job,
) -> Result:
    """Construct an MPQP result from Quantinuum state-vector amplitudes.

    Args:
        amplitudes: State-vector amplitudes returned by the execution backend.
        job: Original MPQP job used for the execution.

    Returns:
        A result containing the state vector and its probabilities.
    """
    state_vector = StateVector(amplitudes, nb_qubits=job.circuit.nb_qubits)
    job.status = JobStatus.DONE
    return Result(job, state_vector, 0, 0, g_phase_handling=False)


def extract_sample_result(
    raw_counts: "Counter[tuple[int, ...]]",
    job: Job,
) -> Result:
    """Construct an MPQP result from Quantinuum sample counts.

    Args:
        raw_counts: Number of occurrences of each measured state.
        job: Original MPQP job used for the execution.

    Returns:
        A result containing the samples and their probabilities.
    """
    if job.measure is None:
        raise ValueError("Cannot extract samples without a measurement.")

    samples = [
        Sample(
            bin_str="".join(str(bit) for bit in outcome),
            nb_qubits=job.circuit.nb_qubits,
            count=int(count),
        )
        for outcome, count in raw_counts.items()
    ]

    job.status = JobStatus.DONE
    return Result(job, samples, None, job.measure.shots)


def extract_result(backend_results: list["BackendResult"], job: Job) -> Result:
    """Construct an MPQP result from backend execution results.

    Args:
        backend_results: TKET results returned by a local backend or retrieved
            from Quantinuum Nexus.
        job: Original MPQP job. It provides the job type, circuit,
            measurement, and target device required to construct the result.

    Returns:
        An MPQP result constructed from the backend results.
    """
    if job.job_type == JobType.STATE_VECTOR:
        return extract_state_vector_result(backend_results[0].get_state(), job)
    if job.job_type == JobType.SAMPLE:
        return extract_sample_result(backend_results[0].get_counts(), job)
    if job.job_type == JobType.OBSERVABLE:
        return extract_remote_observable_grouped_result(backend_results, job)
    raise ValueError(f"Job type {job.job_type} not handled on {job.device}.")


def get_result_from_quantinuum_job_id(
    job_id: str,
    job: Optional[Job] = None,
) -> Result:
    """Retrieve and parse the result of a Quantinuum Nexus job.

    If the job is still running, wait until its execution is complete.

    Args:
        job_id: ID of the remote Quantinuum Nexus job.
        job: Original MPQP job used for submission. Required when retrieving
            an observable result.

    Returns:
        The result converted to our format.
    """
    import qnexus as qnx

    job_ref = qnx.jobs.get(id=job_id)

    if TYPE_CHECKING:
        assert isinstance(job_ref, ExecuteJobRef)

    if job is None and job_ref.annotations.description == "mpqp:observable":
        raise ValueError(
            "Retrieving a Quantinuum observable result requires the original MPQP `Job`."
        )

    backend_results = fetch_nexus_results(job_ref)

    if job is not None and job.job_type == JobType.OBSERVABLE:
        return extract_remote_observable_grouped_result(backend_results, job)

    backend_result = backend_results[0]
    if TYPE_CHECKING:
        assert isinstance(backend_result, BackendResult)

    backend_config = job_ref.backend_config_store
    if backend_config is None:
        raise ValueError(
            f"Quantinuum Nexus job '{job_id}' does not contain backend "
            "configuration information."
        )

    if isinstance(backend_config, qnx.AerStateConfig):
        amplitudes = backend_result.get_state()
        nb_qubits = int(math.log2(len(amplitudes)))
        job = Job(
            JobType.STATE_VECTOR,
            QCircuit(nb_qubits),
            QUANTINUUMDevice.NEXUS_AER_STATEVECTOR_SIMULATOR,
        )
        job.id = job_id
        return extract_state_vector_result(amplitudes, job)

    if isinstance(backend_config, qnx.AerConfig):
        device = QUANTINUUMDevice.NEXUS_AER_SIMULATOR
    elif isinstance(backend_config, qnx.QulacsConfig):
        device = QUANTINUUMDevice.NEXUS_QULACS_SIMULATOR
    elif isinstance(backend_config, qnx.QuantinuumConfig):
        device_name = backend_config.device_name
        try:
            device = QUANTINUUMDevice(device_name)
        except ValueError as error:
            raise ValueError(
                f"Quantinuum Nexus job '{job_id}' targeted unsupported device "
                f"'{device_name}'."
            ) from error
    else:
        raise ValueError(
            f"Quantinuum Nexus job '{job_id}' used unsupported backend "
            f"configuration '{type(backend_config).__name__}'."
        )

    if (
        device == QUANTINUUMDevice.NEXUS_QULACS_SIMULATOR
        and not backend_result.contains_measured_results
        and backend_result.contains_state_results
    ):
        amplitudes = backend_result.get_state()
        nb_qubits = int(math.log2(len(amplitudes)))
        job = Job(JobType.STATE_VECTOR, QCircuit(nb_qubits), device)
        job.id = job_id
        return extract_state_vector_result(amplitudes, job)

    raw_counts = backend_result.get_counts()
    if not raw_counts:
        raise ValueError(f"Quantinuum Nexus job '{job_id}' returned no sample counts.")

    nb_qubits = len(list(raw_counts)[0])
    shots = sum(raw_counts.values())
    circuit = QCircuit(
        [BasisMeasure(list(range(nb_qubits)), shots=shots)],
        nb_qubits=nb_qubits,
    )
    job = Job(JobType.SAMPLE, circuit, device)
    job.id = job_id

    return extract_sample_result(raw_counts, job)
