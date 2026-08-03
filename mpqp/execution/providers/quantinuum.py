from __future__ import annotations

import math
from collections import Counter
from numbers import Complex, Real
from typing import TYPE_CHECKING

import numpy as np

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure, ExpectationMeasure
from mpqp.core.instruction.measurement.pauli_string import CommutingTypes
from mpqp.core.languages import Language
from mpqp.execution.connection.quantinuum_connection import get_quantinuum_config
from mpqp.execution.devices import QUANTINUUMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.result import Result, Sample, StateVector

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from pytket import Circuit as TKETCircuit
    from pytket.backends.backend import Backend
    from pytket.backends.backendresult import BackendResult
    from pytket.utils.operators import QubitPauliOperator
    from qiskit import QuantumCircuit
    from qnexus.models.references import (
        CircuitRef,
        CompilationResultRef,
        ExecuteJobRef,
        ExecutionResultRef,
    )


def run_quantinuum(job: Job) -> Result:
    """Execute the job on the selected Quantinuum device (local or remote),
    wait until execution is complete, and return the result.

    Args:
        job: Job to execute. It must target a
            :class:`mpqp.execution.devices.QUANTINUUMDevice`.

    Returns:
        The result of the job.

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    try:
        if not isinstance(job.device, QUANTINUUMDevice):
            raise ValueError(
                "`job` must correspond to a `QUANTINUUMDevice`, but corresponds "
                f"to a {job.device} instead."
            )
        if not job.device.is_remote():
            return run_tket_local(job)
        if job.job_type == JobType.OBSERVABLE:
            return run_quantinuum_observable(job)

        _, execute_job_ref = submit_job_quantinuum(job)

        import qnexus as qnx

        execution_status = qnx.jobs.wait_for(execute_job_ref)
        result_refs = qnx.jobs.results(execute_job_ref)
        if not result_refs:
            status = execution_status.status.value
            raise RuntimeError(
                f"Quantinuum Nexus execution job '{execute_job_ref.id}' finished "
                f"with status '{status}', but no result was returned."
            )

        result_ref = result_refs[0]
        if TYPE_CHECKING:
            assert isinstance(result_ref, ExecutionResultRef)

        backend_result = result_ref.download_result()
        if TYPE_CHECKING:
            assert isinstance(backend_result, BackendResult)

        return extract_result(backend_result, job)
    except Exception as error:
        job.status = JobStatus.ERROR
        job.status_message = str(error)
        raise


def run_tket_local(job: Job) -> Result:
    """Execute a job using a local TKET backend.

    Args:
        job: Job targeting a local TKET device.

    Returns:
        The result after local compilation and execution.
    """
    if not isinstance(job.device, QUANTINUUMDevice) or job.device.is_remote():
        raise ValueError("The job must target a local TKET device.")

    if job.job_type == JobType.SAMPLE:
        if not job.device.supports_samples():
            raise ValueError(
                f"{job.device} does not support `SAMPLE` jobs. Use "
                f"{QUANTINUUMDevice.TKET_AER_SIMULATOR} instead."
            )
        if not isinstance(job.measure, BasisMeasure):
            raise ValueError("Sample jobs must have a `BasisMeasure`.")
    elif job.job_type == JobType.STATE_VECTOR:
        if not job.device.supports_state_vector():
            raise ValueError(
                f"{job.device} does not support `STATE_VECTOR` jobs. Use "
                f"{QUANTINUUMDevice.TKET_AER_STATE_SIMULATOR} instead."
            )
    elif job.job_type == JobType.OBSERVABLE:
        if not isinstance(job.measure, ExpectationMeasure):
            raise ValueError("Observable jobs must have an `ExpectationMeasure`.")
        if job.measure.shots > 0:
            return run_quantinuum_observable(job)
        if not job.device.supports_observable_ideal():
            raise ValueError(
                f"{job.device} does not support observable jobs without sampling."
            )
    else:
        raise ValueError(f"Job type {job.job_type} not handled on {job.device}.")

    from pytket.extensions.qiskit.backends.aer import AerBackend, AerStateBackend
    from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk
    from pytket.extensions.qulacs.backends.qulacs_backend import QulacsBackend

    if job.circuit.transpiled_circuit is None:
        tket_circuit = job.circuit.to_other_device(job.device)
    else:
        qiskit_circuit = job.circuit.transpiled_circuit
        if TYPE_CHECKING:
            assert isinstance(qiskit_circuit, QuantumCircuit)
        tket_circuit = qiskit_to_tk(qiskit_circuit)

    if job.device == QUANTINUUMDevice.TKET_AER_SIMULATOR:
        backend = AerBackend()
    elif job.device == QUANTINUUMDevice.TKET_AER_STATE_SIMULATOR:
        backend = AerStateBackend()
    elif job.device == QUANTINUUMDevice.TKET_QULACS_SIMULATOR:
        backend = QulacsBackend()
    else:
        raise ValueError(f"Local TKET device {job.device} is not handled.")

    compiled_circuit = backend.get_compiled_circuit(
        tket_circuit,
        optimisation_level=0,
    )
    if job.job_type == JobType.OBSERVABLE:
        return run_tket_observable(job, compiled_circuit, backend)

    if job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert isinstance(job.measure, BasisMeasure)
        n_shots = job.measure.shots
    else:
        n_shots = None

    job.status = JobStatus.RUNNING
    backend_result = backend.run_circuit(compiled_circuit, n_shots=n_shots)
    return extract_result(backend_result, job)


def run_tket_observable(
    job: Job,
    tket_circuit: "TKETCircuit",
    backend: "Backend",
) -> Result:
    """Return the result of an exact `OBSERVABLE` job using the built in
    observable evaluation of a local TKET backend.

    The backend computes the expectation values directly. This function should
    be called by :func:`run_tket_local` for observable jobs without sampling, it
    is not intended for remote Nexus jobs.

    Args:
        job: Job to execute.
        tket_circuit: Compiled TKET circuit on which the observables are evaluated.
        backend: Local TKET backend used to evaluate the observables.

    Returns:
        A result containing the exact expectation values of the observables.
    """
    if not isinstance(job.measure, ExpectationMeasure):
        raise ValueError("Observable jobs must have an `ExpectationMeasure`.")

    job.status = JobStatus.RUNNING
    expectation_values: dict[str, float] = {}
    for label, observable in zip(
        job.measure.observables_labels,
        job.measure.observables,
    ):
        operator = observable.to_other_language(Language.TKET)
        if TYPE_CHECKING:
            assert isinstance(operator, QubitPauliOperator)
        expectation_value = backend.get_operator_expectation_value(
            tket_circuit,
            operator,
        )
        expectation_values[label] = float(expectation_value.real)

    return extract_observable_result(job, expectation_values, 0.0, 0)


def run_quantinuum_observable(job: Job) -> Result:
    """Execute an observable job using a supported Quantinuum backend.

    Exact observables are computed from a state vector when `shots=0`.
    When `shots>0`, observables are estimated from measurement results.

    Args:
        job: Job to execute.

    Returns:
        A result containing the expectation values of the observables.
    """
    if not isinstance(job.device, QUANTINUUMDevice):
        raise ValueError(
            "`job` must correspond to a `QUANTINUUMDevice`, but corresponds to "
            f"a {job.device} instead."
        )
    if not isinstance(job.measure, ExpectationMeasure):
        raise ValueError("Observable jobs must have an `ExpectationMeasure`.")

    observable_labels = job.measure.observables_labels

    circuit = job.circuit.without_measurements()

    if job.measure.shots == 0:
        if not job.device.supports_state_vector():
            raise ValueError(
                f"{job.device} requires a positive number of shots for "
                "observable jobs."
            )
        state_job = Job(
            JobType.STATE_VECTOR,
            circuit,
            job.device,
        )
        state_result = run_quantinuum(state_job)
        job.id = state_job.id
        job.status = state_job.status

        expectation_values: dict[str, float] = {}
        for label, observable in zip(
            observable_labels,
            job.measure.observables,
        ):
            expectation_values[label] = float(
                np.vdot(
                    state_result.amplitudes,
                    observable.matrix @ state_result.amplitudes,
                ).real
            )
        return extract_observable_result(job, expectation_values, 0.0, 0)

    if not job.device.supports_samples():
        raise ValueError(
            f"{job.device} does not support sampled observable jobs. Set `shots=0` "
            "to evaluate the observable without sampling."
        )
    if job.measure.commuting_type != CommutingTypes.QUBITWISE:
        raise NotImplementedError(
            "Quantinuum sample derived observable jobs currently support only "
            "qubit-wise commuting Pauli grouping."
        )

    from mpqp.tools.pauli_grouping import (
        find_qubitwise_rotations,
        pauli_monomial_eigenvalues,
    )

    grouped_probabilities = []
    grouping = job.measure.get_pauli_grouping()
    eigenvalues = [
        {monomial.name: pauli_monomial_eigenvalues(monomial) for monomial in group}
        for group in grouping
    ]
    for group in grouping:
        pre_measure = QCircuit(find_qubitwise_rotations(group))
        sample_circuit = circuit + pre_measure
        sample_circuit.add(
            BasisMeasure(
                list(range(job.circuit.nb_qubits)),
                shots=job.measure.shots,
            )
        )
        sample_job = Job(JobType.SAMPLE, sample_circuit, job.device)
        sample_result = run_quantinuum(sample_job)
        job.id = sample_job.id
        grouped_probabilities.append(sample_result.probabilities)

    expectation_values: dict[str, float] = {}
    errors: dict[str, float] = {}
    for label, observable in zip(
        observable_labels,
        job.measure.observables,
    ):
        expectation_value = 0.0
        variance = 0.0
        monomials = {
            monomial.name: monomial for monomial in observable.pauli_string.monomials
        }

        for group, group_eigenvalues, probabilities in zip(
            grouping,
            eigenvalues,
            grouped_probabilities,
        ):
            weighted_eigenvalues = np.zeros_like(probabilities)
            for monomial in group:
                observable_monomial = monomials.get(monomial.name)
                if observable_monomial is None:
                    continue
                if TYPE_CHECKING:
                    assert isinstance(observable_monomial.coef, Real)
                weighted_eigenvalues += (
                    float(observable_monomial.coef) * group_eigenvalues[monomial.name]
                )

            group_expectation = float(np.dot(weighted_eigenvalues, probabilities))
            expectation_value += group_expectation
            variance += (
                max(
                    0.0,
                    float(np.dot(weighted_eigenvalues**2, probabilities))
                    - group_expectation**2,
                )
                / job.measure.shots
            )

        expectation_values[label] = expectation_value
        errors[label] = math.sqrt(variance)

    return extract_observable_result(
        job,
        expectation_values,
        errors,
        job.measure.shots,
    )


def extract_observable_result(
    job: Job,
    expectation_values: dict[str, float],
    errors: float | dict[str, float],
    shots: int,
) -> Result:
    """Construct an MPQP result from Quantinuum expectation values."""
    job.status = JobStatus.DONE
    if len(expectation_values) == 1:
        label = list(expectation_values)[0]
        expectation_value = expectation_values[label]
        error = errors[label] if isinstance(errors, dict) else errors
        return Result(job, expectation_value, error, shots)
    if not isinstance(errors, dict):
        errors = dict.fromkeys(expectation_values, errors)
    return Result(job, expectation_values, errors, shots)


def submit_job_quantinuum(job: Job) -> tuple[str, "ExecuteJobRef"]:
    """Submit a job to a supported Quantinuum Nexus backend."""
    if not isinstance(job.device, QUANTINUUMDevice):
        raise ValueError(
            "`job` must correspond to a `QUANTINUUMDevice`, but corresponds to "
            f"a {job.device} instead."
        )

    if job.job_type == JobType.SAMPLE:
        if not job.device.supports_samples():
            raise ValueError(f"{job.device} does not support `SAMPLE` jobs.")
        if job.measure is None:
            raise ValueError("Sample jobs must have a measurement.")
    elif job.job_type == JobType.STATE_VECTOR:
        if not job.device.supports_state_vector():
            raise ValueError(f"{job.device} does not support `STATE_VECTOR` jobs.")
    else:
        raise ValueError(f"Job type {job.job_type} not handled on {job.device}.")

    import qnexus as qnx
    from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

    if job.circuit.transpiled_circuit is None:
        tket_circuit = job.circuit.to_other_device(job.device)
    else:
        qiskit_circuit = job.circuit.transpiled_circuit
        if TYPE_CHECKING:
            assert isinstance(qiskit_circuit, QuantumCircuit)
        tket_circuit = qiskit_to_tk(qiskit_circuit)

    backend_config = get_quantinuum_config(job.device)

    name = f"mpqp-{job.job_type.name.lower()}-{job.device.value}"

    uploaded_circuit_ref = qnx.circuits.upload(
        circuit=tket_circuit,
        name=f"{name}-circuit",
    )

    compile_job_ref = qnx.start_compile_job(
        programs=[uploaded_circuit_ref],
        backend_config=backend_config,
        optimisation_level=0,
        name=f"{name}-compilation-job",
    )

    compilation_status = qnx.jobs.wait_for(compile_job_ref)

    compiled_circuit_refs = qnx.jobs.results(compile_job_ref)
    if not compiled_circuit_refs:
        status = compilation_status.status.value
        raise RuntimeError(
            f"Quantinuum Nexus compilation job '{compile_job_ref.id}' finished "
            f"with status '{status}', but no compiled circuit was returned."
        )

    compilation_result_ref = compiled_circuit_refs[0]
    if TYPE_CHECKING:
        assert isinstance(compilation_result_ref, CompilationResultRef)
    compiled_circuit_ref = compilation_result_ref.get_output()
    if TYPE_CHECKING:
        assert isinstance(compiled_circuit_ref, CircuitRef)

    job.status = JobStatus.RUNNING

    n_shots: list[int] | list[None]
    if job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert job.measure is not None
        n_shots = [job.measure.shots]
    else:
        n_shots = [None]

    execute_job_ref = qnx.start_execute_job(
        programs=[compiled_circuit_ref],
        backend_config=backend_config,
        n_shots=n_shots,
        name=f"{name}-execution-job",
    )

    job.id = str(execute_job_ref.id)

    return job.id, execute_job_ref


def extract_state_vector_result(
    amplitudes: "list[Complex] | npt.NDArray[np.complex128]",
    job: Job,
) -> Result:
    """Construct an MPQP result from Quantinuum state-vector amplitudes.

    Args:
        amplitudes: State-vector amplitudes returned by Quantinuum Nexus.
        job: Original MPQP job used for the execution.

    Returns:
        A result containing the state vector and its probabilities.
    """
    state_vector = StateVector(amplitudes, nb_qubits=job.circuit.nb_qubits)
    job.status = JobStatus.DONE
    return Result(job, state_vector, 0, 0)


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


def extract_result(backend_result: "BackendResult", job: Job) -> Result:
    """Construct a Result from a backend execution result.

    Args:
        backend_result: TKET result returned by a local backend or retrieved from
            Quantinuum Nexus.
        job: Original MPQP job used for the execution. It provides the job type,
            circuit, measurement, and target device required to construct the result

    Returns:
        The backend result converted to our format.
    """
    if job.job_type == JobType.STATE_VECTOR:
        return extract_state_vector_result(backend_result.get_state(), job)
    elif job.job_type == JobType.SAMPLE:
        return extract_sample_result(backend_result.get_counts(), job)
    else:
        raise ValueError(f"Job type {job.job_type} not handled on {job.device}.")


def get_result_from_quantinuum_job_id(job_id: str) -> Result:
    """Retrieve a Quantinuum Nexus job result and convert it to an MPQP result."""
    import qnexus as qnx

    job_ref = qnx.jobs.get(id=job_id)
    if TYPE_CHECKING:
        assert isinstance(job_ref, ExecuteJobRef)
    execution_status = qnx.jobs.wait_for(job_ref)
    result_refs = qnx.jobs.results(job_ref)
    if not result_refs:
        status = execution_status.status.value
        raise RuntimeError(
            f"Quantinuum Nexus execution job '{job_id}' finished with status "
            f"'{status}', but no result was returned."
        )

    result_ref = result_refs[0]
    if TYPE_CHECKING:
        assert isinstance(result_ref, ExecutionResultRef)

    backend_config = job_ref.backend_config_store
    if backend_config is None:
        raise ValueError(
            f"Quantinuum Nexus job '{job_id}' does not contain backend "
            "configuration information."
        )

    if isinstance(backend_config, qnx.AerStateConfig):
        backend_result = result_ref.download_result()
        if TYPE_CHECKING:
            assert isinstance(backend_result, BackendResult)
        amplitudes = backend_result.get_state()
        nb_qubits = int(math.log2(len(amplitudes)))
        job = Job(
            JobType.STATE_VECTOR,
            QCircuit(nb_qubits),
            QUANTINUUMDevice.NEXUS_AER_STATE_SIMULATOR,
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

    backend_result = result_ref.download_result()
    if TYPE_CHECKING:
        assert isinstance(backend_result, BackendResult)
    if (
        device == QUANTINUUMDevice.NEXUS_QULACS_SIMULATOR
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
