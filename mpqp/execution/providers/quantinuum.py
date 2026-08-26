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
from mpqp.tools.errors import DeviceJobIncompatibleError

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
            raise DeviceJobIncompatibleError(
                "`job` must correspond to a `QUANTINUUMDevice`, but corresponds "
                f"to a {job.device} instead."
            )

        if not job.device.is_remote():
            return run_tket_local(job)
        if job.job_type == JobType.OBSERVABLE:
            check_job_compatibility(job)
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


def check_job_compatibility(job: Job) -> None:
    """Checks whether the job in parameter has coherent and compatible
    attributes.

    Args:
        job: Job for which we want to check compatibility.

    Raises:
        DeviceJobIncompatibleError: If there is a mismatch between information
            contained in the job (measure and job_type, device and job_type,
            etc...).
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

    if (
        job.job_type == JobType.OBSERVABLE
        and isinstance(job.measure, ExpectationMeasure)
        and job.measure.shots == 0
    ):
        supports_exact_observable = (
            job.device.supports_state_vector()
            if job.device.is_remote()
            else job.device.supports_observable_ideal()
        )
        if not supports_exact_observable:
            raise DeviceJobIncompatibleError(
                f"{job.device} requires a positive number of shots for "
                "observable jobs."
            )

    if (
        job.job_type == JobType.OBSERVABLE
        and isinstance(job.measure, ExpectationMeasure)
        and job.measure.shots > 0
        and (not job.device.supports_observable() or not job.device.supports_samples())
    ):
        raise DeviceJobIncompatibleError(
            f"{job.device} does not support sampled observable jobs."
        )

    if (
        job.job_type == JobType.OBSERVABLE
        and isinstance(job.measure, ExpectationMeasure)
        and job.measure.shots > 0
        and job.measure.commuting_type != CommutingTypes.QUBITWISE
    ):
        raise DeviceJobIncompatibleError(
            "Quantinuum sampld observable jobs currently require qubit-wise "
            "commuting Pauli grouping."
        )


def run_tket_local(job: Job) -> Result:
    """Execute a job using a local TKET backend.

    Args:
        job: Job targeting a local TKET device.

    Returns:
        The result after local compilation and execution.
    """
    check_job_compatibility(job)
    if TYPE_CHECKING:
        assert isinstance(job.device, QUANTINUUMDevice)
    if job.device.is_remote():
        raise ValueError("The job must target a local TKET device.")

    if job.job_type == JobType.OBSERVABLE:
        if TYPE_CHECKING:
            assert isinstance(job.measure, ExpectationMeasure)
        if job.measure.shots > 0:
            return run_quantinuum_observable(job)

    if job.circuit.transpiled_circuit is None:
        tket_circuit = job.circuit.to_other_device(job.device)
    else:
        from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

        qiskit_circuit = job.circuit.transpiled_circuit
        if TYPE_CHECKING:
            assert isinstance(qiskit_circuit, QuantumCircuit)
        tket_circuit = qiskit_to_tk(qiskit_circuit)

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
    if TYPE_CHECKING:
        assert isinstance(job.device, QUANTINUUMDevice)
        assert isinstance(job.measure, ExpectationMeasure)

    observable_labels = job.measure.observables_labels

    circuit = job.circuit.without_measurements()

    if job.measure.shots == 0:
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

    from mpqp.tools.pauli_grouping import find_qubitwise_rotations

    grouped_probabilities = []
    grouping = job.measure.get_pauli_grouping()
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

    return extract_sampled_observable_result(job, grouped_probabilities)


def submit_job_quantinuum(job: Job) -> tuple[str, "ExecuteJobRef"]:
    """Submit a job to a supported Quantinuum Nexus backend."""
    if job.job_type == JobType.OBSERVABLE:
        return submit_quantinuum_observable(job)

    check_job_compatibility(job)
    n_shots: int | list[None]
    if job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert job.measure is not None
        n_shots = job.measure.shots
    else:
        n_shots = [None]

    return submit_circuits_to_nexus(
        job,
        [job.circuit],
        n_shots,
        name=f"mpqp-{job.job_type.name.lower()}-{job.device.value}",
    )


def submit_quantinuum_observable(job: Job) -> tuple[str, "ExecuteJobRef"]:
    """Submit an observable as one Nexus execution job.

    Exact observables submit one state-vector circuit. For sampled observables,
    each qubit-wise commuting Pauli group is submitted as a circuit containing
    the required basis change followed by a measurement. The original
    MPQP `Job` is required to reconstruct the expectation value from the
    returned states or counts.
    """
    check_job_compatibility(job)
    if TYPE_CHECKING:
        assert isinstance(job.device, QUANTINUUMDevice)
        assert isinstance(job.measure, ExpectationMeasure)
    if not job.device.is_remote():
        raise ValueError("Observable submission requires a remote Quantinuum device.")

    circuit = job.circuit.without_measurements()
    n_shots: int | list[None]
    if job.measure.shots == 0:
        circuits = [circuit]
        n_shots = [None]
    else:
        from mpqp.tools.pauli_grouping import find_qubitwise_rotations

        circuits = []
        for group in job.measure.get_pauli_grouping():
            sample_circuit = circuit + QCircuit(find_qubitwise_rotations(group))
            sample_circuit.add(
                BasisMeasure(
                    list(range(job.circuit.nb_qubits)),
                    shots=job.measure.shots,
                )
            )
            circuits.append(sample_circuit)
        n_shots = job.measure.shots

    return submit_circuits_to_nexus(
        job,
        circuits,
        n_shots,
        name=f"mpqp-observable-{job.device.value}",
        description="mpqp:observable",
    )


def submit_circuits_to_nexus(
    job: Job,
    circuits: list[QCircuit],
    n_shots: int | list[None],
    name: str,
    description: str = "",
) -> tuple[str, "ExecuteJobRef"]:
    """Upload, compile, and submit several circuits as one Nexus execute job."""

    if not isinstance(job.device, QUANTINUUMDevice) or not job.device.is_remote():
        raise ValueError("Nexus submission requires a remote Quantinuum device.")

    import qnexus as qnx
    from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

    tket_circuits = []
    for circuit in circuits:
        if circuit.transpiled_circuit is None:
            tket_circuits.append(circuit.to_other_device(job.device))
        else:
            qiskit_circuit = circuit.transpiled_circuit
            if TYPE_CHECKING:
                assert isinstance(qiskit_circuit, QuantumCircuit)
            tket_circuits.append(qiskit_to_tk(qiskit_circuit))

    backend_config = get_quantinuum_config(job.device)
    uploaded_circuit_refs = [
        qnx.circuits.upload(
            circuit=tket_circuit,
            name=f"{name}-circuit-{index}",
        )
        for index, tket_circuit in enumerate(tket_circuits)
    ]

    compile_job_ref = qnx.start_compile_job(
        programs=uploaded_circuit_refs,
        backend_config=backend_config,
        optimisation_level=0,
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

    return job.id, execute_job_ref


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


def extract_sampled_observable_result(
    job: Job,
    grouped_probabilities: list[npt.NDArray[np.float64]],
) -> Result:
    """Reconstruct a sampled observable result from one probability vector per
    Pauli group by combining the measured probabilities with the eigenvalues and
    coefficients of the observable's Pauli terms.
    """
    if not isinstance(job.measure, ExpectationMeasure):
        raise ValueError("Observable jobs must have an `ExpectationMeasure`.")

    grouping = job.measure.get_pauli_grouping()
    if len(grouped_probabilities) != len(grouping):
        raise ValueError(
            "The number of Quantinuum observable results "
            f"({len(grouped_probabilities)}) does not match the number of "
            f"submitted Pauli groups ({len(grouping)})."
        )

    from mpqp.tools.pauli_grouping import pauli_monomial_eigenvalues

    eigenvalues = [
        {monomial.name: pauli_monomial_eigenvalues(monomial) for monomial in group}
        for group in grouping
    ]

    expectation_values: dict[str, float] = {}
    errors: dict[str, float] = {}
    for label, observable in zip(
        job.measure.observables_labels,
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
        errors[label] = variance

    return extract_observable_result(
        job,
        expectation_values,
        errors,
        job.measure.shots,
    )


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


def get_result_from_quantinuum_job_id(
    job_id: str,
    job: Job | None = None,
) -> Result:
    """Retrieve and parse the result of a Quantinuum Nexus job.

    If the job is still running, wait until its execution is complete.

    Args:
        job_id: id of the remote Quantinuum Nexus job.
        job: Original MPQP job used for submission. Required when retrieving
        an observable result.

    Returns:
        The result converted to our format.
    """
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

    if job is not None and job.job_type == JobType.OBSERVABLE:
        if not isinstance(job.measure, ExpectationMeasure):
            raise ValueError("Observable jobs must have an `ExpectationMeasure`.")
        job.id = job_id
        if job.measure.shots == 0:
            if len(result_refs) != 1:
                raise ValueError(
                    "An exact Quantinuum observable job must return one state-vector "
                    f"result, but returned {len(result_refs)}."
                )
            result_ref = result_refs[0]
            if TYPE_CHECKING:
                assert isinstance(result_ref, ExecutionResultRef)
            backend_result = result_ref.download_result()
            if TYPE_CHECKING:
                assert isinstance(backend_result, BackendResult)
            state = backend_result.get_state()
            expectation_values = {
                label: float(np.vdot(state, observable.matrix @ state).real)
                for label, observable in zip(
                    job.measure.observables_labels,
                    job.measure.observables,
                )
            }
            return extract_observable_result(job, expectation_values, 0.0, 0)

        grouped_probabilities = []
        for result_ref in result_refs:
            if TYPE_CHECKING:
                assert isinstance(result_ref, ExecutionResultRef)
            backend_result = result_ref.download_result()
            if TYPE_CHECKING:
                assert isinstance(backend_result, BackendResult)
            raw_counts = backend_result.get_counts()
            shots = sum(raw_counts.values())
            if shots == 0:
                raise ValueError(
                    f"Quantinuum observable job '{job_id}' returned no sample counts."
                )
            probabilities = np.zeros(2**job.circuit.nb_qubits, dtype=np.float64)
            for outcome, count in raw_counts.items():
                index = int("".join(str(bit) for bit in outcome), 2)
                probabilities[index] = count / shots
            grouped_probabilities.append(probabilities)
        return extract_sampled_observable_result(job, grouped_probabilities)

    if job is None and job_ref.annotations.description == "mpqp:observable":
        raise ValueError(
            "Retrieving a Quantinuum observable result requires the original MPQP `Job`."
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
