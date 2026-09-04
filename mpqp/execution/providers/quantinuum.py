from __future__ import annotations

import math
from collections import Counter
from numbers import Complex, Real
from typing import TYPE_CHECKING, Optional

import numpy as np

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure, ExpectationMeasure
from mpqp.core.instruction.measurement.pauli_string import CommutingTypes
from mpqp.core.languages import Language
from mpqp.execution.connection.quantinuum_connection import get_quantinuum_config
from mpqp.execution.devices import QUANTINUUMDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.providers.providers_params import TketParams
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.tools.errors import DeviceJobIncompatibleError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from mpqp.core.instruction.measurement.pauli_string import (
        CommutingTypes,
        PauliStringMonomial,
    )

    # from pytket import Circuit as TKETCircuit
    from pytket.backends.backend import Backend
    from pytket.backends.backendresult import BackendResult
    from qnexus.models.references import (
        CircuitRef,
        CompilationResultRef,
        ExecuteJobRef,
        ExecutionResultRef,
    )


def run_quantinuum(job: Job, provider_params: Optional[TketParams] = None) -> Result:
    """Executes the job on the selected Quantinuum device (local or remote),
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
    if not isinstance(job.device, QUANTINUUMDevice):
        raise DeviceJobIncompatibleError(
            "`job` must correspond to a `QUANTINUUMDevice`, but corresponds "
            f"to a {job.device} instead."
        )

    check_job_compatibility(job)
    try:
        if not job.device.is_remote():
            return run_tket_local(job, provider_params)

        _, execute_job_ref = submit_job_nexus(job, provider_params)

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

    if job.job_type == JobType.OBSERVABLE:
        if TYPE_CHECKING:
            assert isinstance(job.measure, ExpectationMeasure)
        if job.measure.shots == 0:
            if job.device.is_remote():
                supports_exact_observable = job.device.supports_state_vector()
            else:
                if job.measure.optimize_measurement:
                    supports_exact_observable = job.device.supports_state_vector()
                else:
                    supports_exact_observable = False
                supports_exact_observable |= job.device.supports_observable_ideal()

            if not supports_exact_observable:
                raise DeviceJobIncompatibleError(
                    f"{job.device} requires a positive number of shots for "
                    "observable jobs."
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
                    f"{job.device} does not support sampled observable jobs."
                )
        if (
            job.measure.shots > 0
            and job.measure.commuting_type != CommutingTypes.QUBITWISE
        ):
            raise DeviceJobIncompatibleError(
                "Quantinuum sampled observable jobs currently require qubit-wise "
                "commuting Pauli grouping."
            )


def run_tket_local(job: Job, provider_params: Optional[TketParams] = None) -> Result:
    """Execute a job using a local TKET backend.

    Args:
        job: Job targeting a local TKET device.

    Returns:
        The result after local compilation and execution.
    """
    if TYPE_CHECKING:
        assert isinstance(job.device, QUANTINUUMDevice)
    if job.device.is_remote():
        raise ValueError("The job must target a local TKET device.")

    if job.circuit.transpiled_circuit is None:
        tket_circuit = job.circuit.to_other_device(job.device)
    else:
        tket_circuit = job.circuit.transpiled_circuit
        if TYPE_CHECKING:
            from pytket.circuit import Circuit as tket_Circuit

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
    compiled_circuit = backend.get_compiled_circuit(
        tket_circuit,
        optimisation_level=(
            0 if provider_params is None else provider_params.optimisation_level
        ),
    )
    if job.job_type == JobType.OBSERVABLE:
        return run_quantinuum_observable(job, backend, provider_params)

    n_shots = None if job.measure is None else job.measure.shots

    job.status = JobStatus.RUNNING
    backend_result = backend.run_circuit(compiled_circuit, n_shots=n_shots)
    return extract_result(backend_result, job)


def run_quantinuum_observable(
    job: Job,
    backend: "Backend",
    provider_params: Optional[TketParams] = None,
) -> Result:
    """Execute an observable job using a supported Quantinuum backend.

    Args:
        job: Job to execute.

    Returns:
        A result containing the expectation values of the observables.
    """
    from pytket.utils.expectations import get_operator_expectation_value

    if TYPE_CHECKING:
        assert isinstance(job.measure, ExpectationMeasure)
    circuit = job.circuit.without_measurements().to_other_language(Language.TKET)

    circuit = backend.get_compiled_circuit(
        circuit,
        optimisation_level=(
            0 if provider_params is None else provider_params.optimisation_level
        ),
    )
    job_change_compatibility = (
        job.measure.shots == 0 and not job.device.supports_state_vector()
    ) or (job.measure.shots != 0 and not job.device.supports_samples())
    exp_values, errors = {}, {}
    if job.measure.optimize_measurement and job_change_compatibility:
        from warnings import warn

        warn(
            "MPQP's optimize_measurement changes the type of the Job to SAMPLE or STATE_VECTOR."
            f"Your chosen device:{job.device} is not compatible with it so this optimization won't be used here."
        )
    if job.measure.optimize_measurement and not job_change_compatibility:
        from mpqp.tools.pauli_grouping import (
            find_qubitwise_rotations,
            pauli_monomial_eigenvalues,
        )

        if job.measure.pre_transpiled is None:
            grouping = job.measure.get_pauli_grouping()
            pre_measure = [
                QCircuit(
                    find_qubitwise_rotations(group, job.measure.targets)
                    + (
                        [BasisMeasure(targets=job.measure.targets)]
                        if job.measure.shots != 0
                        else []
                    )
                )
                for group in grouping
            ]
            transpiled_pre_measures = [
                pre_m.to_other_language(Language.TKET) for pre_m in pre_measure
            ]
            eigenvalues = [
                {
                    monomial.name: pauli_monomial_eigenvalues(monomial)
                    for monomial in group
                }
                for group in grouping
            ]
        else:
            eigenvalues, transpiled_pre_measures = (
                job.measure.pre_transpiled
            )  # pyright: ignore[reportGeneralTypeIssues]

        expectation_values = {}
        # For each group, runs the circuit and store the computed exp_values
        for eigenv, pre_measure in zip(eigenvalues, transpiled_pre_measures):
            job.status = JobStatus.RUNNING

            cirq = circuit.copy()
            cirq.append(pre_measure)
            local_result = backend.run_circuit(
                cirq, n_shots=job.measure.shots if job.measure.shots != 0 else None
            )
            # Runs a StateVector
            # TODO: Find a way to return a statevector only on parts of the circuit
            # if the observable doesn't cover the whole circuit we'll get diff results here
            # if at all...
            if job.measure.shots == 0:
                values = local_result.get_state()
                sorted_values = []
                for i in range(len(values)):
                    sorted_values.append(float(np.abs(values[i]) ** 2))
            else:
                length = 2**job.measure.nb_qubits
                measurements = local_result.get_counts()
                sorted_values: list[float] = []
                for i in range(length):
                    binary_state = f"{bin(i)[2:].zfill(len(bin(length))- 3)}"
                    tket_binary = tuple(int(b) for b in binary_state)
                    if tket_binary in measurements:
                        sorted_values.append(
                            measurements[tket_binary].real / job.measure.shots
                        )
                    else:
                        sorted_values.append(0)

            for name, eigenvalue in eigenv.items():
                expectation_value: float = np.dot(
                    eigenvalue,
                    np.array(sorted_values, dtype=np.float64),
                )
                expectation_values[name] = expectation_value

        # Put the pauli string's exp_value back together
        for i, obs in enumerate(job.measure.observables):
            string = obs.pauli_string
            local: float = 0
            for monoms in string.monomials:
                if TYPE_CHECKING:
                    assert isinstance(monoms.coef, (int, float))
                local += expectation_values[monoms.name] * monoms.coef
            exp_values.update(
                {f"observable_{i}" if obs.label is None else obs.label: local}
            )
            if job.measure.shots == 0:
                variance = 0.0
            else:
                variance = (1.0 - local**2) / job.measure.shots
            errors.update(
                {f"observable_{i}" if obs.label is None else obs.label: variance}
            )
        if len(exp_values) == 1:
            return Result(
                job,
                next(iter(exp_values.values())),
                next(iter(errors.values())),
                shots=job.measure.shots,
            )
        return Result(job, exp_values, errors, shots=job.measure.shots)

    else:  # No optimization by MPQP but could have some by pytket
        if provider_params is not None:
            if provider_params.optimisation_strategy is not None:
                optimisation_strat = provider_params.optimisation_strategy
        optimisation_strat = None
        for i, o in enumerate(job.measure.observables):
            translated_obs = o.to_other_language(
                Language.TKET, targets=job.measure.targets
            )

            exp_value = get_operator_expectation_value(
                circuit, translated_obs, backend, job.measure.shots, optimisation_strat
            ).real
            exp_values.update(
                {f"observable_{i}" if o.label is None else o.label: exp_value}
            )
            if job.measure.shots == 0:
                variance = 0.0
            else:
                variance = (1.0 - exp_value**2) / job.measure.shots
            errors.update({f"observable_{i}" if o.label is None else o.label: variance})
        if len(exp_values) == 1:
            return Result(
                job,
                next(iter(exp_values.values())),
                next(iter(errors.values())),
                shots=job.measure.shots,
            )
    return Result(job, exp_values, errors, shots=job.measure.shots)


def submit_job_nexus(
    job: Job, provider_params: Optional[TketParams] = None
) -> tuple[str, "ExecuteJobRef"]:
    """Submit a job to a supported Quantinuum Nexus backend."""
    if job.job_type == JobType.OBSERVABLE:
        return submit_nexus_observable(job, provider_params)

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
        provider_params=provider_params,
    )


def submit_nexus_observable(
    job: Job, provider_params: Optional[TketParams] = None
) -> tuple[str, "ExecuteJobRef"]:
    """Submit an observable as one Nexus execution job.

    Exact observables submit one state-vector circuit. For sampled observables,
    each qubit-wise commuting Pauli group is submitted as a circuit containing
    the required basis change followed by a measurement. The original
    MPQP `Job` is required to reconstruct the expectation value from the
    returned states or counts.
    """
    if TYPE_CHECKING:
        assert isinstance(job.measure, ExpectationMeasure)

    circuit = job.circuit.without_measurements()
    n_shots: int | list[None]
    if job.measure.optimize_measurement:
        # If for some reason the device supports state vector
        # Otherwise this quirk was caught way before arriving here

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
        n_shots = (
            job.measure.shots if job.measure.shots != 0 else [None] * len(grouping)
        )
    else:
        raise ValueError(
            "Cannot submit Observable jobs as is through Nexus. Enable optimize_measurement to proceed."
        )

    return submit_circuits_to_nexus(
        job,
        circuits,
        n_shots,
        name=f"mpqp-observable-{job.device.value}",
        description="mpqp:observable",
        provider_params=provider_params,
        grouping=grouping,
    )


def submit_circuits_to_nexus(
    job: Job,
    circuits: list[QCircuit],
    n_shots: int | list[None],
    name: str,
    description: str = "",
    provider_params: Optional[TketParams] = None,
    grouping: Optional[list[list[PauliStringMonomial]]] = None,
) -> tuple[str, "ExecuteJobRef"]:
    """This function compiles the inputted circuit(s) and send them as one Job to Nexus.
    The generated job can contain multiple circuits if the jobType is OBSERVABLE because of Pauli grouping.
    In this case one circuit will be generated and sent by commuting groups of monomials.

    If the jobType is something different or optimize_measurement is set at False this function should send 1 circuit through 1 job.

    Note:
        If you want to retrieve which circuits is for which group, the index of the group in measure.get_pauli_grouping(),
        is the same as the circuit's index or the one present in its description.
        Since PauliMonomials can get quite long we cannot put everything in the description;
    Args:
        job: The job to be executed
        circuits: One or several circuit(s) to be submitted
        n_shots: The number of shots to be requested on the hardware, if at None the jobType will be STATE_VECTOR or OBSERVABLE (ideal).
        name: The name of the job
        decription: The description of the job, holds the index of the group being ran.
        provider_params: Provider specific parameters
        grouping: Optional grouping kept in memory for perforances reasons.
    """

    import qnexus as qnx

    if TYPE_CHECKING:
        assert isinstance(job.device, QUANTINUUMDevice)
    tket_circuits = []
    for circuit in circuits:
        if circuit.transpiled_circuit is None:
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
                else ""
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

    return job.id, execute_job_ref


def extract_observable_result(
    job: Job,
    expectation_values: dict[str, float],
    errors: float | dict[str, float],
    shots: int,
) -> Result:
    """Fills out the data of a MPQP Result with the results of a Quantinuum OBSERVABLE job."""
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
    grouping: Optional[list[list[PauliStringMonomial]]] = None,
) -> Result:
    """Returns an MPQP Result computed from the probability vectors obtained by executing the submitted circuits.

    Args:
        job: The Job being executed.
        grouped_probabilities: list of vectors from which the expectation_measures are going to be computed.
        grouping: Optional list of PauliStringMonomial holds the grouping of the monomials into commuting sets.
    """
    if TYPE_CHECKING:
        assert isinstance(job.measure, ExpectationMeasure)
    if grouping is None:
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
    amplitudes: list[Complex] | npt.NDArray[np.complex128],
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
    grouping: Optional[list[list[PauliStringMonomial]]] = None,
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
    if job is None and (
        JobType.OBSERVABLE or job_ref.annotations.description == "mpqp:observable"
    ):
        raise ValueError(
            "Retrieving a Quantinuum observable result requires the original MPQP `Job`."
        )

    if job is not None and job.job_type == JobType.OBSERVABLE:

        from mpqp.tools.pauli_grouping import pauli_monomial_eigenvalues

        if grouping is None:
            grouping = []
        if not isinstance(job.measure, ExpectationMeasure):
            raise ValueError("Observable jobs must have an `ExpectationMeasure`.")
        job.id = job_id
        grouping = job.measure.get_pauli_grouping()
        eigenvalues = [
            {monomial.name: pauli_monomial_eigenvalues(monomial) for monomial in group}
            for group in grouping
        ]
        if len(result_refs) != len(grouping):
            raise ValueError(
                "The number of circuit sent for an OBSERVABLE job must be the same as the number of groups in the pauli grouping"
                "This can happen because of different grouping algorithm make sure you're using the same observable(s) and grouping method as this job."
            )
        exp_values, errors = {}, {}
        for index, result_ref in enumerate(result_refs):
            if TYPE_CHECKING:
                assert isinstance(result_ref, ExecutionResultRef)
            backend_result = result_ref.download_result()
            if TYPE_CHECKING:
                assert isinstance(backend_result, BackendResult)
            if job.measure.shots == 0:
                state = backend_result.get_state()
                sorted_values = []
                for i in range(len(state)):
                    sorted_values.append(float(np.abs(state[i]) ** 2))
            else:
                raw_counts = backend_result.get_counts()
                received_shots = sum(raw_counts.values())
                if received_shots != job.measure.shots:
                    raise ValueError(
                        "Received number of shots is different from given number of shots."
                    )
                length = 2**job.measure.nb_qubits
                sorted_values: list[float] = []
                for i in range(length):
                    binary_state = f"{bin(i)[2:].zfill(len(bin(length))- 3)}"
                    tket_binary = tuple(int(b) for b in binary_state)
                    if tket_binary in raw_counts:
                        sorted_values.append(
                            raw_counts[tket_binary].real / job.measure.shots
                        )
                    else:
                        sorted_values.append(0)
            for name, eigenvalue in eigenvalues[index].items():
                expectation_value: float = np.dot(
                    eigenvalue,
                    np.array(sorted_values, dtype=np.float64),
                )
                exp_values[name] = expectation_value
        for i, obs in enumerate(job.measure.observables):
            string = obs.pauli_string
            local: float = 0
            for monoms in string.monomials:
                if TYPE_CHECKING:
                    assert isinstance(monoms.coef, (int, float))
                local += exp_values[monoms.name] * monoms.coef
            exp_values.update(
                {f"observable_{i}" if obs.label is None else obs.label: local}
            )
            if job.measure.shots == 0:
                variance = 0.0
            else:
                variance = (1.0 - local**2) / job.measure.shots
            errors.update(
                {f"observable_{i}" if obs.label is None else obs.label: variance}
            )

        return Result(job, exp_values, errors, shots=job.measure.shots)

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
