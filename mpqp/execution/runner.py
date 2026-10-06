"""
Once the circuit is defined, you can execute it and retrieve the result using
the function :func:`run`. You can execute said circuit on one or several devices
(local or remote). The function will wait (blocking) until the job is completed
and will return a :class:`~mpqp.execution.result.Result` if only one
device was given or a :class:`~mpqp.execution.result.BatchResult`
otherwise (see the section :ref:`Results` for more details).

Alternatively, when running jobs on a remote device, you might prefer to
retrieve the result asynchronously, without having to wait and block the
application until the computation is completed. In that case, you can use the
:func:`submit` instead. This will submit the job and
return the corresponding job id and :class:`~mpqp.execution.job.Job` object.

.. note::
    Unlike :func:`run`, we can only submit on one device at a time.
"""

from __future__ import annotations

from copy import copy, deepcopy
from itertools import pairwise
from numbers import Number
from textwrap import indent
from typing import TYPE_CHECKING, Optional, Sequence, Union, overload

import numpy as np

from mpqp.core.circuit import QCircuit
from mpqp.core.circuitbinding import CircuitBinding
from mpqp.core.instruction.breakpoint import Breakpoint
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.execution.devices import (
    ATOSDevice,
    AvailableDevice,
    AWSDevice,
    AZUREDevice,
    GOOGLEDevice,
    IBMDevice,
)
from mpqp.execution.job import ExecutionMode, Job, JobStatus, JobType
from mpqp.execution.providers.atos import run_atos, submit_QLM
from mpqp.execution.providers.aws import run_braket, submit_job_braket
from mpqp.execution.providers.azure import run_azure, submit_job_azure
from mpqp.execution.providers.google import run_google
from mpqp.execution.providers.providers_params import (
    AWSParams,
    ProviderParams,
    QiskitParams,
)
from mpqp.execution.result import BatchResult, Result
from mpqp.tools.display import state_vector_ket_shape
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError
from mpqp.tools.generics import OneOrMany, find_index

if TYPE_CHECKING:
    from sympy import Expr


ValuesKey = Union["Expr", str]
ValuesDict = dict[ValuesKey, Number]
BatchValuesInput = Optional[Union[ValuesDict, Sequence[ValuesDict]]]


def prepare_run_batch_inputs(
    circuits: list[QCircuit],
    values: BatchValuesInput,
) -> tuple[list[QCircuit], list[Optional[ValuesDict]]]:
    # TODO: docs

    if values is None:
        return circuits, [None] * len(circuits)

    if isinstance(values, dict):
        return circuits, [values] * len(circuits)
    values_list = list(values)

    if len(circuits) == 1 and len(values_list) > 1:
        return [circuits[0] for _ in range(len(values_list))], list(values_list)

    if len(values_list) == 1 and len(circuits) > 1:
        return circuits, [values_list[0]] * len(circuits)

    if len(values_list) == len(circuits):
        return circuits, list(values_list)

    raise ValueError(
        "In BATCH mode, number of circuits must match number of values dicts "
        f"Got {len(circuits)} circuits and {len(values_list)} values sets."
    )


def adjust_measure(measure: ExpectationMeasure, nb_qubits: int):
    """A measure can be incomplete and not span the entire circuit, but providers
    usually do not support this behavior. To make this work, we tweak the measure
    this function to match the expected behavior.

    In order to do this, we place identity operators on the qubits not targeted
    by the measure. If the targets are not ordered, each observable is first
    reordered so that its local qubit order matches the sorted target order.
    Pauli observables are directly embedded on their target qubits, while matrix
    observables are padded with identity matrices when the targets are ordered
    and contiguous, and are otherwise embedded through their pauli decomposition.

    Args:
        measure: The expectation measure, potentially incomplete.
        nb_qubits: The number of qubits in the circuit.

    Returns:
        A measure targeting all circuit qubits, with observables embedded into
        the full register.
    """
    # TODO: use this only for specific provider
    if measure.nb_qubits > nb_qubits:
        raise ValueError(
            f"Number of provided qubits: {nb_qubits} is more than the number of qubits of the measure: {measure.nb_qubits}"
        )
    if measure.targets == list(range(nb_qubits)):
        return measure

    targets = measure.targets

    targets_is_ordered = all(a < b for a, b in pairwise(targets))
    if not targets_is_ordered:
        ordered_targets = sorted(targets)
        contiguous_targets = [targets.index(t) for t in ordered_targets]
        for obs in measure.observables:
            if (
                obs._matrix is None  # pyright: ignore[reportPrivateUsage]
                or measure.optimize_measurement
            ):  # Order pauli string
                obs._pauli_string = (  # pyright: ignore[reportPrivateUsage]
                    obs.pauli_string.rearrange(contiguous_targets)
                )
            else:  # Order the matrix
                from mpqp.tools.maths import rearrange_matrix

                obs.matrix = rearrange_matrix(obs.matrix, contiguous_targets)

    targets_is_contiguous = len(targets) > 0 and (
        targets[-1] - targets[0] + 1 == len(sorted(targets))
    )

    tweaked_observables: list[Observable] = []

    for obs in measure.observables:
        from mpqp.core.instruction.measurement.pauli_string import (
            PauliString,
            PauliStringMonomial,
        )
        from mpqp.measures import pI

        if (
            obs._pauli_string is None  # pyright: ignore[reportPrivateUsage]
            and targets_is_contiguous
        ):
            n_before = targets[0]
            n_after = nb_qubits - targets[-1] - 1

            full_matrix = obs.matrix

            Id_before = np.eye(2**n_before)
            Id_after = np.eye(2**n_after)

            if n_before > 0:
                full_matrix = np.kron(Id_before, full_matrix)

            if n_after > 0:
                full_matrix = np.kron(full_matrix, Id_after)

            tweaked_observables.append(
                Observable(
                    full_matrix,  # pyright: ignore[reportArgumentType]
                    label=obs.label,
                )
            )
        else:
            pauli = obs.pauli_string
            embedded = PauliString()

            for mono in pauli.monomials:
                full_register = [pI] * nb_qubits

                for local_idx, target in enumerate(targets):
                    full_register[target] = mono.atoms[local_idx]

                embedded += PauliStringMonomial(mono.coef, full_register)

            tweaked_observables.append(Observable(embedded.simplify(), label=obs.label))

    tweaked_measure = ExpectationMeasure(
        tweaked_observables,
        list(range(nb_qubits)),
        measure.shots,
        measure.commuting_type,
        measure.grouping_method,
        label=measure.label,
        optimize_measurement=measure.optimize_measurement,
        optim_diagonal=measure.optim_diagonal,
    )
    return tweaked_measure


def generate_job(
    circuit: QCircuit | CircuitBinding,
    device: AvailableDevice,
    values: Optional[ValuesDict] = None,
    exec_mode: Optional[ExecutionMode] = None,
) -> Job:
    """Creates the Job of appropriate type and containing the information needed
    for the execution of the circuit.

    If the circuit contains symbolic variables (see section :ref:`VQA` for more
    information), the ``values`` parameter is used to perform the necessary
    substitutions.

    Args:
        circuit: Circuit to be run.
        device: Device on which the circuit will be run.
        values: Set of values to substitute for symbolic variables.
        exec_mode: Execution mode. ``None`` selects sequential job execution.

    Returns:
        The Job containing information about the execution of the circuit.
    """
    exec_mode = exec_mode or ExecutionMode.JOB
    if values is not None:
        if isinstance(circuit, CircuitBinding):
            raise ValueError("values must be specified in CircuitBinding")
        if device in circuit.transpiled_circuit:
            circuit.bind_parameters(device, values)
        else:
            circuit = circuit.subs(values)  # pyright: ignore[reportArgumentType]

    if isinstance(circuit, CircuitBinding):
        job = Job(circuit.job_type, circuit, device, exec_mode)
        return job

    m_list = circuit.measurements
    nb_meas = len(m_list)

    if nb_meas == 0:
        job = Job(JobType.STATE_VECTOR, circuit, device, exec_mode)

    elif nb_meas == 1:
        measurement = m_list[0]
        if isinstance(measurement, BasisMeasure):
            job = (
                Job(JobType.STATE_VECTOR, circuit, device, exec_mode)
                if measurement.shots <= 0
                else Job(JobType.SAMPLE, circuit, device, exec_mode)
            )

        elif isinstance(measurement, ExpectationMeasure):
            if not (measurement.optimize_measurement and isinstance(device, AWSDevice)):
                m = adjust_measure(measurement, circuit.nb_qubits)
                circuit = circuit.without_measurements(deep_copy=False)
                circuit.add(m)
            job = Job(
                JobType.OBSERVABLE,
                circuit,
                device,
                exec_mode,
            )

        else:
            raise NotImplementedError(
                f"Measurement type {type(measurement)} not handled"
            )
    else:
        raise NotImplementedError(
            "The current version of MPQP does not support multiple measurements in a "
            "circuit."
        )

    if values is not None and device.is_remote():
        job.values = values

    return job


def _run_diagonal_observables(
    circuit: QCircuit,
    exp_measure: ExpectationMeasure,
    device: AvailableDevice,
    observable_job: Job,
    values: Optional[ValuesDict] = None,
    mode: Optional[ExecutionMode] = ExecutionMode.JOB,
) -> Result:
    adapted_circuit = circuit.without_measurements(deep_copy=False)
    adapted_circuit.add(BasisMeasure(exp_measure.targets, shots=exp_measure.shots))

    result = _run_single(adapted_circuit, device, values, False, mode)
    return _compute_result_diagonal_observables(result, exp_measure, observable_job)


def _compute_result_diagonal_observables(
    result: Result,
    exp_measure: ExpectationMeasure,
    observable_job: Job,
) -> Result:
    """Compute diagonal-observable expectation values from sample probabilities.

    Args:
        result: Sampling result containing the computational-basis
            probabilities.
        exp_measure: Diagonal expectation measurement to evaluate.
        observable_job: Original observable job attached to the returned result.

    Returns:
        A result containing either one expectation value or a mapping from
        observable labels to expectation values.
    """

    probas = result.probabilities

    error = 0 if exp_measure.shots == 0 else None
    if exp_measure.nb_observables == 1:
        exp_value = float(probas.dot(exp_measure.observables[0].diagonal_elements))
        return Result(
            observable_job,
            exp_value,
            error,
            exp_measure.shots,
        )

    exp_values = dict()
    errors = dict()
    for obs in exp_measure.observables:
        # 3M-TODO: replace this dot product with cupy, apparently more optim
        exp_values[obs.label] = float(probas.dot(obs.diagonal_elements))
        errors[obs.label] = error

    return Result(
        observable_job,
        exp_values,
        errors,
        exp_measure.shots,
    )


def _run_single(
    circuit: QCircuit,
    device: AvailableDevice,
    values: Optional[ValuesDict] = None,
    display_breakpoints: bool = True,
    mode: Optional[ExecutionMode] = ExecutionMode.JOB,
    provider_params: Optional[ProviderParams] = None,
) -> Result:
    """Runs the circuit on the ``backend``. If the circuit depends on variables,
    the ``values`` given in parameters are used to do the substitution.

    Args:
        circuit: QCircuit to be run.
        device: Device, on which the circuit will be run.
        values: Set of values to substitute symbolic variables. Defaults to ``{}``.
        display_breakpoints: If ``False``, breakpoints will be disabled. Each
            breakpoint adds an execution of the circuit(s), so you may use this
            option for performance if need be.
        provider_params: Provider's specific parameters, mainly for remote runs.

    Returns:
        The Result containing information about the measurement required.

    Raises:
        DeviceJobIncompatibleError: if a non-noisy simulator is given in
            parameter and the circuit contains noise
        NotImplementedError: If the device is not handled for noisy simulation
            or other submissions.

    Example:
        >>> c = QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], shots=1000)], label="Bell pair")
        >>> result = run(c, IBMDevice.AER_SIMULATOR)
        >>> print(result) # doctest: +SKIP
        Result: IBMDevice, AER_SIMULATOR
         Probabilities: [0.523, 0, 0, 0.477]
         Counts: [523, 0, 0, 477]
         Samples:
          State: 00, Index: 0, Count: 523, Probability: 0.523
          State: 11, Index: 3, Count: 477, Probability: 0.477
         Error: None

    """
    from mpqp.execution.simulated_devices import (
        SimulatedDevice,
        StaticIBMSimulatedDevice,
    )

    if display_breakpoints:
        for k in range(len(circuit.breakpoints)):
            display_kth_breakpoint(circuit, k, device)

    circ_transpile = None
    if device in circuit.transpiled_circuit:
        circ_transpile = copy(circuit.transpiled_circuit[device])

    job = generate_job(circuit, device, values, mode)
    job.status = JobStatus.INIT

    if len(circuit.measurements) == 1:
        measure = circuit.measurements[0]
        if isinstance(measure, ExpectationMeasure):
            if measure.optim_diagonal and measure.only_diagonal_observables():
                return _run_diagonal_observables(
                    circuit, measure, device, job, values, mode
                )

    if len(circuit.noises) != 0:
        if not device.is_noisy_simulator():
            raise DeviceJobIncompatibleError(
                f"Device {device} cannot simulate circuits containing NoiseModels."
            )
        elif not isinstance(
            device, (ATOSDevice, AWSDevice, IBMDevice, GOOGLEDevice, SimulatedDevice)
        ):
            raise NotImplementedError(f"Noisy simulations not supported on {device}.")

    try:
        if isinstance(device, (IBMDevice, StaticIBMSimulatedDevice)):
            from mpqp.execution.providers.ibm import run_ibm, run_remote_ibm_batch

            if provider_params is not None and not isinstance(
                provider_params, QiskitParams
            ):
                raise ValueError(
                    f"provider_params should be QiskitParam not {type(provider_params)}"
                )

            if job.mode == ExecutionMode.BATCH and device.is_remote():
                batch_results = run_remote_ibm_batch([job])
                return batch_results[0]

            result = run_ibm(job, provider_params)
            if not isinstance(result, Result):
                raise TypeError("A single circuit execution must return a Result.")
            return result

        elif isinstance(device, ATOSDevice):
            return run_atos(job)
        elif isinstance(device, AWSDevice):
            if provider_params is not None and not isinstance(
                provider_params, AWSParams
            ):
                raise ValueError(
                    f"provider_params should be AWSParams, not {type(provider_params)}"
                )
            result = run_braket(job, provider_params=provider_params)
            if not isinstance(result, Result):
                raise TypeError("A single circuit execution must return a Result.")
            return result
        elif isinstance(device, GOOGLEDevice):
            return run_google(job)
        elif isinstance(device, AZUREDevice):
            return run_azure(job)

        else:
            raise NotImplementedError(f"Device {device} not handled")
    finally:
        if circ_transpile is not None:
            circuit.transpiled_circuit[device] = circ_transpile


def _run_circuit_binding(
    circuit_binding: CircuitBinding,
    device: AvailableDevice,
    display_breakpoints: bool = True,
    mode: Optional[ExecutionMode] = None,
    provider_params: Optional[ProviderParams] = None,
) -> BatchResult:
    """Execute every expansion of a circuit binding on one device.

    Args:
        circuit_binding: Lazy collection of circuits, parameter values and
            measurements to execute.
        device: Device on which all binding executions are run.
        display_breakpoints: Whether breakpoints should be displayed. Breakpoint
            display for bindings is currently not implemented.
        mode: Execution mode propagated to provider jobs and fallbacks.
        provider_params: Provider-specific execution configuration.

    Returns:
        A batch containing one result per resolved binding execution.

    Raises:
        DeviceJobIncompatibleError: If a noisy binding targets a device that
            cannot simulate noise.
        NotImplementedError: If circuit bindings are unsupported by the
            selected provider.
    """
    from mpqp.execution.simulated_devices import (
        SimulatedDevice,
        StaticIBMSimulatedDevice,
    )

    if display_breakpoints:
        pass
        # TODO: implement display breakpoints for CircuitBinding
        # raise ValueError(
        #    "display_breakpoints is not supported with CircuitBinding"
        # )

    if circuit_binding.is_noisy:
        if not device.is_noisy_simulator():
            raise DeviceJobIncompatibleError(
                f"Device {device} cannot simulate circuits containing NoiseModels."
            )
        elif not isinstance(
            device,
            (ATOSDevice, AWSDevice, IBMDevice, GOOGLEDevice, SimulatedDevice),
        ):
            raise NotImplementedError(f"Noisy simulations not supported on {device}.")

    execution_mode = mode or ExecutionMode.JOB

    job = generate_job(circuit_binding, device, exec_mode=execution_mode)

    if isinstance(device, (IBMDevice, StaticIBMSimulatedDevice)):
        if provider_params is not None and not isinstance(
            provider_params, QiskitParams
        ):
            raise ValueError(
                f"provider_params should be QiskitParams, not {type(provider_params)}"
            )
        from mpqp.execution.providers.ibm import run_ibm

        result = run_ibm(
            job,
            provider_params,
        )
    elif isinstance(device, AWSDevice):
        if provider_params is not None and not isinstance(provider_params, AWSParams):
            raise ValueError(
                f"provider_params should be AWSParams, not {type(provider_params)}"
            )
        result = run_braket(job, provider_params=provider_params)
    elif isinstance(device, (ATOSDevice, GOOGLEDevice, AZUREDevice)):
        results: list[Result] = []
        for circuit, values, measurement in circuit_binding.unroll():
            executable = circuit.without_measurements()
            if measurement is not None:
                executable.add(deepcopy(measurement))
            results.append(
                _run_single(
                    executable,
                    device,
                    values,
                    display_breakpoints,
                    mode=execution_mode,
                    provider_params=provider_params,
                )
            )
        return BatchResult(results)
    else:
        raise NotImplementedError(f"Device {device} not handled")

    # for i, (exp_measure, job) in run_diagonal_observables.items():
    #    result.results[i] = _compute_result_diagonal_observables(
    #        result[i], exp_measure, job
    #    )

    return result if isinstance(result, BatchResult) else BatchResult([result])


@overload
def run(
    circuit: CircuitBinding | OneOrMany[QCircuit],
    device: Sequence[AvailableDevice],
    values: BatchValuesInput = None,
    display_breakpoints: bool = True,
    mode: Optional[ExecutionMode] = None,
    provider_params: Optional[ProviderParams] = None,
) -> BatchResult: ...


@overload
def run(
    circuit: CircuitBinding,
    device: OneOrMany[AvailableDevice],
    values: Optional[ValuesDict] = None,
    display_breakpoints: bool = True,
    mode: Optional[ExecutionMode] = None,
    provider_params: Optional[ProviderParams] = None,
) -> BatchResult: ...


@overload
def run(
    circuit: OneOrMany[QCircuit],
    device: AvailableDevice,
    values: Optional[ValuesDict] = None,
    display_breakpoints: bool = True,
    mode: Optional[ExecutionMode] = None,
    provider_params: Optional[ProviderParams] = None,
) -> Result: ...


def run(
    circuit: OneOrMany[QCircuit] | CircuitBinding,
    device: OneOrMany[AvailableDevice],
    values: BatchValuesInput = None,
    display_breakpoints: bool = True,
    mode: Optional[ExecutionMode] = None,
    provider_params: Optional[ProviderParams] = None,
) -> Result | BatchResult:
    """Runs the circuit on the backend, or list of backend, provided in
    parameter.

    If the circuit contains symbolic variables (see section :ref:`VQA` for more
    information on them), the ``values`` parameter is used perform the necessary
    substitutions.

    Args:
        circuit: Circuit, or list of circuits, to be run.
        device: Device, or list of devices, on which the circuit will be run.
        values: Set of values to substitute symbolic variables. Defaults to ``{}``.
        display_breakpoints: If ``False``, breakpoints will be disabled. Each
            breakpoint adds an execution of the circuit(s), so you may use this
            option for performance if need be.
        provider_params: Provider's specific parameters, mainly for remote runs

    Returns:
        The Result containing information about the measurement required.

    Examples:
        >>> c = QCircuit(
        ...     [X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=1000)],
        ...     label="X CNOT circuit",
        ... )
        >>> result = run(c, IBMDevice.AER_SIMULATOR) # doctest: +QISKIT
        >>> print(result) # doctest: +QISKIT
        Result: X CNOT circuit, IBMDevice, AER_SIMULATOR
          Counts: [0, 0, 0, 1000]
          Probabilities: [0, 0, 0, 1]
          Samples:
            State: 11, Index: 3, Count: 1000, Probability: 1
          Error: None
        >>> batch_result = run(  # doctest: +MYQLM, +BRAKET
        ...     c,
        ...     [ATOSDevice.MYQLM_PYLINALG, AWSDevice.BRAKET_LOCAL_SIMULATOR]
        ... )
        >>> print(batch_result) # doctest: +MYQLM, +BRAKET
        BatchResult: 2 results
            Result: X CNOT circuit, ATOSDevice, MYQLM_PYLINALG
              Counts: [0, 0, 0, 1000]
              Probabilities: [0, 0, 0, 1]
              Samples:
                State: 11, Index: 3, Count: 1000, Probability: 1
              Error: 0.0
            Result: X CNOT circuit, AWSDevice, BRAKET_LOCAL_SIMULATOR
              Counts: [0, 0, 0, 1000]
              Probabilities: [0, 0, 0, 1]
              Samples:
                State: 11, Index: 3, Count: 1000, Probability: 1
              Error: None
        >>> c2 = QCircuit(
        ...     [X(0), X(1), BasisMeasure([0, 1], shots=1000)],
        ...     label="X circuit",
        ... )
        >>> result = run([c,c2], IBMDevice.AER_SIMULATOR) # doctest: +QISKIT
        >>> print(result) # doctest: +QISKIT
        BatchResult: 2 results
            Result: X CNOT circuit, IBMDevice, AER_SIMULATOR
              Counts: [0, 0, 0, 1000]
              Probabilities: [0, 0, 0, 1]
              Samples:
                State: 11, Index: 3, Count: 1000, Probability: 1
              Error: None
            Result: X circuit, IBMDevice, AER_SIMULATOR
              Counts: [0, 0, 0, 1000]
              Probabilities: [0, 0, 0, 1]
              Samples:
                State: 11, Index: 3, Count: 1000, Probability: 1
              Error: None
        >>> ibm_instance = "crn:v1:****:public:quantum-computing:us-east:a/****"
        >>> qp = QiskitParams(instance=ibm_instance) # doctest: +SKIP
        >>> run(c2, IBMDevice.IBM_FEZ, provider_params=qp) # doctest: +SKIP

    """

    def namer(circ: QCircuit, i: int) -> QCircuit:
        if not isinstance(circuit, QCircuit) and circ.label is None:
            circ.label = f"circuit {i}"
        return circ

    devices = [device] if isinstance(device, AvailableDevice) else list(device)
    exec_mode = mode or ExecutionMode.JOB

    if isinstance(circuit, CircuitBinding):
        if values is not None:
            raise ValueError("values must be specified inside CircuitBinding")
        results: list[Result] = []
        for target_device in devices:
            batch = _run_circuit_binding(
                circuit,
                target_device,
                display_breakpoints,
                mode=exec_mode,
                provider_params=provider_params,
            )
            results.extend(batch.results)
        return BatchResult(results)

    circuits = [circuit] if isinstance(circuit, QCircuit) else list(circuit)

    if exec_mode == ExecutionMode.BATCH:
        if len(devices) != 1:
            raise ValueError(
                "Batch mode is only defined for a single backend, but got "
                f"{len(devices)} devices."
            )

        per_run_circuits, per_run_values = prepare_run_batch_inputs(circuits, values)
        target_device = devices[0]
        jobs = [
            generate_job(
                namer(circ, i + 1), target_device, per_run_values[i], exec_mode
            )
            for i, circ in enumerate(per_run_circuits)
        ]

        if isinstance(target_device, IBMDevice) and target_device.is_remote():
            from mpqp.execution.providers.ibm import run_remote_ibm_batch

            for job in jobs:
                if job.job_type != JobType.OBSERVABLE:
                    raise ValueError(
                        "IBM batch execution supports only observable jobs "
                        f"(found {job.job_type} in circuit "
                        f"'{getattr(job.circuit, 'label', None)}')."
                    )
            return run_remote_ibm_batch(jobs)

        return BatchResult(
            [
                _run_single(
                    circ,
                    target_device,
                    per_run_values[i],
                    display_breakpoints,
                    mode=exec_mode,
                    provider_params=provider_params,
                )
                for i, circ in enumerate(per_run_circuits)
            ]
        )

    if values is not None and not isinstance(values, dict):
        raise ValueError(
            "A sequence of parameter mappings requires ExecutionMode.BATCH."
        )
    results = [
        _run_single(
            namer(circ, i + 1),
            target_device,
            values,
            display_breakpoints,
            mode=exec_mode,
            provider_params=provider_params,
        )
        for target_device in devices
        for i, circ in enumerate(circuits)
    ]
    if (
        len(results) == 1
        and isinstance(circuit, QCircuit)
        and isinstance(device, AvailableDevice)
    ):
        return results[0]
    return BatchResult(results)


def submit(
    circuit: QCircuit,
    device: AvailableDevice,
    values: Optional[ValuesDict] = None,
    mode: Optional[ExecutionMode] = None,
    provider_params: Optional[ProviderParams] = None,
) -> tuple[str, Job]:
    """Submit the job related to the circuit on the remote backend provided in
    parameter. The submission returns a ``job_id`` that can be used to retrieve
    the :class:`~mpqp.execution.result.Result` later using the
    :func:`~mpqp.execution.remote_handler.get_remote_result`
    function.

    If the circuit contains symbolic variables (see section :ref:`VQA` for more
    information), the ``values`` parameter is used perform the necessary
    substitutions.

    Note that this function only supports single device submissions.

    Args:
        circuit: QCircuit to be run.
        device: Remote device to which the circuit will be submitted.
        values: Values to substitute for symbolic variables. Defaults to ``{}``.
        provider_params: Provider's specific parameters for remote submissions

    Returns:
        The job id provided by the remote device after submission of the job.

    Example:
        >>> circuit = QCircuit([H(0), CNOT(0,1), BasisMeasure([0,1], shots=10)])
        >>> job_id, job = submit(circuit, ATOSDevice.QLM_LINALG) #doctest: +SKIP
        Logging as user <qlm_user>...
        Submitted a new batch: Job766
        >>> print(f"Status of {job_id}: {job.job_status}") #doctest: +SKIP
        Status of Job766: JobStatus.RUNNING

    Note:
        Unlike :func:`run`, you can only submit on one device at a time.
    """
    if values is None:
        values = {}
    if not device.is_remote():
        raise RemoteExecutionError(
            "submit(...) function is only made for remote device."
        )

    job = generate_job(circuit, device, values, mode)
    job.status = JobStatus.INIT

    if isinstance(device, IBMDevice):
        # TODO: we said that provider specific stuff should only go into the provider specific execution file ,
        #  here ibm.py, to keep the logic simple on runner.py
        if provider_params is not None and not isinstance(
            provider_params, QiskitParams
        ):
            raise ValueError(
                f"provider_params should be QiskitParam not {type(provider_params)}"
            )

        if mode == ExecutionMode.SESSION:
            from mpqp.execution.connection.ibm_connection import (
                get_backend,
                get_or_create_ibm_session,
            )
            from mpqp.execution.providers.ibm import submit_remote_ibm_session

            backend = get_backend(device)
            session = get_or_create_ibm_session(backend)
            job_id, _ = submit_remote_ibm_session(job, session)
        else:
            from mpqp.execution.providers.ibm import submit_remote_ibm

            job_id, _ = submit_remote_ibm(job, provider_params)

    elif isinstance(device, ATOSDevice):
        job_id, _ = submit_QLM(job)
    elif isinstance(device, AWSDevice):
        if provider_params is not None and not isinstance(provider_params, AWSParams):
            raise ValueError(
                f"provider_params should be AWSParams, not {type(provider_params)}"
            )
        job_id, _ = submit_job_braket(job, provider_params=provider_params)
    elif isinstance(device, AZUREDevice):
        job_id, _ = submit_job_azure(job)
    else:
        raise NotImplementedError(f"Device {device} not handled")

    return job_id, job


def display_kth_breakpoint(
    circuit: QCircuit, k: int, device: AvailableDevice = ATOSDevice.MYQLM_CLINALG
):
    """Prints to the standard output the state vector corresponding to the state
    of the system when it encounters the `k^{th}` breakpoint.

    See the documentation of
    :class:`~mpqp.core.instruction.breakpoint.Breakpoint` for examples of
    breakpoints.

    Args:
        circuit: The circuit to be examined.
        k: The state desired is met at the `k^{th}` breakpoint.
        device: The device to use for the simulation.
    """
    bp = circuit.breakpoints[k]
    if bp.enabled:
        name_part = "" if bp.label is None else f", at breakpoint `{bp.label}`"
        relevant_instructions = list(
            filter(
                lambda i: i is bp or not isinstance(i, Breakpoint), circuit.instructions
            )
        )
        bp_instructions_index = find_index(relevant_instructions, lambda i: i is bp)
        copy = QCircuit(
            relevant_instructions[:bp_instructions_index],
            nb_qubits=circuit.nb_qubits,
            nb_cbits=circuit.nb_cbits,
            label=circuit.label,
        )
        res = _run_single(copy, device, None, False)
        if TYPE_CHECKING:
            assert isinstance(res, Result)
        print(f"DEBUG: After instruction {bp_instructions_index}{name_part}, state is")
        print("       " + state_vector_ket_shape(res.amplitudes))
        if bp.draw_circuit:
            print("       and circuit is")
            print(indent(str(copy), "       "))
