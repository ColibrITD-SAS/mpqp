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

from numbers import Complex, Number
from textwrap import indent
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union, overload

import numpy as np

from mpqp.core.circuit import QCircuit
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
from mpqp.execution.result import BatchResult, Result
from mpqp.tools.display import state_vector_ket_shape
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError
from mpqp.tools.generics import OneOrMany, find_index, flatten

if TYPE_CHECKING:
    from qiskit.circuit import Parameter
    from sympy import Basic, Expr


ValuesKey = Union["Expr", "Parameter", "Basic", str]
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


def adjust_measure(measure: ExpectationMeasure, circuit: QCircuit):
    """We allow the measure to not span the entire circuit, but providers
    usually do not support this behavior. To make this work, we tweak the measure
    this function to match the expected behavior.

    In order to do this, we add identity measures on the qubits not targeted by
    the measure. In addition to this, some swaps are automatically added so the
    the qubits measured are ordered and contiguous (though this is done in
    :func:`generate_job`)

    Args:
        measure: The expectation measure, potentially incomplete.
        circuit: The circuit to which will be added the potential swaps allowing
            the user to get the expectation value of the qubits in an arbitrary
            order (this part is not handled by this function).

    Returns:
        The measure padded with identities before and after.
    """
    # TODO: use this only for specific provider

    if measure.targets == list(range(circuit.nb_qubits)):
        return measure

    tweaked_observables = []
    n_before = measure.rearranged_targets[0]
    n_after = circuit.nb_qubits - measure.rearranged_targets[-1] - 1
    for obs in measure.observables:
        if obs._pauli_string is not None:  # pyright: ignore[reportPrivateUsage]
            from mpqp.measures import pI

            pauli = pI(n_before - 1) @ obs.pauli_string @ pI(n_after - 1)
            tweaked_observables.append(Observable(pauli))
        else:
            Id_before = np.eye(2**n_before)
            Id_after = np.eye(2**n_after)
            tweaked_observables.append(
                Observable(
                    np.kron(
                        np.kron(Id_before, obs.matrix), Id_after
                    )  # pyright: ignore[reportArgumentType]
                )
            )

    tweaked_measure = ExpectationMeasure(
        tweaked_observables,
        list(range(circuit.nb_qubits)),
        measure.shots,
        measure.commuting_type,
        measure.grouping_method,
        optimize_measurement=measure.optimize_measurement,
    )
    return tweaked_measure


def generate_job(
    circuit: QCircuit,
    device: AvailableDevice,
    values: Optional[ValuesDict] = None,
    exec_mode: Optional[ExecutionMode] = ExecutionMode.JOB,
) -> Job:
    # TODO: docstring
    """Creates the Job of appropriate type and containing the information needed
    for the execution of the circuit.

    If the circuit contains symbolic variables (see section :ref:`VQA` for more
    information), the ``values`` parameter is used to perform the necessary
    substitutions.

    Args:
        circuit: Circuit to be run.
        device: Device on which the circuit will be run.
        values: Set of values to substitute for symbolic variables.
        exec_mode:

    Returns:
        The Job containing information about the execution of the circuit.
    """
    if values is not None and not device.is_remote():  # TODO : check why is remote
        from sympy import Expr

        subs_values: dict[Expr | str, Complex] = {}
        for k, v in values.items():
            if isinstance(k, (str, Expr)):
                if not isinstance(v, Complex):
                    raise TypeError(
                        f"Parameter binding requires numeric values; got {type(v).__name__}."
                    )
                subs_values[k] = v

        circuit = circuit.subs(subs_values, True)

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
            m = adjust_measure(measurement, circuit)
            c = circuit.without_measurements(deep_copy=False)
            c.add(m)
            job = Job(
                JobType.OBSERVABLE,
                c,
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
    reservation_arn: Optional[str] = None,
) -> Result:
    # TODO: docstring + replace reservation_arn by dict for provider specific options
    """Runs the circuit on the ``backend``. If the circuit depends on variables,
    the ``values`` given in parameters are used to do the substitution.

    Args:
        circuit: QCircuit to be run.
        device: Device, on which the circuit will be run.
        values: Set of values to substitute symbolic variables. Defaults to ``{}``.
        display_breakpoints: If ``False``, breakpoints will be disabled. Each
            breakpoint adds an execution of the circuit(s), so you may use this
            option for performance if need be.

    Returns:
        The Result containing information about the measurement required.

    Raises:
        DeviceJobIncompatibleError: if a non noisy simulator is given in
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

    if isinstance(device, (IBMDevice, StaticIBMSimulatedDevice)):
        from mpqp.execution.providers.ibm import run_ibm, run_remote_ibm_batch

        if job.mode == ExecutionMode.BATCH and device.is_remote():
            batch_results = run_remote_ibm_batch([job])
            return batch_results[0]

        return run_ibm(job)

    elif isinstance(device, ATOSDevice):
        return run_atos(job)
    elif isinstance(device, AWSDevice):
        return run_braket(job, reservation_arn=reservation_arn)
    elif isinstance(device, GOOGLEDevice):
        return run_google(job)
    elif isinstance(device, AZUREDevice):
        return run_azure(job)

    else:
        raise NotImplementedError(f"Device {device} not handled")


@overload
def run(
    circuit: OneOrMany[QCircuit],
    device: Sequence[AvailableDevice],
    values: BatchValuesInput = None,
    display_breakpoints: bool = True,
    reservation_arn: Optional[str] = None,
    mode: Optional[ExecutionMode] = None,
    values_batch: Optional[list[ValuesDict]] = None,
) -> BatchResult: ...


# TODO: why using values and values_batch at the same time


@overload
def run(
    circuit: Sequence[QCircuit],
    device: OneOrMany[AvailableDevice],
    values: Optional[ValuesDict] = None,
    display_breakpoints: bool = True,
    reservation_arn: Optional[str] = None,
    mode: Optional[ExecutionMode] = None,
    values_batch: Optional[list[ValuesDict]] = None,
) -> BatchResult: ...


@overload
def run(
    circuit: QCircuit,
    device: AvailableDevice,
    values: Optional[ValuesDict] = None,
    display_breakpoints: bool = True,
    reservation_arn: Optional[str] = None,
    mode: Optional[ExecutionMode] = None,
    values_batch: Optional[list[ValuesDict]] = None,
) -> Result: ...


def run(
    circuit: OneOrMany[QCircuit],
    device: OneOrMany[AvailableDevice],
    values: BatchValuesInput = None,
    display_breakpoints: bool = True,
    reservation_arn: Optional[str] = None,
    mode: Optional[ExecutionMode] = None,
    values_batch: Optional[list[ValuesDict]] = None,
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

    """

    def namer(circ: QCircuit, i: int):
        circ.label = f"circuit {i}" if circ.label is None else circ.label
        return circ

    circuits = [circuit] if isinstance(circuit, QCircuit) else list(circuit)
    devices = [device] if isinstance(device, AvailableDevice) else list(device)

    exec_mode = mode or ExecutionMode.JOB

    if values_batch is not None and exec_mode != ExecutionMode.BATCH:
        raise ValueError("values_batch is only supported when mode == VQAMode.BATCH")

    if exec_mode == ExecutionMode.BATCH:
        if len(devices) != 1:
            raise ValueError(
                "Batch mode is only defined for a single backend, but got "
                f"{len(devices)} devices."
            )

        if values_batch is not None and len(values_batch) != len(circuits):
            raise ValueError("values_batch must have the same length as circuits.")

        target_device = devices[0]
        per_run_circuits, per_run_values = prepare_run_batch_inputs(circuits, values)

        jobs = []
        for i, circ in enumerate(per_run_circuits):
            jobs.append(
                generate_job(
                    namer(circ, i + 1), target_device, per_run_values[i], exec_mode
                )
            )

        # TODO: batch only supported for IBM ? maybe raise an error otherwise.
        #  And why only observable here ?
        if isinstance(target_device, IBMDevice) and target_device.is_remote():
            from mpqp.execution.providers.ibm import run_remote_ibm_batch

            for job in jobs:
                if job.job_type != JobType.OBSERVABLE:
                    raise ValueError(
                        "IBM batch execution supports only observable jobs "
                        f"(found {job.job_type} in circuit '{job.circuit.label}')."
                    )
            return run_remote_ibm_batch(jobs)

        if isinstance(circuit, Iterable) or isinstance(device, Iterable):
            return BatchResult(
                [
                    _run_single(
                        namer(circ, i + 1),
                        dev,
                        values,
                        display_breakpoints,
                    )
                    for i, circ in enumerate(flatten(circuit))
                    for dev in flatten(device)
                ]
            )

        # TODO : remark, remove weird management of multi circuit and multi device, it was already done in a more
        #  compact way
    else:
        return _run_single(circuit, device, values, display_breakpoints)


def submit(
    circuit: QCircuit,
    device: AvailableDevice,
    values: Optional[ValuesDict] = None,
    mode: Optional[ExecutionMode] = None,
    reservation_arn: Optional[str] = None,
) -> tuple[str, Job]:
    # TODO replace reservation_arn + docstring
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

            job_id, _ = submit_remote_ibm(job)

    elif isinstance(device, ATOSDevice):
        job_id, _ = submit_QLM(job)
    elif isinstance(device, AWSDevice):
        job_id, _ = submit_job_braket(job, reservation_arn=reservation_arn)
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
