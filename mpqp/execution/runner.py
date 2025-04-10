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

from numbers import Complex
from textwrap import indent
from typing import TYPE_CHECKING, Iterable, Optional

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
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.providers.atos import run_atos, submit_QLM
from mpqp.execution.providers.aws import run_braket, submit_job_braket
from mpqp.execution.providers.azure import run_azure, submit_job_azure
from mpqp.execution.providers.google import run_google
from mpqp.execution.providers.ibm import run_ibm, submit_remote_ibm
from mpqp.execution.result import BatchResult, Result
from mpqp.execution.simulated_devices import IBMSimulatedDevice, SimulatedDevice
from mpqp.tools.display import state_vector_ket_shape
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError
from mpqp.tools.generics import OneOrMany, find_index, flatten
from sympy import Expr
from typeguard import typechecked


@typechecked
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
    Id_before = np.eye(2 ** measure.rearranged_targets[0])
    Id_after = np.eye(2 ** (circuit.nb_qubits - measure.rearranged_targets[-1] - 1))

    tweaked_observables = [
        Observable(np.kron(np.kron(Id_before, obs.matrix), Id_after))
        # TODO: avoid to force using matrix representation to add identities if the observable is defined by
        #  pauli string (use I @ obs @ I) or diagonal coefficients (perform kron product with [1,1] instead of I matrix)
        for obs in measure.observables
    ]

    tweaked_measure = ExpectationMeasure(
        tweaked_observables,
        list(range(circuit.nb_qubits)),
        measure.shots,
    )
    return tweaked_measure


@typechecked
def generate_job(
    circuit: QCircuit, device: AvailableDevice, values: dict[Expr | str, Complex] = {}
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

    Returns:
        The Job containing information about the execution of the circuit.
    """
    circuit = circuit.subs(values, True)

    m_list = circuit.measurements
    nb_meas = len(m_list)

    if nb_meas == 0:
        job = Job(JobType.STATE_VECTOR, circuit, device)
    elif nb_meas == 1:
        measurement = m_list[0]
        if isinstance(measurement, BasisMeasure):
            if measurement.shots <= 0:
                job = Job(JobType.STATE_VECTOR, circuit, device, measurement)
            else:
                job = Job(JobType.SAMPLE, circuit, device, measurement)
        elif isinstance(measurement, ExpectationMeasure):
            job = Job(
                JobType.OBSERVABLE,
                circuit,
                device,
                adjust_measure(measurement, circuit),
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

    return job


@typechecked
def _run_diagonal_observables(
    circuit: QCircuit,
    exp_measure: ExpectationMeasure,
    device: AvailableDevice,
    observable_job: Job,
    values: dict[Expr | str, Complex],
    translation_warning: bool = True,
) -> Result:

    adapted_circuit = circuit.without_measurements()
    adapted_circuit.add(BasisMeasure(exp_measure.targets, shots=exp_measure.shots))

    result = _run_single(
        adapted_circuit, device, values, False, translation_warning=translation_warning
    )
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


@typechecked
def _run_single(
    circuit: QCircuit,
    device: AvailableDevice,
    values: dict[Expr | str, Complex],
    display_breakpoints: bool = True,
    translation_warning: bool = True,
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
        translation_warning: If `True`, a warning will be raised.

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

    if display_breakpoints:
        for k in range(len(circuit.breakpoints)):
            display_kth_breakpoint(circuit, k, device)

    circuit = circuit.without_breakpoints()
    job = generate_job(circuit, device, values)
    job.status = JobStatus.INIT

    if len(circuit.measurements) == 1:
        measure = circuit.measurements[0]
        if isinstance(measure, ExpectationMeasure):
            if measure.optim_diagonal and measure.are_all_diagonal():
                return _run_diagonal_observables(
                    circuit, measure, device, job, values, translation_warning
                )

    if len(circuit.noises) != 0:
        if not device.is_noisy_simulator():
            raise DeviceJobIncompatibleError(
                f"Device {device} cannot simulate circuits containing NoiseModels."
            )
        elif not isinstance(
            device, (ATOSDevice, AWSDevice, IBMDevice, SimulatedDevice)
        ):
            raise NotImplementedError(f"Noisy simulations not supported on {device}.")

    if isinstance(device, (IBMDevice, IBMSimulatedDevice)):
        return run_ibm(job, translation_warning)
    elif isinstance(device, ATOSDevice):
        return run_atos(job, translation_warning)
    elif isinstance(device, AWSDevice):
        return run_braket(job, translation_warning)
    elif isinstance(device, GOOGLEDevice):
        return run_google(job, translation_warning)
    elif isinstance(device, AZUREDevice):
        return run_azure(job, translation_warning)
    else:
        raise NotImplementedError(f"Device {device} not handled")


@typechecked
def run(
    circuit: OneOrMany[QCircuit],
    device: OneOrMany[AvailableDevice],
    values: Optional[dict[Expr | str, Complex]] = None,
    display_breakpoints: bool = True,
    translation_warning: bool = True,
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
        translation_warning: If `True`, a warning will be raised.

    Returns:
        The Result containing information about the measurement required.

    Examples:
        >>> c = QCircuit(
        ...     [X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=1000)],
        ...     label="X CNOT circuit",
        ... )
        >>> result = run(c, IBMDevice.AER_SIMULATOR)
        >>> print(result)
        Result: X CNOT circuit, IBMDevice, AER_SIMULATOR
          Counts: [0, 0, 0, 1000]
          Probabilities: [0, 0, 0, 1]
          Samples:
            State: 11, Index: 3, Count: 1000, Probability: 1
          Error: None
        >>> batch_result = run(
        ...     c,
        ...     [ATOSDevice.MYQLM_PYLINALG, AWSDevice.BRAKET_LOCAL_SIMULATOR]
        ... )
        >>> print(batch_result)
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
        >>> result = run([c,c2], IBMDevice.AER_SIMULATOR)
        >>> print(result)
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
    if values is None:
        values = {}

    def namer(circ: QCircuit, i: int):
        circ.label = f"circuit {i}" if circ.label is None else circ.label
        return circ

    # TODO: here detect that we have a full diag observable job

    if isinstance(circuit, Iterable) or isinstance(device, Iterable):
        return BatchResult(
            [
                _run_single(
                    namer(circ, i + 1),
                    dev,
                    values,
                    display_breakpoints,
                    translation_warning,
                )
                for i, circ in enumerate(flatten(circuit))
                for dev in flatten(device)
            ]
        )
    else:
        return _run_single(
            circuit, device, values, display_breakpoints, translation_warning
        )


@typechecked
def submit(
    circuit: QCircuit,
    device: AvailableDevice,
    values: Optional[dict[Expr | str, Complex]] = None,
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

    job = generate_job(circuit, device, values)
    job.status = JobStatus.INIT

    if isinstance(device, IBMDevice):
        job_id, _ = submit_remote_ibm(job)
    elif isinstance(device, ATOSDevice):
        job_id, _ = submit_QLM(job)
    elif isinstance(device, AWSDevice):
        job_id, _ = submit_job_braket(job)
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
        res = _run_single(copy, device, {}, False)
        if TYPE_CHECKING:
            assert isinstance(res, Result)
        print(f"DEBUG: After instruction {bp_instructions_index}{name_part}, state is")
        print("       " + state_vector_ket_shape(res.amplitudes))
        if bp.draw_circuit:
            print("       and circuit is")
            print(indent(str(copy), "       "))
