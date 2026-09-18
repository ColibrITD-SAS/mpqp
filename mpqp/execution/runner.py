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

from itertools import pairwise
from numbers import Complex
from textwrap import indent
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, overload

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
from mpqp.execution.providers.providers_params import ProviderParams, QiskitParams
from mpqp.execution.result import BatchResult, Result
from mpqp.tools.display import state_vector_ket_shape
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError
from mpqp.tools.generics import OneOrMany, find_index, flatten

if TYPE_CHECKING:
    from sympy import Expr


def adjust_measure(
    measure: ExpectationMeasure,
    circuit: QCircuit,
    expand_to_full_register: bool = True,
) -> ExpectationMeasure:
    """A measure can be incomplete and not span the entire circuit, but providers
    usually do not support this behavior. The function therefore adjusts each
    observable to span the circuit's full qubit register.

    In order to do this, we place identity operators on the qubits not targeted
    by the measure. If the targets are not ordered, each observable is first
    reordered so that its local qubit order matches the sorted target order.
    When ``expand_to_full_register`` is enabled, Pauli observables are directly
    embedded on their target qubits, while matrix observables are padded with
    identity matrices when the targets are ordered and contiguous, and are
    otherwise embedded through their pauli decomposition.

    Args:
        measure: The expectation measure, potentially incomplete.
        circuit: The circuit defining the full qubit register.
        expand_to_full_register: Whether observables must span the complete
            circuit register. Providers supporting local observable targets can
            disable this while still benefiting from target reordering.

    Returns:
        The reordered measure. Its observables target the complete circuit
        register when ``expand_to_full_register`` is enabled.

    Raises:
        ValueError: If the number of target qubits does not match the number
            of qubits represented by an observable.

    """
    # TODO: use this only for specific provider

    nb_qubits = circuit.nb_qubits
    targets = list(measure.targets)

    observables = measure.observables

    if any(observable.nb_qubits != len(targets) for observable in observables):
        raise ValueError(
            f"Each observable must act on {len(targets)} qubits to match the "
            "measurement targets."
        )

    if targets == list(range(nb_qubits)):
        return measure

    targets_are_ordered = all(
        first_target < second_target
        for first_target, second_target in pairwise(targets)
    )

    if not targets_are_ordered:
        ordered_targets = sorted(targets)

        pauli_permutation = [ordered_targets.index(target) for target in targets]
        # Reorder observables to match the sorted target order
        reordered_observables: list[Observable] = []

        for observable in observables:
            if observable._matrix is None or (  # pyright: ignore[reportPrivateUsage]
                measure.optimize_measurement and expand_to_full_register
            ):
                reordered_observables.append(
                    Observable(
                        observable.pauli_string.rearrange(pauli_permutation),
                        label=observable.label,
                    )
                )
            else:
                from mpqp.tools.maths import rearrange_matrix

                reordered_observables.append(
                    Observable(
                        rearrange_matrix(observable.matrix, targets),
                        label=observable.label,
                    )
                )

        targets = ordered_targets
        observables = reordered_observables

    adjusted_observables = observables
    adjusted_targets = targets
    if expand_to_full_register:
        targets_are_contiguous = bool(targets) and targets == list(
            range(targets[0], targets[-1] + 1)
        )
        adjusted_observables = []

        from mpqp.core.instruction.measurement.pauli_string import (
            PauliString,
            PauliStringMonomial,
            pI,
        )

        for observable in observables:
            if (
                observable._pauli_string is None  # pyright: ignore[reportPrivateUsage]
                and targets_are_contiguous
            ):
                nb_qubits_before = targets[0]
                nb_qubits_after = nb_qubits - targets[-1] - 1

                full_matrix = observable.matrix

                if nb_qubits_before > 0:
                    identity_before = np.eye(2**nb_qubits_before)
                    full_matrix = np.kron(identity_before, full_matrix)

                if nb_qubits_after > 0:
                    identity_after = np.eye(2**nb_qubits_after)
                    full_matrix = np.kron(full_matrix, identity_after)

                adjusted_observables.append(
                    Observable(
                        full_matrix,  # pyright: ignore[reportArgumentType]
                        label=observable.label,
                    )
                )
            else:
                embedded_pauli_string = PauliString()

                for monomial in observable.pauli_string.monomials:
                    embedded_atoms = [pI] * nb_qubits

                    for local_idx, target in enumerate(targets):
                        embedded_atoms[target] = monomial.atoms[local_idx]

                    embedded_pauli_string += PauliStringMonomial(
                        monomial.coef,
                        embedded_atoms,
                    )

                adjusted_observables.append(
                    Observable(
                        embedded_pauli_string.simplify(),
                        label=observable.label,
                    )
                )
        adjusted_targets = list(range(nb_qubits))

    adjusted_measure = ExpectationMeasure(
        adjusted_observables,
        adjusted_targets,
        measure.shots,
        measure.commuting_type,
        measure.grouping_method,
        label=measure.label,
        optimize_measurement=measure.optimize_measurement,
        optim_diagonal=measure.optim_diagonal,
    )
    return adjusted_measure


def generate_job(
    circuit: QCircuit,
    device: AvailableDevice,
    values: "Optional[dict[Expr | str, Complex]]" = None,
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
    if values is not None:
        circuit = circuit.subs(values)

    m_list = circuit.measurements
    nb_meas = len(m_list)

    if nb_meas == 0:
        job = Job(JobType.STATE_VECTOR, circuit, device)
    elif nb_meas == 1:
        measurement = m_list[0]
        if isinstance(measurement, BasisMeasure):
            if measurement.shots <= 0:
                job = Job(JobType.STATE_VECTOR, circuit, device)
            else:
                job = Job(JobType.SAMPLE, circuit, device)
        elif isinstance(measurement, ExpectationMeasure):
            m = adjust_measure(
                measurement,
                circuit,
                expand_to_full_register=not isinstance(device, AWSDevice),
            )
            circuit = circuit.without_measurements(deep_copy=False)
            circuit.add(m)
            job = Job(
                JobType.OBSERVABLE,
                circuit,
                device,
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


def _run_diagonal_observables(
    circuit: QCircuit,
    exp_measure: ExpectationMeasure,
    device: AvailableDevice,
    observable_job: Job,
    values: "Optional[dict[Expr | str, Complex]]" = None,
) -> Result:

    adapted_circuit = circuit.without_measurements(deep_copy=False)
    adapted_circuit.add(BasisMeasure(exp_measure.targets, shots=exp_measure.shots))

    result = _run_single(adapted_circuit, device, values, False)
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
    values: "Optional[dict[Expr | str, Complex]]" = None,
    display_breakpoints: bool = True,
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

    job = generate_job(circuit, device, values)
    job.status = JobStatus.INIT
    if len(circuit.measurements) == 1:
        measure = circuit.measurements[0]
        if isinstance(measure, ExpectationMeasure):
            if measure.optim_diagonal and measure.only_diagonal_observables():
                return _run_diagonal_observables(circuit, measure, device, job, values)

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
        if provider_params is not None and not isinstance(
            provider_params, QiskitParams
        ):
            raise ValueError(
                f"provider_params should be QiskitParam not {type(provider_params)}"
            )
        return run_ibm(job, provider_params)
    elif isinstance(device, ATOSDevice):
        return run_atos(job)
    elif isinstance(device, AWSDevice):
        return run_braket(job)
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
    values: "Optional[dict[Expr | str, Complex]]" = None,
    display_breakpoints: bool = True,
    provider_params: Optional[ProviderParams] = None,
) -> BatchResult: ...


@overload
def run(
    circuit: Sequence[QCircuit],
    device: OneOrMany[AvailableDevice],
    values: "Optional[dict[Expr | str, Complex]]" = None,
    display_breakpoints: bool = True,
    provider_params: Optional[ProviderParams] = None,
) -> BatchResult: ...


@overload
def run(
    circuit: QCircuit,
    device: AvailableDevice,
    values: "Optional[dict[Expr | str, Complex]]" = None,
    display_breakpoints: bool = True,
    provider_params: Optional[ProviderParams] = None,
) -> Result: ...


def run(
    circuit: OneOrMany[QCircuit],
    device: OneOrMany[AvailableDevice],
    values: "Optional[dict[Expr | str, Complex]]" = None,
    display_breakpoints: bool = True,
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

    def namer(circ: QCircuit, i: int):
        circ.label = f"circuit {i}" if circ.label is None else circ.label
        return circ

    if isinstance(circuit, Iterable) or isinstance(device, Iterable):
        return BatchResult(
            [
                _run_single(
                    namer(circ, i + 1),
                    dev,
                    values,
                    display_breakpoints,
                    provider_params,
                )
                for i, circ in enumerate(flatten(circuit))
                for dev in flatten(device)
            ]
        )
    else:
        return _run_single(
            circuit, device, values, display_breakpoints, provider_params
        )


def submit(
    circuit: QCircuit,
    device: AvailableDevice,
    values: Optional[dict[Expr | str, Complex]] = None,
    provider_params: Optional[ProviderParams] = None,
) -> tuple[str | list[str], Job]:
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
        The job id or ids provided by the remote device after
        submission and the associated MPQP job.

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
        if provider_params is not None and not isinstance(
            provider_params, QiskitParams
        ):
            raise ValueError(
                f"provider_params should be QiskitParam not {type(provider_params)}"
            )
        job_id, _ = submit_remote_ibm(job, provider_params)
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
        res = _run_single(copy, device, None, False)
        if TYPE_CHECKING:
            assert isinstance(res, Result)
        print(f"DEBUG: After instruction {bp_instructions_index}{name_part}, state is")
        print("       " + state_vector_ket_shape(res.amplitudes))
        if bp.draw_circuit:
            print("       and circuit is")
            print(indent(str(copy), "       "))
