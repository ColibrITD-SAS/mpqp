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
from mpqp.tools.display import state_vector_ket_shape
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError
from mpqp.tools.generics import OneOrMany, find_index, flatten

if TYPE_CHECKING:
    from sympy import Expr


def adjust_measure(measure: ExpectationMeasure, circuit: QCircuit):
    # TODO: to enhance docs

    """We allow the measure to not span the entire circuit, but providers
    usually do not support this behavior. To make this work, we tweak the measure
    this function to match the expected behavior.

    In order to do this, we add identity measures on the qubits not targeted by
    the measure. pauli observables are directly embeded on their target qubits,
    while matrix observables are padded with identity matrices when the targets
    are ordered and contiguous, otherwise are embedded through their pauli decomposition.

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

    nb_qubits = circuit.nb_qubits
    targets = measure.targets

    targets_is_ordered = all(
        [targets[i] > targets[i - 1] for i in range(1, len(targets))]
    )
    targets_is_contiguous = (
        len(targets) > 0
        and targets_is_ordered
        and (targets[-1] - targets[0] + 1 == len(targets))
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
                    full_matrix, label=obs.label  # pyright: ignore[reportArgumentType]
                )
            )
            continue

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
        list(range(circuit.nb_qubits)),
        measure.shots,
        measure.commuting_type,
        measure.grouping_method,
        label=measure.label,
        optimize_measurement=measure.optimize_measurement,
        optim_diagonal=measure.optim_diagonal,
    )
    return tweaked_measure


def generate_job(
    circuit: QCircuit,
    device: AvailableDevice,
    values: "Optional[dict[Expr | str, Complex]]" = None,
    use_emulator: bool = False,
) -> Job:
    """Creates the Job of appropriate type and containing the information needed
    for the execution of the circuit.
    """
    if values is not None:
        circuit = circuit.subs(values)

    m_list = circuit.measurements
    nb_meas = len(m_list)

    if nb_meas == 0:
        job = Job(JobType.STATE_VECTOR, circuit, device, use_emulator=use_emulator)
    elif nb_meas == 1:
        measurement = m_list[0]
        if isinstance(measurement, BasisMeasure):
            if measurement.shots <= 0:
                job = Job(
                    JobType.STATE_VECTOR,
                    circuit,
                    device,
                    use_emulator=use_emulator,
                )
            else:
                job = Job(
                    JobType.SAMPLE,
                    circuit,
                    device,
                    use_emulator=use_emulator,
                )
        elif isinstance(measurement, ExpectationMeasure):
            if measurement.optimize_measurement and isinstance(device, AWSDevice):
                job = Job(
                    JobType.OBSERVABLE,
                    circuit,
                    device,
                    use_emulator=use_emulator,
                )
            else:
                m = adjust_measure(measurement, circuit)
                c = circuit.without_measurements(deep_copy=False)
                c.add(m)
                job = Job(
                    JobType.OBSERVABLE,
                    c,
                    device,
                    use_emulator=use_emulator,
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
    use_emulator: bool = False,
) -> Result:
    """Runs the circuit on the backend."""
    from mpqp.execution.simulated_devices import (
        SimulatedDevice,
        StaticIBMSimulatedDevice,
    )

    if use_emulator and not isinstance(device, AWSDevice):
        raise DeviceJobIncompatibleError(
            "`use_emulator=True` is only supported for AWSDevice."
        )

    if display_breakpoints:
        for k in range(len(circuit.breakpoints)):
            display_kth_breakpoint(circuit, k, device)

    job = generate_job(circuit, device, values, use_emulator=use_emulator)
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
        return run_ibm(job)
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


def run(
    circuit: OneOrMany[QCircuit],
    device: OneOrMany[AvailableDevice],
    values: "Optional[dict[Expr | str, Complex]]" = None,
    display_breakpoints: bool = True,
    use_emulator: bool = False,
) -> Result | BatchResult:
    """Runs the circuit on the backend, or list of backend, provided in
    parameter.
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
                    use_emulator,
                )
                for i, circ in enumerate(flatten(circuit))
                for dev in flatten(device)
            ]
        )
    else:
        return _run_single(
            circuit,
            device,
            values,
            display_breakpoints,
            use_emulator,
        )


def submit(
    circuit: QCircuit,
    device: AvailableDevice,
    values: Optional[dict[Expr | str, Complex]] = None,
    use_emulator: bool = False,
) -> tuple[str, Job]:
    """Submit the job related to the circuit on the remote backend provided in
    parameter.
    """
    if values is None:
        values = {}

    if use_emulator:
        raise RemoteExecutionError(
            "submit(..., use_emulator=True) is not supported because the AWS emulator runs locally. "
            "Use run(..., use_emulator=True) instead."
        )

    if not device.is_remote():
        raise RemoteExecutionError(
            "submit(...) function is only made for remote device."
        )

    job = generate_job(circuit, device, values, use_emulator=use_emulator)
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
        res = _run_single(copy, device, None, False)
        if TYPE_CHECKING:
            assert isinstance(res, Result)
        print(f"DEBUG: After instruction {bp_instructions_index}{name_part}, state is")
        print("       " + state_vector_ket_shape(res.amplitudes))
        if bp.draw_circuit:
            print("       and circuit is")
            print(indent(str(copy), "       "))
