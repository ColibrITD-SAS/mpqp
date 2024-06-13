"""
Once the circuit is defined, you can to execute it and retrieve the result using
the function :func:`run`. You can execute said circuit on one or several devices
(local or remote). The function will wait (blocking) until the job is completed
and will return a :class:`Result<mpqp.execution.result.Result>` in only one
device was given or a :class:`BatchResult<mpqp.execution.result.BatchResult>` 
otherwise (see :ref:`below<Results>`).

Alternatively, when running jobs on a remote device, you could prefer to
retrieve the result asynchronously, without having to wait and block the
application until the computation is completed. In that case, you can use the
:func:`submit` instead. It will submit the job and
return the corresponding job id and :class:`Job<mpqp.execution.job.Job>` object.

.. note::
    Unlike :func:`run`, we can only submit on one device at a time.
"""

from __future__ import annotations

from numbers import Complex
from typing import Iterable, Optional, Union

import numpy as np
from sympy import Expr
from typeguard import typechecked

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.execution.devices import (
    ATOSDevice,
    AvailableDevice,
    AWSDevice,
    GOOGLEDevice,
    IBMDevice,
)
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.providers.atos import run_atos, submit_QLM
from mpqp.execution.providers.aws import run_braket, submit_job_braket
from mpqp.execution.providers.google import run_google
from mpqp.execution.providers.ibm import run_ibm, submit_ibmq
from mpqp.execution.result import BatchResult, Result
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError
from mpqp.tools.generics import OneOrMany


@typechecked
def adjust_measure(measure: ExpectationMeasure, circuit: QCircuit):
    """We allow the measure to not span the entire circuit, but providers
    usually don't support this behavior. To make this work we tweak the measure
    this function to match the expected behavior.

    In order to do this, we add identity measures on the qubits not targeted by
    the measure. In addition of this, some swaps are automatically added so the
    the qubits measured are ordered and contiguous (though this is done in
    :func:`generate_job`)

    Args:
        measure: The expectation measure, potentially incomplete.
        circuit: The circuit to which will be added the potential swaps allowing
            the user to get the expectation value of the qubits in an arbitrary
            order (this part is not handled by this function).

    Returns:
        The measure padded with the identities before and after.
    """
    Id_before = np.eye(2 ** measure.rearranged_targets[0])
    Id_after = np.eye(2 ** (circuit.nb_qubits - measure.rearranged_targets[-1] - 1))
    tweaked_measure = ExpectationMeasure(
        list(range(circuit.nb_qubits)),
        Observable(
            np.kron(
                np.kron(Id_before, measure.observable.matrix), Id_after
            )  # pyright: ignore[reportArgumentType]
        ),
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
    information on them), the ``values`` parameter is used perform the necessary
    substitutions.

    Args:
        circuit: Circuit to be run.
        device: Device on which the circuit will be run.
        values: Set of values to substitute symbolic variables.

    Returns:
        The Job containing information about the execution of the circuit.
    """
    circuit = circuit.subs(values, True)

    m_list = circuit.get_measurements()
    nb_meas = len(m_list)

    if nb_meas == 0:
        job = Job(JobType.STATE_VECTOR, circuit, device)
    elif nb_meas == 1:
        measurement = m_list[0]
        if isinstance(measurement, BasisMeasure):
            # TODO: handle other basis by adding the right rotation (change
            # of basis) before measuring in the computational basis
            # Muhammad: circuit.add(CustomGate(UnitaryMatrix(change_of_basis_inverse)))
            if measurement.shots <= 0:
                job = Job(JobType.STATE_VECTOR, circuit, device)
            else:
                job = Job(JobType.SAMPLE, circuit, device, measurement)
        elif isinstance(measurement, ExpectationMeasure):
            job = Job(
                JobType.OBSERVABLE,
                circuit + measurement.pre_measure,
                device,
                adjust_measure(measurement, circuit),
            )
        else:
            raise NotImplementedError(
                f"Measurement type {type(measurement)} not handled"
            )
    else:
        raise NotImplementedError(
            "Current version of MPQP do not support multiple measurement in a "
            "circuit."
        )

    return job


@typechecked
def _run_single(
    circuit: QCircuit, device: AvailableDevice, values: dict[Expr | str, Complex]
) -> Result:
    """Runs the circuit on the ``backend``. If the circuit depends on variables,
    the ``values`` given in parameters are used to do the substitution.

    Args:
        circuit: QCircuit to be run.
        device: Device, on which the circuit will be run.
        values: Set of values to substitute symbolic variables. Defaults to ``{}``.

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
    job = generate_job(circuit, device, values)
    job.status = JobStatus.INIT

    if circuit.noises:
        if not device.is_noisy_simulator():
            raise DeviceJobIncompatibleError(
                f"Device {device} cannot simulate circuits containing NoiseModels."
            )
        elif not (isinstance(device, ATOSDevice) or isinstance(device, AWSDevice)):
            raise NotImplementedError(
                f"Noisy simulations are not yet available on devices of type {type(device).name}."
            )

    if isinstance(device, IBMDevice):
        return run_ibm(job)
    elif isinstance(device, ATOSDevice):
        return run_atos(job)
    elif isinstance(device, AWSDevice):
        return run_braket(job)
    elif isinstance(device, GOOGLEDevice):
        return run_google(job)
    else:
        raise NotImplementedError(f"Device {device} not handled")


@typechecked
def run(
    circuit: OneOrMany[QCircuit],
    device: OneOrMany[AvailableDevice],
    values: Optional[dict[Expr | str, Complex]] = None,
) -> Union[Result, BatchResult]:
    """Runs the circuit on the backend, or list of backend, provided in
    parameter.

    If the circuit contains symbolic variables (see section :ref:`VQA` for more
    information on them), the ``values`` parameter is used perform the necessary
    substitutions.

    Args:
        circuit: QCircuit to be run.
        device: Device, or list of devices, on which the circuit will be run.
        values: Set of values to substitute symbolic variables. Defaults to ``{}``.

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
          State: 11, Index: 3, Count: 1000, Probability: 1.0
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
          State: 11, Index: 3, Count: 1000, Probability: 1.0
         Error: 0.0
        Result: X CNOT circuit, AWSDevice, BRAKET_LOCAL_SIMULATOR
         Counts: [0, 0, 0, 1000]
         Probabilities: [0, 0, 0, 1]
         Samples:
          State: 11, Index: 3, Count: 1000, Probability: 1.0
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
          State: 11, Index: 3, Count: 1000, Probability: 1.0
         Error: None
        Result: X circuit, IBMDevice, AER_SIMULATOR
         Counts: [0, 0, 0, 1000]
         Probabilities: [0, 0, 0, 1]
         Samples:
          State: 11, Index: 3, Count: 1000, Probability: 1.0
         Error: None

    """
    if values is None:
        values = {}

    if isinstance(circuit, Iterable):
        if isinstance(device, Iterable):
            return BatchResult([_run_single(circ, dev, values) for circ in circuit for dev in device])
        else:
            return BatchResult([_run_single(circ, device, values) for circ in circuit])
    else:
        if isinstance(device, Iterable):
            return BatchResult([_run_single(circuit, dev, values) for dev in device])
        else:
            return _run_single(circuit, device, values)

@typechecked
def submit(
    circuit: QCircuit, device: AvailableDevice, values: dict[Expr | str, Complex] = {}
) -> tuple[str, Job]:
    """Submit the job related with the circuit on the remote backend provided in
    parameter. The submission returns a ``job_id`` that can be used to retrieve
    the :class:`Result<mpqp.execution.result.Result>` later, using the
    :func:`get_remote_result<mpqp.execution.remote_handler.get_remote_result>`
    function.

    If the circuit contains symbolic variables (see section :ref:`VQA` for more
    information on them), the ``values`` parameter is used perform the necessary
    substitutions.

    Mind that this function only support single device submissions.

    Args:
        circuit: QCircuit to be run.
        device: Remote device on which the circuit will be submitted.
        values: Set of values to substitute symbolic variables.

    Returns:
        The job id provided by the remote device after submission of the job.

    Example:
        >>> circuit = QCircuit([H(0), CNOT(0,1), BasisMeasure([0,1], shots=10)])
        >>> job_id, job = submit(circuit, ATOSDevice.QLM_LINALG) #doctest: +SKIP
        Logging as user <qlm_user>...
        Submitted a new batch: Job766
        >>> print("Status of " +job_id +":", job.job_status) #doctest: +SKIP
        Status of Job766: JobStatus.RUNNING

    """
    if not device.is_remote():
        raise RemoteExecutionError(
            "submit(...) function is only made for remote device."
        )

    job = generate_job(circuit, device, values)
    job.status = JobStatus.INIT

    if isinstance(device, IBMDevice):
        job_id, _ = submit_ibmq(job)
    elif isinstance(device, ATOSDevice):
        job_id, _ = submit_QLM(job)
    elif isinstance(device, AWSDevice):
        job_id, _ = submit_job_braket(job)
    else:
        raise NotImplementedError(f"Device {device} not handled")

    return job_id, job
