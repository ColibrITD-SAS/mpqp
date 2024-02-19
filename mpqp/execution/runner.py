from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sympy import Expr
from typeguard import typechecked
from numbers import Complex

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.execution.devices import AvailableDevice, IBMDevice, ATOSDevice, AWSDevice
from mpqp.execution.providers_execution.aws_execution import (
    run_braket,
    submit_job_braket,
)
from mpqp.execution.providers_execution.atos_execution import run_atos, submit_QLM
from mpqp.execution.providers_execution.ibm_execution import run_ibm, submit_ibmq
from mpqp.execution.result import Result, BatchResult
from mpqp.execution.job import Job, JobType, JobStatus
from mpqp.tools.errors import RemoteExecutionError


@typechecked
def adjust_measure(measure: ExpectationMeasure, circuit: QCircuit):
    """We allow the measure to not span the entire circuit, but none of our
    providers allow for that. So we patch the measure with this function, with
    identities on the qubits that no not interest our user.

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
    """Create a Job from the circuit and the eventual measurements. If the
    circuit depends on variables, the values given in parameters are used to do
    the substitution.

    Args:
        circuit: Circuit to be run.
        device: Device on which the circuit will be run.
        values: Set of values to substitute symbolic variables.

    Returns:
        The Job containing information about the execution of the circuit.
    """
    circuit = circuit.subs(values, True)

    # get the measurements of this circuit
    m_list = circuit.get_measurements()
    nb_meas = len(m_list)

    # determine the job type and create the right job
    if nb_meas == 0:
        job = Job(JobType.STATE_VECTOR, circuit, device)
    elif nb_meas == 1:
        measurement = m_list[0]
        if isinstance(measurement, BasisMeasure):
            # 3M-TODO: handle other basis by adding the right rotation (change of basis) before
            #       measuring in the computational basis
            # 3M-TODO: Muhammad circuit.add(CustomGate(UnitaryMatrix(change_of_basis_inverse)))
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
            "Cannot handle several measurement in the current version"
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

    Example:
        >>> c = QCircuit([H(0), CNOT(0, 1), BasisMeasure([0, 1], shots=1000)], label="Bell pair")
        >>> result = run(c, IBMDevice.AER_SIMULATOR)
        >>> print(result)
        Result: IBMDevice, AER_SIMULATOR
        Counts: [512, 0, 0, 488]
        Probabilities: [0.512 0.    0.    0.488]
        State: 00, Index: 0, Count: 512, Probability: 0.512
        State: 11, Index: 3, Count: 488, Probability: 0.488
        Error: None
    """
    job = generate_job(circuit, device, values)
    job.status = JobStatus.INIT

    if isinstance(device, IBMDevice):
        return run_ibm(job)
    elif isinstance(device, ATOSDevice):
        return run_atos(job)
    elif isinstance(device, AWSDevice):
        return run_braket(job)
    else:
        raise NotImplementedError(f"Device {device} not handled")


@typechecked
def run(
    circuit: QCircuit,
    device: AvailableDevice | list[AvailableDevice],
    values: Optional[dict[Expr | str, Complex]] = None,
) -> Union[Result, BatchResult]:
    """Runs the circuit on the backend, or list of backend, provided in
    parameter. If the circuit depends on variables, the values given in
    parameters are used to do the substitution.

    Args:
        circuit: QCircuit to be run.
        device: Device, or list of devices, on which the circuit will be run.
        values: Set of values to substitute symbolic variables. Defaults to ``{}``.

    Returns:
        The Result containing information about the measurement required.

    Examples:
        >>> c = QCircuit(
        ...     [H(0), CNOT(0, 1), BasisMeasure([0, 1], shots=1000)],
        ...     label="Bell pair",
        ... )
        >>> result = run(c, IBMDevice.AER_SIMULATOR)
        >>> print(result)
        Result: IBMDevice, AER_SIMULATOR
        Counts: [512, 0, 0, 488]
        Probabilities: [0.512 0.    0.    0.488]
        State: 00, Index: 0, Count: 512, Probability: 0.512
        State: 11, Index: 3, Count: 488, Probability: 0.488
        Error: None
        >>> batch_result = run(
        ...     c,
        ...     [ATOSDevice.MYQLM_PYLINALG, AWSDevice.BRAKET_LOCAL_SIMULATOR]
        ... )
        >>> print(batch_result)
        BatchResult: 2 results
        Result: AWSDevice, BRAKET_LOCAL_SIMULATOR
        Counts: [492, 0, 0, 508]
        Probabilities: [0.492 0.    0.    0.508]
        State: 00, Index: 0, Count: 492, Probability: 0.492
        State: 11, Index: 3, Count: 508, Probability: 0.508
        Error: None
        Result: ATOSDevice, MYQLM_PYLINALG
        Counts: [462, 0, 0, 538]
        Probabilities: [0.462 0.    0.    0.538]
        State: 00, Index: 0, Count: 462, Probability: 0.462
        State: 11, Index: 3, Count: 538, Probability: 0.538
        Error: 0.015773547629015002
    """

    if values is None:
        values = {}

    if isinstance(device, list):
        # Duplicate devices are removed
        set_device = list(set(device))
        if len(set_device) == 1:
            return _run_single(circuit, set_device[0], values)

        return BatchResult([_run_single(circuit, dev, values) for dev in set_device])

    return _run_single(circuit, device, values)


@typechecked
def submit(
    circuit: QCircuit, device: AvailableDevice, values: dict[Expr | str, Complex] = {}
) -> tuple[str, Job]:
    """
    Submit the job related with the circuit on the remote backend provided in parameter.
    The submission returns a job_id that can be used to retrieve the Result later.
    If the circuit depends on variables, the values given in parameters are used to do the substitution.
    Unlike :meth:`run`, for the moment, one can only submit a circuit to a single device.

    Example:
        >>> circuit = QCircuit([H(0), CNOT(0,1), BasisMeasure([0,1], shots=10)])
        >>> job_id, job = submit(circuit, ATOSDevice.QLM_LINALG)
        Logging as user <qlm_user>...
        Submitted a new batch: Job766
        >>> print("Status of " +job_id +":", job.job_status)
        Status of Job766: JobStatus.RUNNING

    Args:
        circuit: QCircuit to be run.
        device: Remote device on which the circuit will be submitted.
        values: Set of values to substitute symbolic variables.

    Returns:
        The job id provided by the remote device after submission of the job.
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
