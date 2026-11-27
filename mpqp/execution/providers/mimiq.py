"""Execution on local and cloud MIMIQ simulators.

For offline execution, install ``mpqp[mimiq]`` and select
:attr:`~mpqp.execution.devices.MIMIQDevice.LOCAL_STATEVECTOR`. No account is
required. Both basis sampling and complete statevectors use MPQP qubit order:

    >>> from mpqp import MIMIQDevice, QCircuit, X, run
    >>> result = run(QCircuit([X(0)], nb_qubits=2), MIMIQDevice.LOCAL_STATEVECTOR) # doctest: +MIMIQ
    >>> result.amplitudes # doctest: +MIMIQ
    array([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j])

For cloud execution, install ``mpqp[mimiq]`` and run ``setup_connections``,
then choose MIMIQ and your cloud platform. Alternatively, call
:func:`~mpqp.execution.connection.mimiq_connection.setup_mimiq_account` once
(defaulting to MIMIQ V2).
Use ``CLOUD_STATEVECTOR`` or ``CLOUD_MPS`` with :func:`~mpqp.execution.runner.run`,
or submit a job and retrieve it later:

    >>> from mpqp import BasisMeasure, MIMIQDevice, QCircuit, X, get_remote_result, submit
    >>> circuit = QCircuit([X(0), BasisMeasure(shots=16)])
    >>> job_id, job = submit(circuit, MIMIQDevice.CLOUD_MPS) # doctest: +SKIP
    >>> result = get_remote_result(job) # doctest: +SKIP

``CircuitBinding`` inputs to :func:`~mpqp.execution.runner.run` execute in
unroll order. Each expansion produces a separate simulator execution, including
a separate cloud job when using a cloud device.

Sample bitstrings follow ``BasisMeasure.targets`` in their declared order,
independently of the classical register indices. Complete statevectors are
available locally. Cloud statevectors, general observable jobs, and MPQP noise
models are not supported. The MPS bond dimension controls the accuracy/cost
tradeoff; results with a reported fidelity estimate
below one emit a truncation warning.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Optional
from warnings import warn

import numpy as np

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.execution.connection.mimiq_connection import get_mimiq_connection
from mpqp.execution.devices import MIMIQDevice
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.providers.providers_params import MimiqParams
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, QCSResults


def _prepare(job: Job) -> Circuit:
    if not isinstance(job.circuit, QCircuit) or not isinstance(job.device, MIMIQDevice):
        raise TypeError('MIMIQ execution requires a QCircuit and MIMIQDevice.')
    if job.circuit.noises:
        raise DeviceJobIncompatibleError(
            'MIMIQ does not support MPQP noise models yet.'
        )
    if job.job_type != JobType.SAMPLE and (
        job.device.is_remote() or job.job_type != JobType.STATE_VECTOR
    ):
        raise DeviceJobIncompatibleError(
            'MIMIQ supports sampling and local statevector jobs only.'
        )
    if job.job_type == JobType.SAMPLE and not isinstance(job.measure, BasisMeasure):
        raise ValueError('MIMIQ sampling requires a BasisMeasure.')
    translated = job.circuit.transpiled_for_device(job.device)
    if TYPE_CHECKING:
        assert isinstance(translated, Circuit)
    if translated.is_symbolic():
        raise ValueError('Bind all circuit parameters before MIMIQ execution.')
    return translated


def _extract_samples(raw: QCSResults, job: Job) -> Result:
    measure = job.measure
    if not isinstance(measure, BasisMeasure) or measure.c_targets is None:
        raise ValueError(
            'MIMIQ samples require resolved classical measurement targets.'
        )
    counts: Counter[str] = Counter()
    for bits, count in raw.histogram().items():
        # MPQP samples follow measurement target order, even when classical
        # targets are sparse or arranged in a different order.
        counts[''.join(str(int(bits[bit])) for bit in measure.c_targets)] += count
    if sum(counts.values()) != measure.shots:
        raise RuntimeError(
            'MIMIQ returned a different number of samples than requested.'
        )
    if any(fidelity < 1 - 1e-12 for fidelity in raw.fidelities):
        warn(
            'MIMIQ MPS sampling involved truncation; reported fidelity estimates: '
            f'{raw.fidelities}. Increase bonddim to improve accuracy.',
            RuntimeWarning,
        )
    result = Result(
        job,
        [
            Sample(len(measure.targets), bin_str=bits, count=count)
            for bits, count in counts.items()
        ],
        shots=measure.shots,
    )
    job.status = JobStatus.DONE
    return result


def _run_local(job: Job, params: MimiqParams) -> Result:
    translated = _prepare(job)
    try:
        from exaqt import ExaqtQCS
    except ModuleNotFoundError as error:
        if error.name != 'exaqt':
            raise
        raise ModuleNotFoundError(
            'Local MIMIQ execution requires Exaqt. Install "mpqp[mimiq]".',
            name='exaqt',
        ) from error

    from mimiqcircuits import QCSResults

    simulator = ExaqtQCS(seed=params.seed)
    job.status = JobStatus.RUNNING
    if job.job_type == JobType.SAMPLE:
        measure = job.measure
        assert isinstance(measure, BasisMeasure)
        raw = simulator.execute(translated, nsamples=measure.shots)
        if not isinstance(raw, QCSResults):
            raise RuntimeError('Exaqt did not return one result for the circuit.')
        return _extract_samples(raw, job)

    state, _ = simulator.evolvezerostate(translated)
    amplitudes = np.asarray(state.q.amplitudes(), dtype=np.complex128)
    # Exaqt numbers qubit 0 as the least significant bit; MPQP uses it first.
    vector = amplitudes.reshape((2,) * translated.num_qubits()).T.reshape(-1)
    # The translated circuit already carries the input global phase.
    result = Result(job, StateVector(vector), errors=0, shots=0, g_phase_handling=False)
    job.status = JobStatus.DONE
    return result


def run_mimiq(job: Job, params: Optional[MimiqParams] = None) -> Result:
    """Execute a MIMIQ job and return its MPQP result.

    Args:
        job: Sampling job, or a local statevector job, to execute.
        params: Simulator settings, or defaults.

    Returns:
        The result in MPQP qubit and measurement target order.

    Raises:
        DeviceJobIncompatibleError: If the requested job type is unsupported.
        ModuleNotFoundError: If Exaqt is unavailable for local execution.
    """
    if not job.device.is_remote():
        return _run_local(job, params or MimiqParams())
    job_id, _ = submit_mimiq(job, params)
    return get_result_from_mimiq_job_id(job_id, job=job)


def submit_mimiq(job: Job, params: Optional[MimiqParams] = None) -> tuple[str, Job]:
    """Submit a cloud sampling job without waiting for its result.

    Args:
        job: Cloud sampling job to submit.
        params: Cloud simulator settings, or defaults.

    Returns:
        The persistent execution ID and associated job.

    Raises:
        RemoteExecutionError: If the selected device is local.
    """
    if not job.device.is_remote():
        raise RemoteExecutionError('MIMIQ cloud submission requires a remote device.')
    params = params or MimiqParams()
    translated = _prepare(job)
    measure = job.measure
    if not isinstance(measure, BasisMeasure):
        raise ValueError("MIMIQ cloud execution requires a BasisMeasure.")
    if measure.shots > 65536:
        raise ValueError('MIMIQ cloud sampling supports at most 65536 shots per job.')
    options: dict[str, Any] = {
        'algorithm': 'mps' if job.device == MIMIQDevice.CLOUD_MPS else 'statevector',
        'nsamples': measure.shots,
        'timelimit': params.timelimit,
    }
    if params.seed is not None:
        options['seed'] = params.seed
    if job.device == MIMIQDevice.CLOUD_MPS:
        options.update(bonddim=params.bonddim, entdim=params.entdim)
    job_id = get_mimiq_connection().submit(translated, **options)
    if not job_id:
        raise RemoteExecutionError('MIMIQ did not return a valid execution ID.')
    job.id = job_id
    job.status = JobStatus.QUEUED
    return job_id, job


def get_mimiq_job_status(job_id: str) -> JobStatus:
    """Return the status of an existing MIMIQ cloud execution."""
    status = get_mimiq_connection().requestInfo(job_id).status
    states = {
        'NEW': JobStatus.QUEUED,
        'RUNNING': JobStatus.RUNNING,
        'DONE': JobStatus.DONE,
        'ERROR': JobStatus.ERROR,
        'CANCELED': JobStatus.CANCELLED,
    }
    if status not in states:
        raise RemoteExecutionError(f'Unrecognized MIMIQ job status: {status}.')
    return states[status]


def _recover_job(job_id: str, device: MIMIQDevice) -> Job:
    import mimiqcircuits as mc

    from mpqp.gates import CustomControlledGate, CustomGate

    circuits, parameters = get_mimiq_connection().get_inputs(job_id)
    if len(circuits) != 1:
        raise NotImplementedError(
            'MIMIQ retrieval requires a single circuit execution.'
        )
    source = circuits[0]
    circuit = QCircuit(source.num_qubits())
    targets: list[int] = []
    c_targets: list[int] = []
    for instruction in source:
        op, qubits = instruction.operation, list(instruction.qubits)
        if isinstance(op, mc.Measure):
            targets.extend(qubits)
            c_targets.extend(instruction.bits)
        elif isinstance(op, mc.Barrier):
            continue
        elif isinstance(op, mc.Gate):
            if targets:
                raise NotImplementedError(
                    'Cannot recover a MIMIQ circuit with mid-circuit measurements.'
                )
            controls = op.num_controls if isinstance(op, mc.Control) else 0
            operation = op.op if isinstance(op, mc.Control) else op
            assert operation is not None
            matrix = operation.matrix
            if callable(matrix):
                matrix = matrix()
            gate = CustomGate(
                np.asarray(matrix.tolist(), dtype=np.complex128), qubits[controls:]
            )
            circuit.add(
                CustomControlledGate(qubits[:controls], gate) if controls else gate
            )
        else:
            raise NotImplementedError(
                f'Cannot recover MIMIQ operation {type(op).__name__}.'
            )
    if not targets:
        raise NotImplementedError(
            'MIMIQ retrieval currently supports sampling executions only.'
        )
    circuit.add(BasisMeasure(targets, c_targets=c_targets, shots=parameters['samples']))
    job = Job(JobType.SAMPLE, circuit, device)
    job.id = job_id
    return job


def get_result_from_mimiq_job_id(
    job_id: str, job: Optional[Job] = None, device: Optional[MIMIQDevice] = None
) -> Result:
    """Retrieve a cloud result using its original job or its ID and device.

    Args:
        job_id: Persistent MIMIQ execution ID.
        job: Original job, when available, preserving its circuit metadata.
        device: Cloud device, required when the original job is unavailable.

    Returns:
        The completed MPQP sampling result.
    """
    if job is None:
        if device is None or not device.is_remote():
            raise ValueError('Provide the MIMIQ cloud device when retrieving by ID.')
        job = _recover_job(job_id, device)
    raw = get_mimiq_connection().get_result(job_id)
    return _extract_samples(raw, job)
