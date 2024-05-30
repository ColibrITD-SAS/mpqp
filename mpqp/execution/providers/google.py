from __future__ import annotations
from typing import Optional

from typeguard import typechecked

from mpqp.execution.devices import GOOGLEDevice
from mpqp.execution.job import JobType, Job
from mpqp.execution.result import Result, Sample, StateVector

from mpqp.core.instruction.measurement import ComputationalBasis
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure

from mpqp import Language

from cirq.work.observable_measurement_data import ObservableMeasuredResult
from cirq.sim.state_vector_simulator import StateVectorTrialResult
from cirq.transformers.optimize_for_target_gateset import optimize_for_target_gateset
from cirq.value.probability import state_vector_to_probabilities
from cirq.sim.sparse_simulator import Simulator
from cirq.circuits.circuit import Circuit as Cirq_circuit
from cirq.study.result import Result as cirq_result
from cirq.ops.linear_combinations import PauliSum as Cirq_PauliSum
from cirq_google.engine.virtual_engine_factory import (
    load_median_device_calibration,
    create_device_from_processor_id,
)
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from qsimcirq.qsim_simulator import QSimSimulator
from cirq.work.observable_measurement import (
    measure_observables,
    RepetitionsStoppingCriteria,
)
import cirq_ionq as ionq
from cirq_ionq.ionq_gateset import IonQTargetGateset
from cirq.devices.line_qubit import LineQubit


@typechecked
def run_google(job: Job) -> Result:
    """
    Executes the job on the right Google device precised in the job in parameter.
    This function is not meant to be used directly, please use ``runner.run(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    """
    return run_local(job) if not job.device.is_remote() else run_google_remote(job)


@typechecked
def run_google_remote(job: Job) -> Result:
    """
    Executes the job remotely on a Google quantum device. At present, only IonQ devices are supported.

    Args:
        job: job to be executed.

    Returns:
        Result: The result after submission and execution of the job.

    Raises:
        ValueError: If the job's device is not an instance of GOOGLEDevice.
        NotImplementedError: If the job's device is not supported (only IonQ devices are supported currently).
        NotImplementedError: If the job type or basis measure is not supported.
    """
    assert type(job.device) == GOOGLEDevice

    job_cirq_circuit = job.circuit.to_other_language(Language.CIRQ)
    assert isinstance(job_cirq_circuit, Cirq_circuit)

    if job.device.is_ionq():
        from mpqp.execution.connection.env_manager import load_env_variables

        load_env_variables()

        if job.job_type != JobType.SAMPLE:
            raise ValueError(
                f"{job.device}: job_type must be {JobType.SAMPLE} but got job type {job.job_type}"
            )
        assert isinstance(job.measure, BasisMeasure)

        if isinstance(job.measure.basis, ComputationalBasis):
            service = ionq.Service(default_target=job.device.value)
            job_cirq_circuit = optimize_for_target_gateset(
                job_cirq_circuit, gateset=IonQTargetGateset()
            )
            job_cirq_circuit = job_cirq_circuit.transform_qubits(
                {qb: LineQubit(i) for i, qb in enumerate(job_cirq_circuit.all_qubits())}
            )
            result_sim = service.run(
                circuit=job_cirq_circuit, repetitions=job.measure.shots
            )
        else:
            raise NotImplementedError(
                "Does not handle other basis than the ComputationalBasis for the moment"
            )
    else:
        raise NotImplementedError(
            f"{job.device} is not handled for the moment. Only IonQ is supported"
        )

    return extract_result(result_sim, job, job.device)


@typechecked
def run_local(job: Job) -> Result:
    """
    Executes the job locally.

    Args:
        job : The job to be executed.

    Returns:
        Result: The result after submission and execution of the job.

    Raises:
        ValueError: If the job device is not GOOGLEDevice.
    """
    assert type(job.device) == GOOGLEDevice

    if job.device.is_processor():
        return run_local_processor(job)

    job_cirq_circuit = job.circuit.to_other_language(Language.CIRQ)
    assert isinstance(job_cirq_circuit, Cirq_circuit)

    simulator = Simulator(noise=None)

    if job.job_type == JobType.STATE_VECTOR:
        result_sim = simulator.simulate(job_cirq_circuit)
    elif job.job_type == JobType.SAMPLE:
        assert isinstance(job.measure, BasisMeasure)
        if isinstance(job.measure.basis, ComputationalBasis):
            result_sim = simulator.run(job_cirq_circuit, repetitions=job.measure.shots)
        else:
            raise NotImplementedError(
                "Does not handle other basis than the ComputationalBasis for the moment"
            )
    elif job.job_type == JobType.OBSERVABLE:
        assert isinstance(job.measure, ExpectationMeasure)

        cirq_obs = job.measure.observable.to_other_language(
            language=Language.CIRQ, circuit=job_cirq_circuit
        )
        assert type(cirq_obs) == Cirq_PauliSum

        if job.measure.shots == 0:
            result_sim = simulator.simulate_expectation_values(
                job_cirq_circuit, observables=cirq_obs
            )
        else:
            result_sim = measure_observables(
                job_cirq_circuit,
                cirq_obs,  # type: ignore[reportArgumentType]
                simulator,
                stopping_criteria=RepetitionsStoppingCriteria(job.measure.shots),
            )
    else:
        raise ValueError(f"Job type {job.job_type} not handled")

    return extract_result(result_sim, job, job.device)


@typechecked
def run_local_processor(job: Job) -> Result:
    """
    Executes the job locally on processor.

    Args:
        job : The job to be executed.

    Returns:
        Result: The result after submission and execution of the job.
    """
    assert type(job.device) == GOOGLEDevice

    calibration = load_median_device_calibration(job.device.value)
    device = create_device_from_processor_id(job.device.value)

    # noise_props = noise_properties_from_calibration(cal)
    # noise_model = NoiseModelFromGoogleNoiseProperties(noise_props)

    simulator = QSimSimulator(noise=None)
    sim_processor = SimulatedLocalProcessor(
        processor_id=job.device.value,
        sampler=simulator,
        device=device,
        calibrations={calibration.timestamp // 1000: calibration},
    )
    simulator = SimulatedLocalEngine([sim_processor])

    job_cirq_circuit = job.circuit.to_other_language(
        Language.CIRQ, cirq_proc_id=job.device.value
    )
    assert isinstance(job_cirq_circuit, Cirq_circuit)

    if job.job_type == JobType.STATE_VECTOR:
        raise NotImplementedError(
            f"Does not handle {job.job_type} for processor for the moment"
        )
    elif job.job_type == JobType.OBSERVABLE:
        raise NotImplementedError(
            f"Does not handle {job.job_type} for processor for the moment"
        )
    elif job.job_type == JobType.SAMPLE:
        assert isinstance(job.measure, BasisMeasure)
        if isinstance(job.measure.basis, ComputationalBasis):
            result_sim = simulator.get_sampler(job.device.value).run(
                job_cirq_circuit, repetitions=job.measure.shots
            )
        else:
            raise NotImplementedError(
                "Does not handle other basis than the ComputationalBasis for the moment"
            )
    else:
        raise ValueError(f"Job type {job.job_type} not handled")

    return extract_result(result_sim, job, job.device)


def extract_result(
    result: (
        StateVectorTrialResult
        | cirq_result
        | list[float]
        | list[ObservableMeasuredResult]
    ),
    job: Optional[Job] = None,
    device: Optional[GOOGLEDevice] = None,
) -> Result:
    """
    Extracts the result from the simulation and formats it into an mpqp Result object.

    Args:
        result : The result of the simulation.
        job : The original job. Defaults to None.
        device : The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.

    Raises:
        NotImplementedError: If the job is None or the type is not supported.
        ValueError: If the result type does not match the expected type for the job type.
    """
    if job is None:
        raise NotImplementedError("result from job None is not implemented")
    else:
        if job.job_type == JobType.SAMPLE:
            if not isinstance(result, cirq_result):
                raise ValueError(
                    f"result: {type(result)}, must be a cirq_result for job type {job.job_type}"
                )
            return extract_result_SAMPLE(result, job, device)
        elif job.job_type == JobType.STATE_VECTOR:
            if not isinstance(result, StateVectorTrialResult):
                raise ValueError(
                    f"result: {type(result)}, must be a cirq_result for job type {job.job_type}"
                )
            return extract_result_STATE_VECTOR(result, job, device)
        elif job.job_type == JobType.OBSERVABLE:
            if isinstance(result, cirq_result):
                raise ValueError(
                    f"result: {type(result)}, must be a cirq_result for job type {job.job_type}"
                )
            return extract_result_OBSERVABLE(result, job, device)
        else:
            raise NotImplementedError("Job type not supported")


def extract_result_SAMPLE(
    result: cirq_result,
    job: Job,
    device: Optional[GOOGLEDevice] = None,
) -> Result:
    """
    Extracts the result from a sample-based job.

    Args:
        result : The result of the simulation.
        job : The original job.
        device : The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.
    """
    nb_qubits = job.circuit.nb_qubits

    keys_in_order = sorted(result.records.keys())
    counts = result.multi_measurement_histogram(keys=keys_in_order)

    data = [
        Sample(
            bin_str="".join(map(str, state)),
            probability=count / sum(counts.values()),
            nb_qubits=nb_qubits,
        )
        for (state, count) in counts.items()
    ]

    shot = job.measure.shots if job.measure is not None else 0
    return Result(job, data, None, shot)


def extract_result_STATE_VECTOR(
    result: StateVectorTrialResult,
    job: Job,
    device: Optional[GOOGLEDevice] = None,
) -> Result:
    """
    Extracts the result from a state vector-based job.

    Args:
        result : The result of the simulation.
        job : The original job.
        device : The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.
    """
    state_vector = result.final_state_vector
    state_vector = StateVector(
        state_vector, job.circuit.nb_qubits, state_vector_to_probabilities(state_vector)
    )
    return Result(job, state_vector, 0, 0)


def extract_result_OBSERVABLE(
    result: list[float] | list[ObservableMeasuredResult],
    job: Job,
    device: Optional[GOOGLEDevice] = None,
) -> Result:
    """
    Extracts the result from an observable-based job.

    Args:
        result : The result of the simulation.
        job : The original job.
        device : The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.
    """
    mean = 0.0
    variance = 0.0
    if job.measure is None:
        raise NotImplementedError("job.measure is None")
    for result1 in result:
        if isinstance(result1, float) or isinstance(result1, complex):
            mean += abs(result1)
        if isinstance(result1, ObservableMeasuredResult):
            mean += result1.mean
            # 3M-TODO variance not supported variance += result1.variance
    return Result(job, mean, variance, job.measure.shots)
