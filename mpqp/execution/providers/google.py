from __future__ import annotations
from typing import Optional

from typeguard import typechecked

from mpqp.execution.devices import GOOGLEDevice
from mpqp.execution.job import JobType, Job
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.qasm import qasm2_to_cirq_Circuit


from mpqp.core.instruction.measurement import ComputationalBasis
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure

from mpqp import Language

from cirq.work.observable_measurement_data import ObservableMeasuredResult
from cirq.sim.state_vector_simulator import StateVectorTrialResult
from cirq.transformers.optimize_for_target_gateset import optimize_for_target_gateset
from cirq.value.probability import state_vector_to_probabilities
from cirq.transformers.routing.route_circuit_cqc import RouteCQC
from cirq.sim.sparse_simulator import Simulator
from cirq.circuits.circuit import Circuit as cirq_circuit
from cirq.study.result import Result as cirq_result
from cirq.ops.linear_combinations import PauliSum as Cirq_PauliSum 
from cirq.transformers.target_gatesets.sqrt_iswap_gateset import SqrtIswapTargetGateset
from cirq_google.engine.virtual_engine_factory import load_median_device_calibration, create_device_from_processor_id
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from qsimcirq.qsim_simulator import QSimSimulator
from cirq.work.observable_measurement import (
    measure_observables,
    RepetitionsStoppingCriteria,
)


@typechecked
def run_google(job: Job) -> Result:
    """
    Execute the job on the right Google device precised in the job in parameter.
    This function is not meant to be used directly, please use ``runner.run(...)`` instead.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
    """
    return run_local(job) if not job.device.is_remote() else run_google_remote(job)


@typechecked
def run_google_remote(job: Job) -> Result:
    raise NotImplementedError("run_google_remote is not yet implemented")


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
    if type(job.device) != GOOGLEDevice:
        raise ValueError("Job device must be GOOGLEDevice")

    cirq_circuit = qasm2_to_cirq_Circuit(job.circuit.to_qasm2())
    sim = Simulator()

    if job.job_type == JobType.STATE_VECTOR:
        if job.device.is_processor():
            raise NotImplementedError(
                f"Does not handle {job.job_type} for processor for the moment"
            )
        result_sim = sim.simulate(cirq_circuit)
        result = extract_result(result_sim, job, GOOGLEDevice.CIRQ)
    elif job.job_type == JobType.SAMPLE:
        assert isinstance(job.measure, BasisMeasure)

        if isinstance(job.measure.basis, ComputationalBasis):
            if job.device.is_processor():
                cirq_circuit, sim = circuit_to_processor_cirq_Circuit(
                    job.device.value, cirq_circuit
                )
                result_sim = sim.get_sampler(job.device.value).run(
                    cirq_circuit, repetitions=job.measure.shots
                )
            else:
                result_sim = sim.run(cirq_circuit, repetitions=job.measure.shots)
        else:
            raise NotImplementedError(
                "Does not handle other basis than the ComputationalBasis for the moment"
            )
        result = extract_result(result_sim, job, GOOGLEDevice.CIRQ)
    elif job.job_type == JobType.OBSERVABLE:
        assert isinstance(job.measure, ExpectationMeasure)
        cirq_obs = job.measure.observable.to_other_language(
            language=Language.CIRQ, circuit=cirq_circuit
        )
        
        if type(cirq_obs) != Cirq_PauliSum:
            raise ValueError("cirq_obs must be a Cirq_PauliSum object")

        if job.measure.shots == 0:
            result_sim = sim.simulate_expectation_values(
                cirq_circuit, observables=cirq_obs
            )
        else:
            result_sim = measure_observables(
                cirq_circuit,
                cirq_obs, # type: ignore[reportArgumentType]
                sim,
                stopping_criteria=RepetitionsStoppingCriteria(job.measure.shots),
            )
        result = extract_result(result_sim, job, GOOGLEDevice.CIRQ)
    else:
        raise ValueError(f"Job type {job.job_type} not handled")

    return result


@typechecked
def circuit_to_processor_cirq_Circuit(processor_id: str, cirq_circuit: cirq_circuit):
    """
    Converts a Cirq circuit to be suitable for simulation on a specific processor.

    Args:
        processor_id : Identifier of the processor.
        cirq_circuit : The Cirq circuit to be converted.

    Returns:
        The converted Cirq circuit and the simulated local engine.

    Raises:
        ValueError: If the device metadata is not available for the specified processor.

    Warnings:
        This function optimizes the input circuit for the target gateset, routes the circuit according to the processor's connectivity,
        and validates the circuit against the device's constraints before simulation.
    """
    cal = load_median_device_calibration(processor_id)
    # noise_props = noise_properties_from_calibration(cal)
    # noise_model = NoiseModelFromGoogleNoiseProperties(noise_props)
    sim = QSimSimulator(noise=None)

    device = create_device_from_processor_id(processor_id)

    if device.metadata is None:
        raise ValueError(
            f"Device {device} does not have metadata for processor {processor_id}"
        )

    router = RouteCQC(device.metadata.nx_graph)

    rcirc, initial_map, swap_map = router.route_circuit(cirq_circuit) # type: ignore[reportUnusedVariable]

    fcirc = optimize_for_target_gateset(rcirc, gateset=SqrtIswapTargetGateset())

    device.validate_circuit(fcirc)

    sim_processor = SimulatedLocalProcessor(
        processor_id=processor_id,
        sampler=sim,
        device=device,
        calibrations={cal.timestamp // 1000: cal},
    )
    sim_engine = SimulatedLocalEngine([sim_processor])

    return fcirc, sim_engine


def extract_result(
    result: StateVectorTrialResult | cirq_result | list[float] | list[ObservableMeasuredResult],
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
                    f"result must be a cirq_result for job type {job.job_type}"
                )
            return extract_result_SAMPLE(result, job, device)
        elif job.job_type == JobType.STATE_VECTOR:
            if not isinstance(result, StateVectorTrialResult):
                raise ValueError(
                    f"result must be a cirq_result for job type {job.job_type}"
                )
            return extract_result_STATE_VECTOR(result, job, device)
        elif job.job_type == JobType.OBSERVABLE:
            if isinstance(result, cirq_result):
                raise ValueError(
                    f"result must be a list[float] | list[ObservableMeasuredResult] for job type {job.job_type}"
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
        device (: The device used for the simulation. Defaults to None.

    Returns:
        Result: The formatted result.
    """
    mean = 0.0
    variance = 0.0
    if job.measure is None:
        raise NotImplementedError("job.measure is None")
    for result1 in result:
            if isinstance(result1, float):
                mean += abs(result1)
            if isinstance(result1, ObservableMeasuredResult):
                mean += result1.mean
                # TODO variance not supported variance += result1.variance
    return Result(job, mean, variance, job.measure.shots)
