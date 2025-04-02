from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

from mpqp.tools.errors import DeviceJobIncompatibleError

if TYPE_CHECKING:
    from cirq.sim.state_vector_simulator import StateVectorTrialResult
    from cirq.study.result import Result as CirqResult
    from cirq.work.observable_measurement_data import ObservableMeasuredResult

from typeguard import typechecked

from mpqp import Language
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.execution.devices import GOOGLEDevice
from mpqp.execution.job import Job, JobType
from mpqp.execution.result import Result, Sample, StateVector


@typechecked
def run_google(job: Job) -> Result:
    """Executes the job on the right Google device precised in the job in
    parameter.

    Args:
        job: Job to be executed.

    Returns:
        A Result after submission and execution of the job.
        Note:

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    return run_local(job) if not job.device.is_remote() else run_google_remote(job)


@typechecked
def run_google_remote(job: Job) -> Result:
    """Executes the job remotely on a Google quantum device. At present, only
    IonQ devices are supported.

    Args:
        job: Job to be executed, it MUST be corresponding to a
            :class:`~mpqp.execution.devices.GOOGLEDevice`.

    Returns:
        The result after submission and execution of the job.

    Raises:
        ValueError: If the job's device is not an instance of GOOGLEDevice.
        NotImplementedError: If the job's device is not supported (only IonQ
            devices are supported currently).
        NotImplementedError: If the job type or basis measure is not supported.
    """
    if not isinstance(job.device, GOOGLEDevice):
        raise ValueError(
            "`job` must correspond to an `GOOGLEDevice`, but corresponds to a "
            f"{job.device} instead"
        )

    import cirq_ionq as ionq
    from cirq.circuits.circuit import Circuit as CirqCircuit
    from cirq.devices.line_qubit import LineQubit
    from cirq.transformers.optimize_for_target_gateset import (
        optimize_for_target_gateset,
    )
    from cirq_ionq.ionq_gateset import IonQTargetGateset

    job_CirqCircuit = job.circuit.to_other_language(Language.CIRQ)
    if TYPE_CHECKING:
        assert isinstance(job_CirqCircuit, CirqCircuit)

    if job.device.is_ionq():
        from mpqp.execution.connection.env_manager import load_env_variables

        load_env_variables()

        if job.job_type != JobType.SAMPLE:
            raise ValueError(
                f"{job.device}: job_type must be {JobType.SAMPLE} but got job type {job.job_type}"
            )

        service = ionq.Service(default_target=job.device.value)
        job_CirqCircuit = optimize_for_target_gateset(
            job_CirqCircuit, gateset=IonQTargetGateset()
        )
        job_CirqCircuit = job_CirqCircuit.transform_qubits(
            {qb: LineQubit(i) for i, qb in enumerate(job_CirqCircuit.all_qubits())}
        )

        if TYPE_CHECKING:
            assert isinstance(job.measure, BasisMeasure)
        return extract_result_SAMPLE(
            service.run(circuit=job_CirqCircuit, repetitions=job.measure.shots), job
        )
    else:
        raise NotImplementedError(
            f"{job.device} is not handled for the moment. Only IonQ is supported"
        )


@typechecked
def run_local(job: Job) -> Result:
    """Executes the job locally.

    Args:
        job : Job to be executed, it MUST be corresponding to a
            :class:`~mpqp.execution.devices.GOOGLEDevice`.

    Returns:
        The result after submission and execution of the job.

    Raises:
        ValueError: If the job device is not GOOGLEDevice.
    """
    if not isinstance(job.device, GOOGLEDevice):
        raise ValueError(
            "`job` must correspond to an `GOOGLEDevice`, but corresponds to a "
            f"{job.device} instead"
        )

    from cirq.circuits.circuit import Circuit as CirqCircuit

    # from cirq.ops.pauli_string import PauliString as CirqPauliString
    from cirq.sim.sparse_simulator import Simulator
    from cirq.work.observable_measurement import (
        RepetitionsStoppingCriteria,
        measure_observables,
    )

    if job.device.is_processor():
        return run_local_processor(job)

    if job.job_type == JobType.STATE_VECTOR:
        # 3M-TODO: careful, if we ever support several measurements, the
        # line bellow will have to changer
        circuit = job.circuit.without_measurements() + job.circuit.pre_measure()
        cirq_circuit = circuit.to_other_language(Language.CIRQ)
        job.circuit.gphase = circuit.gphase
    else:
        cirq_circuit = job.circuit.to_other_language(Language.CIRQ)
    if TYPE_CHECKING:
        assert isinstance(cirq_circuit, CirqCircuit)

    simulator = Simulator(noise=None)

    if job.job_type == JobType.STATE_VECTOR:
        return extract_result_STATE_VECTOR(simulator.simulate(cirq_circuit), job)
    elif job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert isinstance(job.measure, BasisMeasure)
        return extract_result_SAMPLE(
            simulator.run(cirq_circuit, repetitions=job.measure.shots), job
        )
    elif job.job_type == JobType.OBSERVABLE:
        from cirq.ops.linear_combinations import PauliSum as Cirq_PauliSum
        from cirq.ops.pauli_string import PauliString as Cirq_PauliString

        if TYPE_CHECKING:
            assert isinstance(job.measure, ExpectationMeasure)
        # TODO: update this to take into account the case when we have list of Observables
        # TODO: check if Cirq allows for a list of observable when computing expectation values (apparently yes)
        cirq_observables: list[Union[Cirq_PauliSum, Cirq_PauliString]] = []
        for obs in job.measure.observables:
            translated = obs.to_other_language(
                language=Language.CIRQ, circuit=cirq_circuit
            )

            if isinstance(translated, Cirq_PauliSum):
                if TYPE_CHECKING:
                    assert isinstance(translated, list)
                    assert all(
                        isinstance(item, Cirq_PauliString) for item in translated
                    )
                translated = next(iter(translated), None)

            if TYPE_CHECKING:
                assert isinstance(translated, (Cirq_PauliSum, Cirq_PauliString))

            if translated is not None:
                cirq_observables.append(translated)

        if job.measure.shots == 0:
            return extract_result_OBSERVABLE_ideal(
                simulator.simulate_expectation_values(
                    cirq_circuit, observables=cirq_observables
                ),
                job,
            )
        else:
            return extract_result_OBSERVABLE_shot_noise(
                # TODO: here precise the 'grouper' argument of measure_observable to precise the pauli grouping strategy
                measure_observables(
                    cirq_circuit,
                    observables=cirq_observables,
                    sampler=simulator,
                    stopping_criteria=RepetitionsStoppingCriteria(job.measure.shots),
                ),
                job,
            )
    else:
        raise ValueError(f"Job type {job.job_type} not handled")


@typechecked
def run_local_processor(job: Job) -> Result:
    """Executes the job locally on processor.

    Args:
        job : Job to be executed, it MUST be corresponding to a
            :class:`~mpqp.execution.devices.GOOGLEDevice`.

    Returns:
        The result after submission and execution of the job.
    """
    if not isinstance(job.device, GOOGLEDevice):
        raise ValueError(
            "`job` must correspond to an `GOOGLEDevice`, but corresponds to a "
            f"{job.device} instead"
        )

    from cirq.circuits.circuit import Circuit as CirqCircuit
    from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
    from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
    from cirq_google.engine.virtual_engine_factory import (
        create_device_from_processor_id,
        load_median_device_calibration,
    )
    from qsimcirq.qsim_simulator import QSimSimulator

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

    cirq_circuit = job.circuit.to_other_language(Language.CIRQ, job.device.value)
    if TYPE_CHECKING:
        assert isinstance(cirq_circuit, CirqCircuit)

    if job.job_type == JobType.STATE_VECTOR:
        raise NotImplementedError(
            f"Does not handle {job.job_type} for processor for the moment"
        )
    elif job.job_type == JobType.OBSERVABLE:
        from cirq.ops.linear_combinations import PauliSum as Cirq_PauliSum
        from cirq.ops.pauli_string import PauliString as Cirq_PauliString

        if TYPE_CHECKING:
            assert isinstance(job.measure, ExpectationMeasure)

        # TODO: update this to take into account the case when we have list of Observables
        cirq_observables: list[Union[Cirq_PauliSum, Cirq_PauliString]] = []
        for obs in job.measure.observables:
            translated = obs.to_other_language(
                language=Language.CIRQ, circuit=cirq_circuit
            )

            if isinstance(translated, Cirq_PauliSum):
                if TYPE_CHECKING:
                    assert isinstance(translated, list)
                    assert all(
                        isinstance(item, Cirq_PauliString) for item in translated
                    )
                translated = next(iter(translated), None)

            if TYPE_CHECKING:
                assert isinstance(translated, (Cirq_PauliSum, Cirq_PauliString))

            if translated is not None:
                cirq_observables.append(translated)

        if job.measure.shots == 0:
            raise DeviceJobIncompatibleError(
                f"Device {job.device.name} need shots != 0."
            )
        return extract_result_OBSERVABLE_processors(
            simulator.get_sampler(job.device.value).sample_expectation_values(
                cirq_circuit,
                observables=cirq_observables,
                num_samples=job.measure.shots,
            ),
            job,
        )
    elif job.job_type == JobType.SAMPLE:
        if TYPE_CHECKING:
            assert isinstance(job.measure, BasisMeasure)

        return extract_result_SAMPLE(
            simulator.get_sampler(job.device.value).run(
                cirq_circuit, repetitions=job.measure.shots
            ),
            job,
        )
    else:
        raise ValueError(f"Job type {job.job_type} not handled")


def extract_result_SAMPLE(
    result: CirqResult,
    job: Job,
) -> Result:
    """Extracts the result from a sample-based job.

    Args:
        result : The result of the simulation.
        job : The original job.

    Returns:
        The formatted result.
    """
    nb_qubits = job.circuit.nb_qubits

    keys_in_order = sorted(result.records.keys())
    counts = result.multi_measurement_histogram(keys=keys_in_order)

    data = [
        Sample(
            bin_str="".join(map(bin, state)),
            count=count,
            nb_qubits=nb_qubits,
        )
        for (state, count) in counts.items()
    ]

    shot = job.measure.shots if job.measure is not None else 0
    return Result(job, data, None, shot)


def extract_result_STATE_VECTOR(
    result: StateVectorTrialResult,
    job: Job,
) -> Result:
    """Extracts the result from a state vector-based job.

    Args:
        result : The result of the simulation.
        job : The original job.

    Returns:
        The formatted result.
    """
    from mpqp.tools.maths import normalize

    state_vector = normalize(result.final_state_vector)
    state_vector = StateVector(state_vector, job.circuit.nb_qubits)
    return Result(job, state_vector, 0, 0)


def extract_result_OBSERVABLE_processors(
    results: Sequence[Sequence[float]],
    job: Job,
) -> Result:
    """Process measurement results for an observable from a quantum job.

    Args:
        results : A sequence of measurement results, where
            each inner sequence represents a set of results for a particular shot.
        job: The original job.

    Returns:
        The formatted result.

    Raises:
        NotImplementedError: If the job does not contain a measurement (i.e.,
            ``job.measure`` is ``None``).
    """
    if job.measure is None:
        raise NotImplementedError("job.measure is None")

    mean = 0
    for res in results:
        mean += sum(res) / len(res)

    shots = job.measure.shots

    if len(results) == 1:
        return Result(job, mean, 0, shots)

    exp_values_dict = dict()
    errors_dict = dict()

    for i, _ in enumerate(results):
        label = (
            job.measure.observables[i].label
            if isinstance(job.measure, ExpectationMeasure)
            else f"cirq_obs_{i}"
        )
        exp_values_dict[label] = mean
        errors_dict[label] = 0

    return Result(job, exp_values_dict, errors_dict, shots)


def extract_result_OBSERVABLE_ideal(
    results: list[float],
    job: Job,
) -> Result:
    """Extracts the result from an observable-based ideal job.

    The simulation from which the result to parse comes from can take in several
    observables, and each observable will have a corresponding value in the
    result. But since we only support a single measure per circuit for now, we
    could simplify this function by only returning the first value.

    Note:
        for some reason, the values we retrieve from cirq are not always float,
        but sometimes are complex. This is likely due to numerical approximation
        since the complex part is always extremely small, so we just remove it,
        but this might result in slightly unexpected results.

    Args:
        results: The result of the simulation.
        job: The original job.

    Returns:
        The formatted result.
    """
    if job.measure is None:
        raise NotImplementedError("job.measure is None")

    mean = 0
    for r in results:
        mean += float(r.real)

    shots = job.measure.shots

    if len(results) == 1:
        return Result(job, mean, 0, shots)

    exp_values_dict = dict()
    errors_dict = dict()

    for i, r in enumerate(results):
        label = (
            job.measure.observables[i].label
            if isinstance(job.measure, ExpectationMeasure)
            else f"cirq_obs_{i}"
        )
        exp_values_dict[label] = float(r.real)
        errors_dict[label] = 0

    return Result(job, exp_values_dict, errors_dict, shots)


def extract_result_OBSERVABLE_shot_noise(
    results: list[ObservableMeasuredResult],
    job: Job,
) -> Result:
    """Extracts the result from an observable-based job.

    Args:
        results: The result of the simulation.
        job: The original job.

    Returns:
        The formatted result.
    """
    if job.measure is None:
        raise NotImplementedError("job.measure is None")

    mean = 0
    variance = 0
    for r in results:
        mean += float(r.mean)
        variance += float(r.variance)

    shots = job.measure.shots

    if len(results) == 1:
        return Result(job, mean, variance, shots)

    exp_values_dict = dict()
    errors_dict = dict()

    for i, r in enumerate(results):
        label = (
            job.measure.observables[i].label
            if isinstance(job.measure, ExpectationMeasure)
            else f"cirq_obs_{i}"
        )
        exp_values_dict[label] = float(r.mean)
        errors_dict[label] = float(r.variance)

    return Result(job, exp_values_dict, errors_dict, shots)
