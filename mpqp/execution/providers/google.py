from __future__ import annotations

from typing import TYPE_CHECKING, Union

from mpqp.core.instruction.gates.native_gates import NativeGate
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.tools.errors import DeviceJobIncompatibleError

if TYPE_CHECKING:
    from cirq.sim.state_vector_simulator import StateVectorTrialResult
    from cirq.study.result import Result as CirqResult
    from cirq.circuits.circuit import Circuit as CirqCircuit
    from cirq.sim.sparse_simulator import Simulator
    from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
    from cirq_ionq import Service

from cirq.circuits.circuit import Circuit
from typeguard import typechecked

from mpqp import Language
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.execution.devices import GOOGLEDevice
from mpqp.execution.job import Job, JobType
from mpqp.execution.result import Result, Sample, StateVector
from mpqp.noise import NoiseModel


@typechecked
def apply_noise_to_cirq_circuit(
    cirq_circuit: "Circuit",
    noises: list[NoiseModel],
) -> "Circuit":
    """Apply noise models to a Cirq circuit.

    This function applies noise models to a given Cirq circuit based on the
    specified noise models and the number of qubits in the circuit. It constructs
    a new circuit by adding noise operations after the original gates and
    returns the circuit with the noise applied.

    Args:
        cirq_circuit: The Cirq circuit to apply noise to.
        noises: The noise models to apply to the circuit.

    Returns:
        A new circuit with the noise operations applied.
    """
    from cirq.circuits.moment import Moment
    from cirq.ops.identity import IdentityGate
    from cirq.ops.measurement_gate import MeasurementGate
    from cirq.ops.raw_types import Gate, Operation

    from mpqp.noise import DimensionalNoiseModel

    qubits = sorted(cirq_circuit.all_qubits())
    noisy_moments = []

    allowed_gates: dict[NoiseModel, set[type[NativeGate]]] = {}
    for noise in noises:
        gates: set[type[NativeGate]] = set()
        for gate in noise.gates:
            cirq_gate = gate.cirq_gate
            gate_cls = type(cirq_gate) if not isinstance(cirq_gate, type) else cirq_gate
            gates.add(gate_cls)
        allowed_gates[noise] = gates
    converted_noises: dict[NoiseModel, Gate] = (
        {  # pyright: ignore[reportAssignmentType]
            noise: noise.to_other_language(Language.CIRQ) for noise in noises
        }
    )

    for moment in cirq_circuit:
        moment_ops = list(moment.operations)
        noisy_moments.append(Moment(moment_ops))

        qubit_noise_op: list[list[Operation]] = [[] for _ in range(len(qubits))]

        for op in moment_ops:
            if isinstance(op.gate, (MeasurementGate, IdentityGate)):
                continue

            for noise in reversed(noises):
                noise_dimension = (
                    noise.dimension if isinstance(noise, DimensionalNoiseModel) else 1
                )

                if len(noise.targets) == 0 or len(noise.targets) == len(qubits):
                    target_qubits = qubits
                else:
                    target_qubits = [qubits[i] for i in noise.targets]

                if (
                    noise_dimension == 2
                    and len(op.qubits) == 2
                    and all(q in target_qubits for q in op.qubits)
                    and (
                        len(allowed_gates[noise]) == 0
                        or isinstance(op.gate, tuple(allowed_gates[noise]))
                    )
                ):
                    noisy_gate = converted_noises[noise].on(*op.qubits)
                    noisy_moments.append(Moment([noisy_gate]))

                elif noise_dimension == 1:
                    for q in op.qubits:
                        if q in target_qubits and (
                            len(allowed_gates[noise]) == 0
                            or isinstance(op.gate, tuple(allowed_gates[noise]))
                        ):
                            qubit_index = qubits.index(q)
                            qubit_noise_op[qubit_index].append(
                                converted_noises[noise].on(q)
                            )

        noisy_moments += [
            Moment(
                [ops[moment_index] for ops in qubit_noise_op if len(ops) > moment_index]
            )
            for moment_index in range(max(len(ops) for ops in qubit_noise_op))
        ]

    return Circuit(noisy_moments)


@typechecked
def run_google(job: Job, translation_warning: bool = True) -> Result:
    """Executes the job on the right Google device precised in the job in
    parameter.

    Args:
        job: Job to be executed.
        translation_warning: If `True`, a warning will be raised.

    Returns:
        A Result after submission and execution of the job.
        Note:

    Note:
        This function is not meant to be used directly, please use
        :func:`~mpqp.execution.runner.run` instead.
    """
    return (
        run_local(job, translation_warning)
        if not job.device.is_remote()
        else run_google_remote(job, translation_warning)
    )


@typechecked
def run_cirq_observable(
    job: Job,
    circuit: "CirqCircuit",
    simulator: Union["Simulator", "SimulatedLocalEngine"],
) -> Result:
    """Returns the result of an OBSERVABLE job.
    Here optimize_measurement allows cirq to do the grouping on the Pauli string of every observable.
    Otherwise each observables are sent without modification.

    This function should be called by run_local and run_local_processor, not with any remote jobs.
    Args:
        job: Job to be executed.
        circuit: The circuit to measure.
        simulator: The simulator on which the circuit is ran.

    Returns:
        A result containing the expectation values of the observables.
    """
    from cirq.ops.pauli_string import PauliString as CirqPauliString
    from cirq.work.observable_measurement import (
        RepetitionsStoppingCriteria,
        measure_observables,
    )
    from mpqp.execution.job import JobStatus

    variances = {}
    if job.measure is None:
        raise NotImplementedError("job.measure is None")
    assert isinstance(job.measure, ExpectationMeasure)

    # TODO: optimize
    if job.measure.optimize_measurement:
        monomials = []
        for obs in job.measure.observables:
            for monom in obs.pauli_string.monomials:
                found = False
                for m in monomials:
                    if monom.name == m.name:
                        found = True
                        break
                if not found:
                    monomials.append(monom / monom.coef)
        expectation_values: dict[str, float] = {}
        result: dict[str, float] = {}

        cirq_obs = [
            monom.to_other_language(Language.CIRQ, circuit=circuit)
            for monom in monomials
        ]
        job.status = JobStatus.RUNNING
        from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine

        if isinstance(simulator, SimulatedLocalEngine):
            local_result = simulator.get_sampler(
                job.device.value
            ).sample_expectation_values(
                circuit,
                observables=cirq_obs,
                num_samples=job.measure.shots,
            )
            for i, res in enumerate(local_result[0]):
                expectation_values.update({f"{monomials[i].name}": res})
                variances.update({f"{monomials[i].name}": None})
        elif job.measure.shots == 0:
            local_result = simulator.simulate_expectation_values(
                circuit, observables=cirq_obs
            )
            for i, res in enumerate(local_result):
                expectation_values.update({f"{monomials[i].name}": res.real})
                variances.update({f"{monomials[i].name}": 0})
        else:
            local_result = measure_observables(
                circuit,
                observables=cirq_obs,
                sampler=simulator,
                stopping_criteria=RepetitionsStoppingCriteria(job.measure.shots),
            )
            for i, res in enumerate(local_result):
                expectation_values.update({f"{monomials[i].name}": res.mean})
                variances.update({f"{monomials[i].name}": res.variance})
        errors = {}
        for i, obs in enumerate(job.measure.observables):
            string = obs.pauli_string
            local: float = 0.0
            var = {}

            for monoms in string.monomials:
                if TYPE_CHECKING:
                    assert isinstance(monoms.coef, (float, int))
                local += expectation_values[monoms.name] * monoms.coef
                var.update({monoms.name: variances[monoms.name]})
            errors.update({f"observable_{i}": var})
            result.update({f"observable_{i}": local})

        job.status = JobStatus.DONE
        if len(result) == 1:
            return Result(
                job,
                result["observable_0"],
                errors["observable_0"],
                shots=job.measure.shots,
            )
        return Result(job, result, errors, shots=job.measure.shots)

    else:
        errors = {}
        expectation_values = {}
        for obs in job.measure.observables:
            cirq_obs = obs.to_other_language(language=Language.CIRQ, circuit=circuit)
            job.status = JobStatus.RUNNING
            from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine

            if isinstance(simulator, SimulatedLocalEngine):
                local_result = simulator.get_sampler(
                    job.device.value
                ).sample_expectation_values(
                    circuit + cirq_obs,
                    observables=cirq_obs,
                    num_samples=job.measure.shots,
                )
                mean = 0
                for res in local_result:
                    mean += sum(res) / len(local_result)
                errors.update({f"observable_{len(errors)}": 0})
                expectation_values.update(
                    {f"observable_{len(expectation_values)}": mean}
                )
            elif job.measure.shots == 0 and not isinstance(
                simulator, SimulatedLocalEngine
            ):
                results = simulator.simulate_expectation_values(
                    circuit, observables=cirq_obs
                )
                for i, res in enumerate(results):
                    errors.update({f"observable_{len(errors)}": 0})
                    expectation_values.update(
                        {f"observable_{len(expectation_values)}": res.real}
                    )
            else:
                results = measure_observables(
                    circuit,
                    observables=(  # pyright: ignore[reportArgumentType]
                        [cirq_obs]
                        if isinstance(cirq_obs, CirqPauliString)
                        else cirq_obs
                    ),
                    sampler=simulator,
                    stopping_criteria=RepetitionsStoppingCriteria(job.measure.shots),
                )
                pauli_mono = PauliString.from_other_language(
                    [r.observable for r in results], job.measure.nb_qubits
                )
                if TYPE_CHECKING:
                    assert isinstance(pauli_mono, list)
                variances = {pm: r.variance for pm, r in zip(pauli_mono, results)}
                expectation_values.update(
                    {
                        f"observable_{len(expectation_values)}": sum(
                            map(lambda r: r.mean, results)
                        )
                    }
                )
                errors.update({f"observable_{len(errors)}": variances})
    job.status = JobStatus.DONE
    if len(expectation_values) == 1:
        return Result(
            job,
            expectation_values["observable_0"],
            errors['observable_0'],
            job.measure.shots,
        )
    return Result(
        job,
        expectation_values,
        errors,
        job.measure.shots,
    )


@typechecked
def run_cirq_observable_remote(
    job: Job, circuit: "CirqCircuit", service: "Service"
) -> Result:
    """Returns the result of an OBSERVABLE job.
    Here optimize_measurements will performs the whole grouping of the pauli monomials in mpqp.
    This function is not available without optimize_measurement at True.

    This function should be called by run_remote, not with any local jobs.
    Args:
        job: Job to be executed.
        circuit: The circuit to measure.
        service: The service on which the circuit is ran.

    Returns:
        A result containing the expectation values of the observables.
    """
    expectation_values = {}
    result = {}
    if job.measure is None:
        raise NotImplementedError("job.measure is None")
    assert isinstance(job.measure, ExpectationMeasure)
    grouping = job.measure.get_pauli_grouping()
    from mpqp.tools.pauli_grouping import (
        find_qubitwise_rotations,
        pauli_monomial_eigenvalues,
    )

    for group in grouping:
        pre_measure = QCircuit(find_qubitwise_rotations(group)).to_other_language(
            Language.CIRQ
        )
        local_result = extract_result_SAMPLE(
            service.run(circuit=circuit + pre_measure, repetitions=job.measure.shots),
            job,
        )
        assert isinstance(local_result, Result)
        for monom in group:
            import numpy as np

            expectation_value: float = np.dot(
                pauli_monomial_eigenvalues(monom), local_result.probabilities
            )
            expectation_values.update({monom.name: expectation_value})

    for i, obs in enumerate(job.measure.observables):
        string = obs.pauli_string
        local: float = 0
        for monoms in string.monomials:
            assert isinstance(monoms.coef, (int, float))
            local += expectation_values[monoms.name] * monoms.coef
        result.update({f"observable_{i}": local})
    if len(result) == 1:
        return Result(job, result["observable_0"])
    return Result(job, result)


@typechecked
def run_google_remote(job: Job, translation_warning: bool = True) -> Result:
    """Executes the job remotely on a Google quantum device. At present, only
    IonQ devices are supported.

    Args:
        job: Job to be executed, it MUST be corresponding to a
            :class:`~mpqp.execution.devices.GOOGLEDevice`.
        translation_warning: If `True`, a warning will be raised.

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

    if job.circuit.transpiled_circuit is None:
        job_CirqCircuit = job.circuit.to_other_device(job.device, translation_warning)
    else:
        job_CirqCircuit = job.circuit.transpiled_circuit

    if TYPE_CHECKING:
        assert isinstance(job_CirqCircuit, CirqCircuit)

    if job.device.is_ionq():
        from mpqp.execution.connection.env_manager import load_env_variables

        load_env_variables()
        service = ionq.Service(default_target=job.device.value)
        if job.job_type == JobType.SAMPLE:
            if TYPE_CHECKING:
                assert isinstance(job.measure, BasisMeasure)

            return extract_result_SAMPLE(
                service.run(circuit=job_CirqCircuit, repetitions=job.measure.shots), job
            )
        elif job.job_type == JobType.OBSERVABLE:
            return run_cirq_observable_remote(job, job_CirqCircuit, service)
        else:
            raise ValueError(
                f"{job.device}: job_type must be {JobType.SAMPLE} or {JobType.OBSERVABLE} but got job type {job.job_type}"
            )

    else:
        raise NotImplementedError(
            f"{job.device} is not handled for the moment. Only IonQ is supported"
        )


@typechecked
def run_local(job: Job, translation_warning: bool = True) -> Result:
    """Executes the job locally.

    Args:
        job : Job to be executed, it MUST be corresponding to a
            :class:`~mpqp.execution.devices.GOOGLEDevice`.
        translation_warning: If `True`, a warning will be raised.
        If `True`, a warning will be raised.

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
    from cirq.sim.sparse_simulator import Simulator

    if job.device.is_processor():
        return run_local_processor(job)

    if job.circuit.transpiled_circuit is None:
        if job.job_type == JobType.STATE_VECTOR:
            # 3M-TODO: careful, if we ever support several measurements, the
            # line bellow will have to changer
            circuit = job.circuit.without_measurements() + job.circuit.pre_measure()
            cirq_circuit = circuit.to_other_device(job.device, translation_warning)
            job.circuit.gphase = circuit.gphase
        else:
            cirq_circuit = job.circuit.to_other_device(job.device, translation_warning)
    else:
        cirq_circuit = job.circuit.transpiled_circuit

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
        return run_cirq_observable(job, cirq_circuit, simulator)
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

    if job.circuit.transpiled_circuit is None:
        cirq_circuit = job.circuit.to_other_device(job.device)
    else:
        cirq_circuit = job.circuit.transpiled_circuit

    if TYPE_CHECKING:
        assert isinstance(cirq_circuit, CirqCircuit)

    if job.job_type == JobType.STATE_VECTOR:
        raise NotImplementedError(
            f"Does not handle {job.job_type} for processor for the moment"
        )
    elif job.job_type == JobType.OBSERVABLE:

        if TYPE_CHECKING:
            assert isinstance(job.measure, ExpectationMeasure)

        if job.measure.shots == 0:
            raise DeviceJobIncompatibleError(
                f"Device {job.device.name} need shots != 0."
            )
        return run_cirq_observable(job, cirq_circuit, simulator)
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
    result: "CirqResult",
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
            bin_str="".join(map(lambda s: f'{s:0{nb_qubits}b}', state)),
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
    state_vector = StateVector(result.final_state_vector, job.circuit.nb_qubits)
    return Result(job, state_vector, 0, 0)
