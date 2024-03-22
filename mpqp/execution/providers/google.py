from typing import Optional

from typeguard import typechecked

from mpqp.execution.devices import GOOGLEDevice
from ..job import Job, JobType
from ..result import Result, Sample, StateVector
from mpqp.qasm import qasm2_to_cirq_Circuit

from mpqp.core.instruction.measurement import ComputationalBasis
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)

from cirq import Simulator, RouteCQC, optimize_for_target_gateset, state_vector_to_probabilities, circuits
from cirq import Result as cirq_result, SqrtIswapTargetGateset
from cirq_google import engine, noise_properties_from_calibration, NoiseModelFromGoogleNoiseProperties, SycamoreTargetGateset
from qsimcirq import QSimSimulator


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
    return run_local(job) if not job.device.is_remote() else print("none")


@typechecked
def run_local(job: Job) -> Result:

    cirq_circuit = qasm2_to_cirq_Circuit(job.circuit.to_qasm2())
    sim = Simulator()
    if job.device.is_processor():
        if job.job_type != JobType.SAMPLE:
            raise NotImplementedError(
                f"Does not handle {job.job_type} for processor for the moment"
            )
        cirq_circuit, sim = circuit_to_processor_cirq_Circuit(job.device.value, cirq_circuit)
    

    if job.job_type == JobType.STATE_VECTOR:
        result_sim = sim.simulate(cirq_circuit)
        result = extract_result_STATE_VECTOR(result_sim, job, GOOGLEDevice.CIRQ)
    elif job.job_type == JobType.SAMPLE:
        assert isinstance(job.measure, BasisMeasure)
        if isinstance(job.measure.basis, ComputationalBasis):
            if job.device.is_processor():
                result_sim = sim.get_sampler(job.device.value).run(cirq_circuit, repetitions=job.measure.shots)
            else :
                result_sim = sim.run(cirq_circuit, repetitions=job.measure.shots)
        else:
            raise NotImplementedError(
                "Does not handle other basis than the ComputationalBasis for the moment"
            )
        
        result = extract_result_SAMPLE(result_sim, job, GOOGLEDevice.CIRQ)
    elif job.job_type == JobType.OBSERVABLE:
        assert isinstance(job.measure, ExpectationMeasure)
        raise NotImplementedError(
                "Does not handle OBSERVABLE for the moment"
            )
        #result = sim.simulate_expectation_values(cirq_circuit, job.measure.observable)
    else:
        raise ValueError(f"Job type {job.job_type} not handled")

    
    return result

@typechecked
def circuit_to_processor_cirq_Circuit(processor_id: str, cirq_circuit: circuits):

    cal = engine.load_median_device_calibration(processor_id)
    noise_props = noise_properties_from_calibration(cal)
    noise_model = NoiseModelFromGoogleNoiseProperties(noise_props)
    sim = QSimSimulator(noise=noise_model)

    device = engine.create_device_from_processor_id(processor_id)

    router = RouteCQC(device.metadata.nx_graph)

    rcirc, initial_map, swap_map = router.route_circuit(cirq_circuit)

    fcirc = optimize_for_target_gateset(rcirc, gateset = SqrtIswapTargetGateset())

    device.validate_circuit(fcirc)

    sim_processor = engine.SimulatedLocalProcessor(
    processor_id=processor_id, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal}
    )
    sim_engine = engine.SimulatedLocalEngine([sim_processor])

    return fcirc, sim_engine


def extract_result_SAMPLE(
    result: cirq_result,
    job: Optional[Job] = None,
    device: Optional[GOOGLEDevice] = None,
) -> Result:
    """
    Parse a result from Cirq execution into an mpqp Result.

    Args:
        result: Result returned by Cirq after run of the circuit.
        job: Original mpqp circuit used to generate the run. Used to retrieve more easily info to instantiate the result.
        device: Cirq Device on which the circuit was simulated.
        repetitions: Number of repetitions for the circuit execution.

    Returns:
        A Result containing the result info extracted from the Cirq result.
    """
    nb_qubits = job.circuit.nb_qubits
    
    keys_in_order = sorted(result.records.keys())
    counts = result.multi_measurement_histogram(keys=keys_in_order)

    data = [
            Sample(
                bin_str=''.join(map(str, state)), probability=count/sum(counts.values()), nb_qubits=nb_qubits
            )
            for (state, count) in counts.items()
        ]
    return Result(job, data, None, job.measure.shots)




def extract_result_STATE_VECTOR(
    result: cirq_result,
    job: Optional[Job] = None,
    device: Optional[GOOGLEDevice] = None,
) -> Result:
    state_vector = result.final_state_vector
    state_vector = StateVector(state_vector, job.circuit.nb_qubits, state_vector_to_probabilities(state_vector))
    return Result(job, state_vector, 0, 0)


def extract_result_OBSERVABLE(
    result: cirq_result,
    job: Optional[Job] = None,
    device: Optional[GOOGLEDevice] = None,
) -> Result:
        shots = 0 if len(result.metadata[0]) == 0 else result.metadata[0]["shots"]
        variance = (
            None if len(result.metadata[0]) == 0 else result.metadata[0]["variance"]
        )
        return Result(job, result.values[0], variance, shots)