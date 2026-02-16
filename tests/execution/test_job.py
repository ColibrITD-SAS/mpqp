"""Unit tests for the Job class and related enums (JobType, JobStatus).

These tests cover local, non-remote functionality only — no SDK connections
are required.

# 3M-TODO
# Remote job status polling tests need a stable account or mocked connections.
"""

import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import (
    ExpectationMeasure,
    Observable,
)
from mpqp.execution.devices import (
    ATOSDevice,
    AWSDevice,
    GOOGLEDevice,
    IBMDevice,
)
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.runner import generate_job
from mpqp.gates import CNOT, H, X

# ---------------------------------------------------------------------------
# JobStatus enum
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_all_statuses_exist(self):
        expected = {"INIT", "QUEUED", "RUNNING", "CANCELLED", "ERROR", "DONE"}
        actual = {s.name for s in JobStatus}
        assert expected == actual

    def test_terminal_statuses(self):
        """DONE, ERROR, CANCELLED should be considered terminal."""
        terminal = {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED}
        non_terminal = {JobStatus.INIT, JobStatus.QUEUED, JobStatus.RUNNING}
        assert terminal.isdisjoint(non_terminal)


# ---------------------------------------------------------------------------
# JobType enum
# ---------------------------------------------------------------------------


class TestJobType:
    def test_all_types_exist(self):
        expected = {"STATE_VECTOR", "SAMPLE", "OBSERVABLE"}
        actual = {t.name for t in JobType}
        assert expected == actual

    def test_state_vector_accepts_basis_measure(self):
        assert BasisMeasure in JobType.STATE_VECTOR.value

    def test_state_vector_accepts_none(self):
        assert type(None) in JobType.STATE_VECTOR.value

    def test_sample_accepts_basis_measure(self):
        assert BasisMeasure in JobType.SAMPLE.value

    def test_observable_accepts_expectation_measure(self):
        assert ExpectationMeasure in JobType.OBSERVABLE.value


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------


class TestJobConstruction:
    def test_basic_construction(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        assert job.job_type == JobType.STATE_VECTOR
        assert job.circuit is circuit
        assert job.device == IBMDevice.AER_SIMULATOR
        assert job.measure is None
        assert job.id is None

    def test_initial_status_is_init(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        assert job._status == JobStatus.INIT

    def test_construction_with_measure(self):
        circuit = QCircuit([H(0), CNOT(0, 1)])
        measure = BasisMeasure([0, 1], shots=100)
        job = Job(JobType.SAMPLE, circuit, IBMDevice.AER_SIMULATOR, measure)
        assert job.measure is not None
        # measure should be deep-copied
        assert job.measure is not measure

    def test_construction_with_different_devices(self):
        circuit = QCircuit(2)
        devices = [
            IBMDevice.AER_SIMULATOR,
            ATOSDevice.MYQLM_PYLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        ]
        for device in devices:
            job = Job(JobType.STATE_VECTOR, circuit, device)
            assert job.device == device


# ---------------------------------------------------------------------------
# Job status property
# ---------------------------------------------------------------------------


class TestJobStatusProperty:
    def test_status_setter(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        job.status = JobStatus.RUNNING
        assert job._status == JobStatus.RUNNING

    def test_local_status_returns_directly(self):
        """For local (non-remote) devices, the status property should return
        the stored status without attempting a remote call."""
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        job.status = JobStatus.DONE
        # Should not raise — no remote lookup attempted for local device
        assert job.status == JobStatus.DONE

    def test_terminal_status_no_remote_check(self):
        """Once a job reaches a terminal state, the status property should
        return immediately for any device."""
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        for terminal in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
            job.status = terminal
            assert job.status == terminal


# ---------------------------------------------------------------------------
# Job equality
# ---------------------------------------------------------------------------


class TestJobEquality:
    def test_equal_jobs(self):
        circuit = QCircuit([H(0), CNOT(0, 1)])
        job1 = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        job2 = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        assert job1 == job2

    def test_different_type(self):
        circuit = QCircuit([H(0)])
        measure = BasisMeasure([0], shots=100)
        job1 = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        job2 = Job(JobType.SAMPLE, circuit, IBMDevice.AER_SIMULATOR, measure)
        assert job1 != job2

    def test_different_device(self):
        circuit = QCircuit([H(0)])
        job1 = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        job2 = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR_STATEVECTOR)
        assert job1 != job2

    def test_different_circuit(self):
        circ1 = QCircuit([H(0)])
        circ2 = QCircuit([X(0)])
        job1 = Job(JobType.STATE_VECTOR, circ1, IBMDevice.AER_SIMULATOR)
        job2 = Job(JobType.STATE_VECTOR, circ2, IBMDevice.AER_SIMULATOR)
        assert job1 != job2

    def test_not_equal_to_non_job(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        assert job != "not a job"
        assert job != 42
        assert job != None


# ---------------------------------------------------------------------------
# Job repr / to_dict
# ---------------------------------------------------------------------------


class TestJobRepr:
    def test_repr_no_measure(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        r = repr(job)
        assert "Job(" in r
        assert "STATE_VECTOR" in r
        assert "AER_SIMULATOR" in r

    def test_repr_with_measure(self):
        circuit = QCircuit([H(0)])
        measure = BasisMeasure([0], shots=100)
        job = Job(JobType.SAMPLE, circuit, IBMDevice.AER_SIMULATOR, measure)
        r = repr(job)
        assert "BasisMeasure" in r

    def test_to_dict_keys(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        d = job.to_dict()
        assert "job_type" in d
        assert "circuit" in d
        assert "device" in d
        assert "measure" in d
        assert "id" in d
        assert "status" in d

    def test_to_dict_values(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        d = job.to_dict()
        assert d["job_type"] == JobType.STATE_VECTOR
        assert d["device"] == IBMDevice.AER_SIMULATOR
        assert d["measure"] is None
        assert d["id"] is None


# ---------------------------------------------------------------------------
# generate_job (from runner.py)
# ---------------------------------------------------------------------------


class TestGenerateJob:
    def test_no_measurement_gives_state_vector(self):
        circuit = QCircuit([H(0), CNOT(0, 1)])
        job = generate_job(circuit, IBMDevice.AER_SIMULATOR)
        assert job.job_type == JobType.STATE_VECTOR
        assert job.measure is None

    def test_basis_measure_zero_shots_gives_state_vector(self):
        circuit = QCircuit([H(0), BasisMeasure([0], shots=0)])
        job = generate_job(circuit, IBMDevice.AER_SIMULATOR)
        assert job.job_type == JobType.STATE_VECTOR
        assert job.measure is not None

    def test_basis_measure_positive_shots_gives_sample(self):
        circuit = QCircuit([H(0), BasisMeasure([0], shots=1000)])
        job = generate_job(circuit, IBMDevice.AER_SIMULATOR)
        assert job.job_type == JobType.SAMPLE
        assert job.measure is not None

    def test_expectation_measure_gives_observable(self):
        obs = Observable(np.array([[1, 0], [0, -1]]))
        circuit = QCircuit([H(0), ExpectationMeasure(obs, [0], shots=0)])
        job = generate_job(circuit, IBMDevice.AER_SIMULATOR)
        assert job.job_type == JobType.OBSERVABLE

    def test_multiple_measurements_raises(self):
        circuit = QCircuit([H(0), CNOT(0, 1)])
        circuit.add(BasisMeasure([0], shots=100))
        circuit.add(BasisMeasure([1], shots=100))
        with pytest.raises(NotImplementedError, match="multiple measurements"):
            generate_job(circuit, IBMDevice.AER_SIMULATOR)

    def test_symbolic_substitution(self):
        from sympy import symbols

        theta = symbols("θ")
        from mpqp.gates import Rx

        circuit = QCircuit([Rx(theta, 0)])
        job = generate_job(circuit, IBMDevice.AER_SIMULATOR, {theta: 1.5})
        # After substitution the circuit should have no free variables
        assert len(job.circuit.variables()) == 0
