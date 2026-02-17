"""Unit tests for the Job class and related enums (JobType, JobStatus).

Tests cover local functionality and mocked remote status polling.
"""

import numpy as np
import pytest
from unittest.mock import patch

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

    def test_construction_with_measure(self):
        measure = BasisMeasure([0, 1], shots=100)
        circuit = QCircuit([H(0), CNOT(0, 1), measure])
        job = Job(JobType.SAMPLE, circuit, IBMDevice.AER_SIMULATOR)
        assert job.measure is not None

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
# Job equality
# ---------------------------------------------------------------------------


class TestJobEquality:
    @pytest.mark.parametrize(
        "make_other, expected_equal",
        [
            pytest.param(
                lambda: Job(
                    JobType.STATE_VECTOR, QCircuit([H(0)]), IBMDevice.AER_SIMULATOR
                ),
                True,
                id="identical",
            ),
            pytest.param(
                lambda: Job(
                    JobType.SAMPLE,
                    QCircuit([H(0), BasisMeasure([0], shots=100)]),
                    IBMDevice.AER_SIMULATOR,
                ),
                False,
                id="different_type",
            ),
            pytest.param(
                lambda: Job(
                    JobType.STATE_VECTOR,
                    QCircuit([H(0)]),
                    IBMDevice.AER_SIMULATOR_STATEVECTOR,
                ),
                False,
                id="different_device",
            ),
            pytest.param(
                lambda: Job(
                    JobType.STATE_VECTOR, QCircuit([X(0)]), IBMDevice.AER_SIMULATOR
                ),
                False,
                id="different_circuit",
            ),
        ],
    )
    def test_equality(self, make_other, expected_equal):
        job = Job(JobType.STATE_VECTOR, QCircuit([H(0)]), IBMDevice.AER_SIMULATOR)
        assert (job == make_other()) == expected_equal

    @pytest.mark.parametrize(
        "other", ["not a job", 42, None], ids=["str", "int", "None"]
    )
    def test_not_equal_to_non_job(self, other):
        job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        assert job != other


# ---------------------------------------------------------------------------
# Job repr / to_dict
# ---------------------------------------------------------------------------


class TestJobReprAndDict:
    def test_repr(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        expected = (
            f"Job({JobType.STATE_VECTOR}, {repr(circuit)}, {IBMDevice.AER_SIMULATOR})"
        )
        assert repr(job) == expected

    def test_to_dict(self):
        circuit = QCircuit(2)
        job = Job(JobType.STATE_VECTOR, circuit, IBMDevice.AER_SIMULATOR)
        d = job.to_dict()
        assert d["job_type"] == JobType.STATE_VECTOR
        assert d["circuit"] is circuit
        assert d["device"] == IBMDevice.AER_SIMULATOR
        assert d["measure"] is None
        assert d["id"] is None
        assert d["status"] == JobStatus.INIT


# ---------------------------------------------------------------------------
# Remote job status (mocked)
# ---------------------------------------------------------------------------


class TestRemoteStatus:
    @pytest.mark.parametrize(
        "device, status_func, remote_status",
        [
            pytest.param(
                IBMDevice.IBM_LEAST_BUSY,
                "mpqp.execution.job.get_ibm_job_status",
                JobStatus.RUNNING,
                id="ibm_running",
            ),
            pytest.param(
                IBMDevice.IBM_LEAST_BUSY,
                "mpqp.execution.job.get_ibm_job_status",
                JobStatus.QUEUED,
                id="ibm_queued",
            ),
            pytest.param(
                ATOSDevice.QLM_LINALG,
                "mpqp.execution.job.get_qlm_job_status",
                JobStatus.RUNNING,
                id="qlm_running",
            ),
        ],
    )
    def test_polling(self, device, status_func, remote_status):
        """Accessing .status on a non-terminal remote job should query the provider."""
        job = Job(JobType.STATE_VECTOR, QCircuit(2), device)
        job.id = "fake-remote-id"
        with patch(status_func, return_value=remote_status) as mock_fn:
            assert job.status == remote_status
            mock_fn.assert_called_once_with("fake-remote-id")

    @pytest.mark.parametrize(
        "terminal", [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]
    )
    def test_terminal_status_skips_polling(self, terminal):
        """Once a remote job reaches a terminal state, no provider call should happen."""
        job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.IBM_LEAST_BUSY)
        job.id = "fake-remote-id"
        job.status = terminal
        with patch("mpqp.execution.job.get_ibm_job_status") as mock_fn:
            assert job.status == terminal
            mock_fn.assert_not_called()

    def test_local_device_never_polls(self):
        """For a local simulator, the status property should never trigger a remote call."""
        job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        job.status = JobStatus.RUNNING
        with patch("mpqp.execution.job.get_ibm_job_status") as mock_fn:
            assert job.status == JobStatus.RUNNING
            mock_fn.assert_not_called()


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
