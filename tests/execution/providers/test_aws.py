"""Tests for the AWS Braket provider.

The direct-reservation tests in this module are unit tests, not remote execution
tests. They call the relevant MPQP functions while replacing AWS boundaries with
mocks: device lookup returns a fake device, task submission returns a fake task,
and reservation contexts only record their arguments. A remote ``AWSDevice`` is
used where appropriate to exercise the correct control flow, but no credentials,
network access, active reservation, or paid task is required.

A mock is a local stand-in that records how it is called and returns configured
values without executing the real AWS SDK operation.
"""

# 3M-TODO
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch

import pytest

import mpqp.execution.runner as runner
from mpqp import CNOT, AWSDevice, BasisMeasure, H, QCircuit, run
from mpqp.core.binding import CircuitBinding
from mpqp.execution.providers.aws import (
    optional_reservation_arn,
    submit_job_braket,
)
from mpqp.execution.providers.providers_params import BraketParams
from mpqp.execution.result import BatchResult


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circuit",
    [
        QCircuit([H(2)]),
        QCircuit([H(0), H(2)]),
        QCircuit([H(0), CNOT(1, 3), H(2)]),
        QCircuit([H(2)], nb_qubits=5),
    ],
)
def test_braket_non_contiguous_qubits(circuit: QCircuit):
    run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("braket")
def test_direct_reservation_uses_submitting_device(monkeypatch: pytest.MonkeyPatch):
    """Forward the exact Braket device and ARN to ``DirectReservation``."""
    braket_device = Mock()
    direct_reservation = Mock()
    reservation_context = MagicMock()
    direct_reservation.return_value = reservation_context
    fake_aws_module = ModuleType("braket.aws")
    fake_aws_module.__dict__["DirectReservation"] = direct_reservation
    monkeypatch.setitem(sys.modules, "braket.aws", fake_aws_module)

    with optional_reservation_arn(braket_device, "reservation-arn"):
        pass

    direct_reservation.assert_called_once_with(
        braket_device, reservation_arn="reservation-arn"
    )
    reservation_context.__enter__.assert_called_once_with()
    reservation_context.__exit__.assert_called_once()


@pytest.mark.provider("braket")
def test_submit_job_uses_direct_reservation_context():
    """Run the remote submission path inside the requested reservation context."""
    import mpqp.execution.providers.aws as aws_provider

    circuit = QCircuit([H(0), BasisMeasure([0], shots=10)])
    job = runner.generate_job(circuit, AWSDevice.IQM_GARNET)
    assert isinstance(job.circuit, QCircuit)
    job.circuit.transpiled_circuit = Mock()
    braket_device = Mock()
    task = Mock(id="task-id")
    braket_device.run.return_value = task
    entered_contexts = []
    fake_circuits_module = ModuleType("braket.circuits")
    fake_circuits_module.__dict__["Circuit"] = object

    @contextmanager
    def capture_reservation(device: object, reservation_arn: str | None):
        entered_contexts.append((device, reservation_arn))
        yield

    with (
        patch.dict(sys.modules, {"braket.circuits": fake_circuits_module}),
        patch.object(aws_provider, "get_braket_device", return_value=braket_device),
        patch.object(
            aws_provider, "optional_reservation_arn", side_effect=capture_reservation
        ),
    ):
        job_id, submitted_task = submit_job_braket(
            job, BraketParams(reservation_arn="reservation-arn")
        )

    assert entered_contexts == [(braket_device, "reservation-arn")]
    assert job_id == "task-id"
    assert submitted_task is task


@pytest.mark.provider("braket")
def test_binding_forwards_braket_params():
    """Forward the original ``BraketParams`` through the binding runner."""
    params = BraketParams(reservation_arn="reservation-arn")
    binding = CircuitBinding(QCircuit(1))
    expected = BatchResult([])

    with patch("mpqp.execution.runner.run_braket", return_value=expected) as execute:
        result = runner._run_circuit_binding(  # pyright: ignore[reportPrivateUsage]
            binding,
            AWSDevice.IQM_GARNET,
            provider_params=params,
        )

    assert result is expected
    assert execute.call_args.args[1] is params
