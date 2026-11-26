"""Offline checks for IBM Runtime session ownership and reuse."""

from unittest.mock import Mock

import pytest
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.exceptions import IBMRuntimeError

from mpqp.execution.connection import ibm_connection
from mpqp.tools.errors import IBMRemoteExecutionError

pytestmark = pytest.mark.provider("qiskit")


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> Mock:
    def create_session(**_kwargs: object) -> Mock:
        return Mock(status=Mock(return_value="Pending"))

    factory = Mock(side_effect=create_session)
    monkeypatch.setattr("qiskit_ibm_runtime.Session", factory)
    monkeypatch.setattr(ibm_connection, "_ibm_sessions", {})
    return factory


def backend(
    monkeypatch: pytest.MonkeyPatch, service: object, instance: str
) -> AerSimulator:
    simulator = AerSimulator()
    monkeypatch.setattr(simulator, "service", service, raising=False)
    monkeypatch.setattr(simulator, "_instance", instance, raising=False)
    return simulator


@pytest.mark.parametrize("different_context", ["service", "instance"])
def test_session_reuse_and_close_respect_context(
    monkeypatch: pytest.MonkeyPatch, sessions: Mock, different_context: str
):
    service = object()
    first_backend = backend(monkeypatch, service, "instance-a")
    same_context = backend(monkeypatch, service, "instance-a")
    other_backend = backend(
        monkeypatch,
        object() if different_context == "service" else service,
        "instance-b" if different_context == "instance" else "instance-a",
    )
    first = ibm_connection.get_or_create_ibm_session(first_backend)
    other = ibm_connection.get_or_create_ibm_session(other_backend)
    assert other is not first
    assert ibm_connection.get_or_create_ibm_session(same_context) is first
    assert sessions.call_count == 2

    ibm_connection.close_ibm_session(other_backend)
    assert ibm_connection.get_or_create_ibm_session(first_backend) is first
    assert ibm_connection.get_or_create_ibm_session(other_backend) is not other
    assert sessions.call_count == 3


@pytest.mark.parametrize(
    "status", ["Closed", None, "In progress, not accepting new jobs"]
)
def test_session_that_cannot_accept_jobs_is_replaced(
    monkeypatch: pytest.MonkeyPatch, sessions: Mock, status: str | None
):
    target = backend(monkeypatch, object(), "instance-a")
    first = ibm_connection.get_or_create_ibm_session(target)
    monkeypatch.setattr(first, "status", lambda: status)
    assert ibm_connection.get_or_create_ibm_session(target) is not first
    assert sessions.call_count == 2


def test_failed_close_keeps_session_for_retry(
    monkeypatch: pytest.MonkeyPatch, sessions: Mock
):
    target = backend(monkeypatch, object(), "instance-a")
    first = ibm_connection.get_or_create_ibm_session(target)
    monkeypatch.setattr(first, "close", Mock(side_effect=IBMRuntimeError("failure")))
    with pytest.raises(IBMRemoteExecutionError, match="you can retry"):
        ibm_connection.close_ibm_session(target)
    assert ibm_connection.get_or_create_ibm_session(target) is first
    assert sessions.call_count == 1
