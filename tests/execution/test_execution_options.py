"""Execution options shared by ordinary runs and VQA clients."""

from types import SimpleNamespace
from typing import Any

import pytest
from sympy import symbols

from mpqp import (
    AWSDevice,
    BindingMode,
    CircuitBinding,
    ExpectationMeasure,
    IBMDevice,
    Observable,
    QCircuit,
    pZ,
)
from mpqp.execution.providers.providers_params import AWSParams, QiskitParams
from mpqp.execution.result import Result
from mpqp.gates import Ry

DEVICE = IBMDevice.AER_SIMULATOR
theta = symbols("theta")


def circuit():
    return QCircuit([Ry(theta, 0), ExpectationMeasure(Observable(pZ))])


def test_runner_forwards_execution_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import mpqp.execution.runner as runner
    from mpqp.execution.job import ExecutionMode

    seen = {}
    sentinel = object()

    def fake_run(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(runner, "_run_single", fake_run)
    provider_options = QiskitParams()
    assert (
        runner.run(
            circuit(),
            DEVICE,
            mode=ExecutionMode.JOB,
            provider_params=provider_options,
        )
        is sentinel
    )
    assert seen == {
        "mode": ExecutionMode.JOB,
        "provider_params": provider_options,
    }


def test_aws_reservation_is_forwarded_through_provider_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mpqp.execution.runner as runner

    expected = object.__new__(Result)
    seen: dict[str, Any] = {}

    def fake_run_braket(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(runner, "run_braket", fake_run_braket)
    monkeypatch.setattr(
        runner, "generate_job", lambda *args, **kwargs: SimpleNamespace()
    )
    params = AWSParams(reservation_arn="reservation")
    result = runner._run_single(
        circuit(),
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
        provider_params=params,
    )
    assert result is expected
    assert seen == {"provider_params": params}


def test_public_binding_api() -> None:
    import mpqp

    assert mpqp.CircuitBinding is CircuitBinding
    assert mpqp.BindingMode is BindingMode
