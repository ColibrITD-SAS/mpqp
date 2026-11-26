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
    result = runner.run(
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


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(IBMDevice.AER_SIMULATOR, marks=pytest.mark.provider("qiskit")),
        pytest.param(
            AWSDevice.BRAKET_LOCAL_SIMULATOR, marks=pytest.mark.provider("braket")
        ),
    ],
)
@pytest.mark.parametrize("cached", [False, True])
def test_batch_parameter_sweep_preserves_template(
    device: IBMDevice | AWSDevice, cached: bool
):
    from copy import deepcopy
    from math import pi

    from mpqp import BatchResult, run
    from mpqp.execution.job import ExecutionMode

    template = circuit()
    original = deepcopy(template)
    if cached:
        template.transpiled_for_device(device)

    batch = run(
        template,
        device,
        values=[{theta: 0.0}, {theta: pi}],
        mode=ExecutionMode.BATCH,
    )

    assert isinstance(batch, BatchResult)
    assert [result.expectation_values for result in batch] == pytest.approx([1.0, -1.0])
    assert all(result.job.mode == ExecutionMode.BATCH for result in batch)
    assert template == original
    assert template.label is None
    assert run(
        template, device, values={theta: 0.0}
    ).expectation_values == pytest.approx(1.0)
