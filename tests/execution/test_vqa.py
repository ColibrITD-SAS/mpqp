from typing import Any, Sequence

import numpy.typing as npt
import numpy as np
import pytest
from sympy import Expr, symbols

from mpqp import (
    BasisMeasure,
    ExpectationMeasure,
    IBMDevice,
    Observable,
    QCircuit,
    pX,
    pZ,
)
from mpqp.gates import Ry, Rz
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.result import Result
from mpqp.execution.providers.providers_params import QiskitParams
from mpqp.execution.vqa.optimizer import OptimizableFunc, OptimizerInput, OptimizerOptions
from mpqp.execution.vqa.vqa import Optimizer, OptimizerData, VQAModule

pytestmark = pytest.mark.provider("qiskit")
theta, phi, scale = symbols("theta phi scale")
DEVICE = IBMDevice.AER_SIMULATOR


def circuit(angle: Expr = theta, measurement: BasisMeasure | ExpectationMeasure | None = None) -> QCircuit:
    return QCircuit([Ry(angle, 0), measurement or ExpectationMeasure(Observable(pZ))])


def test_compile_once_and_bind_expressions(monkeypatch: pytest.MonkeyPatch) -> None:
    original = QCircuit.to_other_device
    calls = []

    def counted(self: QCircuit, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(QCircuit, "to_other_device", counted)
    source = circuit(2 * theta + phi)
    vqa = VQAModule(source, DEVICE, parameters=[theta, phi])
    for params in ([0.2, 0.1], [0.7, -0.2], [0.2, 0.1]):
        assert vqa.cost(params) == pytest.approx(np.cos(2 * params[0] + params[1]))
    assert len(calls) == 1
    assert source.transpiled_circuit == {}
    assert source.variables() == {theta, phi}


def test_shared_parameters_and_observable_dictionary() -> None:
    measure = ExpectationMeasure([Observable(pZ), Observable(pX)])
    vqa = VQAModule([circuit(theta, measure), circuit(theta)], DEVICE)
    assert vqa.variables == (theta,)
    assert vqa.cost([0.3]) == pytest.approx(2 * np.cos(0.3) + np.sin(0.3))


def test_custom_cost_with_classical_parameter() -> None:
    def loss(params: npt.NDArray[np.float64], results: Sequence[Result]) -> float:
        expectation = results[0].expectation_values
        assert not isinstance(expectation, dict)
        return float((params[1] * expectation - 0.5) ** 2 + params[1] ** 2)

    vqa = VQAModule(circuit(), DEVICE, parameters=[theta, scale], cost_function=loss)
    assert vqa.cost([0.2, 2]) == pytest.approx((2 * np.cos(0.2) - 0.5) ** 2 + 4)


def test_statevector_and_sampling() -> None:
    state = VQAModule(
        QCircuit([Ry(theta, 0)]),
        DEVICE,
        cost_function=lambda p, results: results[0].probabilities[1],
    )
    assert state.cost([0.4]) == pytest.approx(np.sin(0.2) ** 2)
    assert state.cost([0.8]) == pytest.approx(np.sin(0.4) ** 2)
    source = circuit(measurement=BasisMeasure(shots=32))
    sampled = VQAModule(
        source, DEVICE, cost_function=lambda p, results: sum(results[0].counts)
    )
    assert sampled.cost([0], shots=16) == 16
    assert sampled.cost([0]) == 32
    assert source.measurements[0].shots == 32


def test_failure_does_not_poison_template(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    module = importlib.import_module("mpqp.execution.vqa.vqa")
    vqa = VQAModule(circuit(), DEVICE)
    original = module.run

    def fail(circ: QCircuit, device: AvailableDevice, **kwargs: Any) -> Result:
        circ.bind_parameters(device, kwargs["values"])
        raise RuntimeError("provider failure")

    monkeypatch.setattr(module, "run", fail)
    with pytest.raises(RuntimeError, match="provider failure"):
        vqa.evaluate([0.9])
    monkeypatch.setattr(module, "run", original)
    assert vqa.cost([0.2]) == pytest.approx(np.cos(0.2))


def test_optimizer_converges_and_preserves_configuration() -> None:
    options = {"gtol": 1e-7}
    callback_values = []
    config = OptimizerData(
        Optimizer.BFGS,
        [0.4],
        maxiter=30,
        optimizer_options=options,
        callback=lambda x: callback_values.append(x.copy()),
        jac=lambda x: np.array([-np.sin(x[0])]),
    )
    vqa = VQAModule(circuit(), DEVICE)
    result = vqa.minimize(config)
    assert result.loss == pytest.approx(-1, abs=1e-7)
    assert result.optimizer_results is not None
    assert result.optimizer_results.success
    assert result.loss_total
    assert callback_values
    assert options == {"gtol": 1e-7}
    assert config.init_params == [0.4]
    assert result.angles[theta] == pytest.approx(result.optimizer_results.x[0])


@pytest.mark.parametrize("values", [[1, 2], [[1]], [float("nan")], [1j]])
def test_invalid_parameter_vectors(values: Any) -> None:
    with pytest.raises(ValueError):
        VQAModule(circuit(), DEVICE).evaluate(values)


def test_parameter_order_validation() -> None:
    source = QCircuit([Ry(theta, 0), Rz(phi, 0)])
    assert VQAModule(source, DEVICE).variables == (phi, theta)
    for order in ([theta], [theta, theta, phi]):
        with pytest.raises(ValueError):
            VQAModule(source, DEVICE, parameters=order)


def test_custom_optimizer_and_objective_override() -> None:
    def optimizer(fun: OptimizableFunc, initial: OptimizerInput, options: OptimizerOptions) -> tuple[float, OptimizerInput]:
        options["changed"] = True
        return fun([0.5]), [0.5]

    config = OptimizerData(optimizer, [0.1], optimizer_options={})
    vqa = VQAModule(circuit(), DEVICE)
    result = vqa.minimize(config, eval_func=lambda x: x[0] ** 2)
    assert result.loss == 0.25
    assert result.angles == {theta: 0.5}
    assert config.optimizer_options == {}


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
            reservation_arn="reservation",
            provider_params=provider_options,
        )
        is sentinel
    )
    assert seen == {
        "mode": ExecutionMode.JOB,
        "reservation_arn": "reservation",
        "provider_params": provider_options,
    }


@pytest.mark.parametrize("initial_shots", [0, 32])
def test_sampling_statevector_switch_prepares_each_variant_once(
    monkeypatch: pytest.MonkeyPatch, initial_shots: int
) -> None:
    original = QCircuit.to_other_device
    calls = []

    def counted(self: QCircuit, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(QCircuit, "to_other_device", counted)
    source = circuit(measurement=BasisMeasure(shots=initial_shots))
    vqa = VQAModule(source, DEVICE)
    for angle in (0.4, 0.8):
        state = vqa.evaluate([angle], shots=0)[0]
        assert state.probabilities[1] == pytest.approx(np.sin(angle / 2) ** 2)
        sampled = vqa.evaluate([0.0], shots=16)[0]
        assert sum(sampled.counts) == 16
        assert sampled.counts[0] == 16
    assert len(calls) == 2
    assert source.measurements[0].shots == initial_shots


def test_cmaes_returns_best_candidate_and_preserves_options() -> None:
    pytest.importorskip("cma")
    options = {"seed": 42, "verbose": -9, "sigma0": 0.3, "tolfun": 1e-10}
    before = options.copy()
    callbacks = []
    vqa = VQAModule(
        circuit(),
        DEVICE,
        parameters=[theta, scale],
    )
    result = vqa.minimize(
        OptimizerData(
            Optimizer.CMAES,
            [0.4, 0.6],
            maxiter=100,
            optimizer_options=options,
            callback=lambda x: callbacks.append(x.copy()),
        ),
        eval_func=lambda x: float(np.dot(x, x)),
    )
    assert result.loss < 1e-7
    assert result.loss == pytest.approx(min(result.loss_total))
    assert sum(value**2 for value in result.angles.values()) == pytest.approx(
        result.loss
    )
    assert callbacks
    assert options == before
