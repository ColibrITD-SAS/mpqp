"""Sequential VQA execution with reusable parametric circuit templates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from numbers import Number
from typing import Callable, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, OptimizeResult, minimize as scipy_minimize
from sympy import Basic, Expr, default_sort_key, lambdify

from mpqp.core import QCircuit
from mpqp.core.instruction import ExpectationMeasure
from mpqp.core.instruction.gates import ParametrizedGate
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.job import ExecutionMode
from mpqp.execution.result import Result
from mpqp.execution.runner import ValuesDict, run
from mpqp.execution.vqa.optimizer import (
    OptimizableFunc,
    Optimizer,
    OptimizerInput,
    OptimizerOptions,
    run_optimizer,
)

OptimizerCallable = Callable[
    [OptimizableFunc, OptimizerInput, OptimizerOptions], tuple[float, OptimizerInput]
]
OptimizerCallback = Callable[[OptimizerInput], None]
OptimizerBounds = Bounds | Sequence[tuple[Optional[float], Optional[float]]]
EvaluationFunc = Callable[[Sequence[Result]], float]
CostFunction = Callable[[npt.NDArray[np.float64], Sequence[Result]], float]


@dataclass
class OptimizerData:
    """Configure one optimization stage.

    Args:
        method: Optimizer selection or callable receiving the objective, initial
            parameters and options, and returning the loss and parameters.
        init_params: Initial vector in the module's parameter order. Defaults
            to zeros when omitted.
        maxiter: Maximum iterations, overriding the corresponding option.
        optimizer_options: Options forwarded to the selected optimizer.
        callback: Function called with the current parameters after an iteration.
            Supported by SciPy and CMA-ES, but not custom optimizers.
        jac: Derivative of the complete objective for SciPy optimizers.
        bounds: SciPy parameter bounds. For CMA-ES, use optimizer options.
    """

    method: Optimizer | OptimizerCallable
    init_params: Optional[OptimizerInput] = None
    maxiter: Optional[int] = None
    optimizer_options: Optional[OptimizerOptions] = None
    callback: Optional[OptimizerCallback] = None
    jac: Optional[Callable[[OptimizerInput], OptimizerInput]] = None
    bounds: Optional[OptimizerBounds] = None


class RunMode(Enum):
    """Measurement categories; select the execution result through the circuit."""

    EXPECTATION = auto()
    SAMPLING = auto()
    STATEVECTOR = auto()


class VQAResult:
    """Objective history, final parameters and optimizer output for one stage."""

    def __init__(self) -> None:
        self.loss_total: list[float] = []
        """Loss at each objective evaluation, in evaluation order."""
        self.angles: dict[Basic, float] = {}
        """Final parameter values, including classical parameters."""
        self.loss: float = 0.0
        """Final loss reported by the optimizer."""
        self.optimizer_results: Optional[OptimizeResult] = None
        """Optimizer output; unavailable until optimization completes."""

    def __str__(self) -> str:
        return f"Loss: {self.loss} \nAngles: {self.angles}"


class VQAModule:
    """Prepare circuits once and execute them sequentially.

    Args:
        circuits: Circuits whose measurements define the requested results.
        device: Device used to prepare and execute the circuits.
        parameters: Vector order, including any classical parameters. Defaults
            to circuit symbols sorted with SymPy's deterministic ordering.
        cost_function: Function receiving the parameter vector and raw results
            in circuit order. Defaults to the sum of all expectation values.

    Raises:
        ValueError: If circuits are empty or parameters contain duplicates,
            omit circuit symbols, are not symbols, or have ambiguous names.

    Note:
        The module owns copies of its circuits. Recreate it when changing the
        circuit structure or device. Binding follows QCircuit's Qiskit/Braket
        support. A measurement variant is prepared on first use when switching
        between zero and positive shots.
    """

    def __init__(
        self,
        circuits: QCircuit | Sequence[QCircuit],
        device: AvailableDevice,
        parameters: Optional[Sequence[Basic]] = None,
        cost_function: Optional[CostFunction] = None,
    ) -> None:
        source = [circuits] if isinstance(circuits, QCircuit) else list(circuits)
        if not source:
            raise ValueError("At least one circuit is required.")
        self.backend = device
        self._circuits = tuple(deepcopy(circ) for circ in source)
        symbols: set[Basic] = set().union(
            *(circ.variables() for circ in self._circuits)
        )
        self.variables: tuple[Basic, ...] = tuple(
            sorted(symbols, key=default_sort_key) if parameters is None else parameters
        )
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("Parameter order contains duplicates.")
        if not symbols.issubset(self.variables):
            raise ValueError("Parameter order must include every circuit symbol.")
        if any(not getattr(var, "is_Symbol", False) for var in self.variables):
            raise ValueError("Parameters must be SymPy symbols.")
        if len({str(var) for var in self.variables}) != len(self.variables):
            raise ValueError("Distinct parameters must have distinct names.")
        self.cost_function = cost_function
        self.result = VQAResult()
        self._bindings: list[tuple[tuple[str, ...], Callable[..., Sequence[float]]]] = (
            []
        )
        self._measurement_templates: dict[int, QCircuit] = {}
        for circ in self._circuits:
            # Providers encode expressions such as 2*theta as named parameters.
            # Compile the original symbolic expressions once, without eval or
            # parsing provider parameter names during optimization.
            expressions = {
                str(expr): expr
                for inst in circ.instructions
                if isinstance(inst, ParametrizedGate)
                for expr in inst.parameters
                if isinstance(expr, Expr)
            }
            self._bindings.append(
                (
                    tuple(expressions),
                    lambdify(self.variables, list(expressions.values()), "numpy"),
                )
            )
            circ.transpiled_for_device(device)

    def _parameters(self, values: OptimizerInput) -> npt.NDArray[np.float64]:
        """Validate and copy a finite real parameter vector."""
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError("Parameters must be real.")
        array = np.array(raw, dtype=float, copy=True)
        if array.shape != (len(self.variables),) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"Expected {len(self.variables)} finite parameters in a vector."
            )
        return array

    def evaluate(
        self,
        current_params: OptimizerInput,
        shots: Optional[int] = None,
        mode: Optional[ExecutionMode] = ExecutionMode.JOB,
    ) -> tuple[Result, ...]:
        """Return raw results without changing templates or optimization history.

        Args:
            current_params: Values in the module's parameter order.
            shots: Shot count override. Defaults to each circuit's configuration.
            mode: Sequential job execution. ``None`` also selects this mode.

        Returns:
            Raw measurement results in circuit order.

        Raises:
            ValueError: If parameters or shots are invalid, the execution mode
                is unsupported, or gate expressions produce non-finite values.
            TypeError: If execution does not return a single result.

        Note:
            Each execution owns its bound circuit, including on provider failure.
        """
        values = self._parameters(current_params)
        if mode not in (None, ExecutionMode.JOB):
            raise ValueError("VQAModule supports sequential JOB execution only.")
        if shots is not None and (type(shots) is not int or shots < 0):
            raise ValueError("shots must be a non-negative integer or None.")
        results: list[Result] = []
        for index, (template, (names, bind)) in enumerate(
            zip(self._circuits, self._bindings)
        ):
            # Switching zero/positive shots changes the presence of measurement
            # instructions. Prepare that structural variant once, before binding.
            if shots is not None and any(
                isinstance(measurement, BasisMeasure)
                and (measurement.shots == 0) != (shots == 0)
                for measurement in template.measurements
            ):
                if index not in self._measurement_templates:
                    variant = deepcopy(template)
                    for measurement in variant.measurements:
                        if isinstance(measurement, BasisMeasure):
                            measurement.shots = shots
                    variant.transpiled_circuit.clear()
                    variant.transpiled_for_device(self.backend)
                    self._measurement_templates[index] = variant
                template = self._measurement_templates[index]
            circuit = deepcopy(template)
            if shots is not None:
                for measurement in circuit.measurements:
                    if isinstance(measurement, (ExpectationMeasure, BasisMeasure)):
                        measurement.shots = shots
            bindings: ValuesDict = {}
            for name, value in zip(names, map(float, bind(*values))):
                if not np.isfinite(value):
                    raise ValueError("Gate expressions produced non-finite parameters.")
                # The runner uses the numeric ABC, which static typing does not
                # recognize as a base class of Python's float.
                assert isinstance(value, Number)
                bindings[name] = value
            result = run(circuit, self.backend, values=bindings or None, mode=mode)
            if not isinstance(result, Result):
                raise TypeError("Sequential execution must return a Result.")
            results.append(result)
        return tuple(results)

    def cost(
        self,
        current_params: OptimizerInput,
        shots: Optional[int] = None,
        mode: Optional[ExecutionMode] = ExecutionMode.JOB,
    ) -> float:
        """Evaluate quantum results and the classical cost at one vector.

        Args:
            current_params: Values in the module's parameter order.
            shots: Shot count override. Defaults to each circuit's configuration.
            mode: Sequential job execution. ``None`` also selects this mode.

        Returns:
            Custom cost, or the sum of expectation values when no cost is set.

        Raises:
            ValueError: If evaluation inputs are invalid or the cost is not finite.

        Note:
            Sampling and statevector results require a custom cost function.
            This method does not append to optimization history.
        """
        values = self._parameters(current_params)
        results = self.evaluate(values, shots=shots, mode=mode)
        if self.cost_function is not None:
            loss = self.cost_function(values, results)
        else:
            loss = 0.0
            for result in results:
                expectations = result.expectation_values
                loss += (
                    sum(expectations.values())
                    if isinstance(expectations, dict)
                    else expectations
                )
        return float(loss)

    def minimize(
        self,
        optimizer_data: OptimizerData,
        eval_func: Optional[OptimizableFunc] = None,
        mode: Optional[ExecutionMode] = ExecutionMode.JOB,
        shots: Optional[int] = None,
    ) -> VQAResult:
        """Optimize without mutating options or initial parameters.

        Args:
            optimizer_data: Configuration for this optimization stage.
            eval_func: Full objective override, bypassing quantum execution.
                Prefer ``cost_function`` for quantum post-processing.
            mode: Sequential job execution. ``None`` also selects this mode.
            shots: Shot count override. Defaults to each circuit's configuration.

        Returns:
            Final loss, parameters, evaluation history and optimizer output.

        Raises:
            ValueError: If parameters or costs are invalid, or the selected
                optimizer does not support the supplied configuration.

        Note:
            The Jacobian differentiates the complete objective. Otherwise SciPy
            uses numerical differentiation. No universal parameter-shift rule
            is assumed for nonlinear costs or shared gate parameters.
            ``loss_total`` records evaluations, not optimizer iterations.
        """
        initial = self._parameters(
            np.zeros(len(self.variables))
            if optimizer_data.init_params is None
            else optimizer_data.init_params
        )
        self.result = VQAResult()

        def objective(params: OptimizerInput) -> float:
            values = self._parameters(params)
            value = float(
                eval_func(values)
                if eval_func is not None
                else self.cost(values, shots=shots, mode=mode)
            )
            self.result.loss = value
            self.result.loss_total.append(value)
            return value

        options = deepcopy(optimizer_data.optimizer_options or {})
        if optimizer_data.maxiter is not None:
            options["maxiter"] = optimizer_data.maxiter
        if optimizer_data.method == Optimizer.CMAES:
            if optimizer_data.jac is not None or optimizer_data.bounds is not None:
                raise ValueError(
                    "CMAES does not accept jac; configure its bounds in optimizer_options."
                )
            loss, params = run_optimizer(
                objective,
                Optimizer.CMAES,
                initial,
                options,
                callback=optimizer_data.callback,
            )
            res = OptimizeResult(fun=loss, x=params)
        elif isinstance(optimizer_data.method, Optimizer):
            res = scipy_minimize(
                objective,
                x0=initial,
                method=optimizer_data.method.value,
                options=options,
                callback=optimizer_data.callback,
                jac=optimizer_data.jac,
                bounds=optimizer_data.bounds,
            )
        else:
            if any(
                value is not None
                for value in (
                    optimizer_data.jac,
                    optimizer_data.bounds,
                    optimizer_data.callback,
                )
            ):
                raise ValueError(
                    "Custom optimizers accept only objective, initial parameters and options."
                )
            loss, params = optimizer_data.method(objective, initial, options)
            res = OptimizeResult(fun=float(loss), x=self._parameters(params))
        final_params = self._parameters(res.x)
        if not np.isfinite(res.fun):
            raise ValueError("Optimizer returned a non-finite cost.")
        self.result.loss = float(res.fun)
        self.result.angles = dict(zip(self.variables, map(float, final_params)))
        self.result.optimizer_results = res
        return self.result
