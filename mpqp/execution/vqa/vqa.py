"""Sequential VQA execution with reusable parametric circuit templates."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field
from numbers import Number, Real
from typing import Callable, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, OptimizeResult
from scipy.optimize import minimize as scipy_minimize
from sympy import Basic, default_sort_key

from mpqp.core import QCircuit
from mpqp.core.circuitbinding import (
    BindingExecution,
    BindingMode,
    CircuitBinding,
)
from mpqp.core.instruction import ExpectationMeasure, Measure
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.job import ExecutionMode, JobType
from mpqp.execution.providers.providers_params import ProviderParams
from mpqp.execution.result import Result
from mpqp.execution.runner import run
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
CostFunction = Callable[[npt.NDArray[np.float64], Sequence[Result]], float]


def _real_parameter_value(value: Number) -> float:
    """Normalize a binding value while rejecting complex VQA parameters."""
    if not isinstance(value, Real):
        raise ValueError("VQA binding values must be real numbers.")
    return float(value)


@dataclass(frozen=True)
class _PreparedCircuit:
    """Provider-ready circuit and its immutable VQA execution metadata."""

    circuit: QCircuit
    base_values: dict[str, float]
    measurement: Optional[Measure]


@dataclass(frozen=True)
class _ExecutionRequest:
    """One ordered execution generated for a parameter vector."""

    index: int
    circuit: QCircuit
    values: dict[str, float]
    measurement: Optional[Measure]


def _isolated_execution_circuit(circuit: QCircuit) -> QCircuit:
    """Share compiled artifacts without exposing the cached VQA template."""
    isolated = copy(circuit)
    isolated.transpiled_circuit = dict(circuit.transpiled_circuit)
    return isolated


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


@dataclass
class VQAResult:
    """Objective history, final parameters and optimizer output for one stage."""

    loss_total: list[float] = field(default_factory=list)
    """Loss at each objective evaluation, in evaluation order."""
    angles: dict[Basic, float] = field(default_factory=dict)
    """Final parameter values, including classical parameters."""
    loss: float = 0.0
    """Final loss reported by the optimizer."""
    optimizer_results: Optional[OptimizeResult] = None
    """Optimizer output; unavailable until optimization completes."""

    def __str__(self) -> str:
        return f"Loss: {self.loss} \nAngles: {self.angles}"


class VQAModule:
    """Prepare parametric circuits once and execute ZIP bindings.

    Args:
        circuits: Circuits or binding whose measurements define the results.
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
        circuit structure or device. Provider circuits are transpiled once;
        evaluations only create ZIP bindings containing new numeric values.
    """

    def __init__(
        self,
        circuits: QCircuit | Sequence[QCircuit] | CircuitBinding,
        device: AvailableDevice,
        parameters: Optional[Sequence[Basic]] = None,
        cost_function: Optional[CostFunction] = None,
    ) -> None:
        binding_execution: list[BindingExecution] = []
        if isinstance(circuits, CircuitBinding):
            binding_execution = deepcopy(circuits).unroll()
        else:
            source = [circuits] if isinstance(circuits, QCircuit) else list(circuits)
            for circuit in source:
                owned = deepcopy(circuit)
                measurements = owned.measurements
                if len(measurements) > 1:
                    raise ValueError(
                        "VQA circuits must contain at most one measurement."
                    )
                binding_execution.append(
                    (
                        owned.without_measurements(),
                        None,
                        measurements[0] if measurements else None,
                    )
                )
        if len(binding_execution) == 0:
            raise ValueError("At least one circuit is required.")
        owned_circuits: dict[int, QCircuit] = {}
        prepared: list[_PreparedCircuit] = []
        for circuit, values, measurement in binding_execution:
            prepared.append(
                _PreparedCircuit(
                    circuit=owned_circuits.setdefault(id(circuit), deepcopy(circuit)),
                    base_values=(
                        {
                            str(key): _real_parameter_value(value)
                            for key, value in values.items()
                        }
                        if values is not None
                        else {}
                    ),
                    measurement=deepcopy(measurement),
                )
            )
        self._prepared: tuple[_PreparedCircuit, ...] = tuple(prepared)

        symbols: set[Basic] = set().union(
            *(item.circuit.variables() for item in self._prepared)
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

        self.backend = device
        self.cost_function = cost_function
        self.result = VQAResult()
        for item in self._prepared:
            item.circuit.transpiled_for_device(device)

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
        provider_params: Optional[ProviderParams] = None,
    ) -> tuple[Result, ...]:
        """Evaluate one parameter vector through ZIP circuit bindings.

        Args:
            current_params: Values in the module's parameter order.
            shots: Shot override. Its priority is: this argument, the source
                ``CircuitBinding.shots``, then the measurement's own value.
            mode: Provider execution mode. ``None`` selects ``JOB``.
            provider_params: Provider-specific execution configuration.

        Returns:
            Raw measurement results in circuit order.

        Raises:
            ValueError: If parameters, shots or gate expressions are invalid.

        Note:
            The cached templates are never bound or passed to a provider.
        """
        return self.evaluate_batch(
            [current_params],
            shots=shots,
            mode=mode,
            provider_params=provider_params,
        )[0]

    def _binding_values(
        self, prepared: _PreparedCircuit, parameters: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        """Merge fixed values with the current optimizer vector."""
        optimized_names = {str(variable) for variable in self.variables}
        values = {
            key: value
            for key, value in prepared.base_values.items()
            if str(key) not in optimized_names
        }
        values.update(
            {
                str(variable): float(value)
                for variable, value in zip(self.variables, parameters)
            }
        )
        return values

    def _measurement(
        self, prepared: _PreparedCircuit, shots: Optional[int]
    ) -> Optional[Measure]:
        measurement = deepcopy(prepared.measurement)
        if measurement is not None and shots is not None:
            measurement.shots = shots
        # A zero-shot basis measurement requests a state vector and therefore
        # must not be translated as a sampling measurement in CircuitBinding.
        if isinstance(measurement, BasisMeasure) and measurement.shots == 0:
            return None
        return measurement

    def evaluate_batch(
        self,
        parameter_batch: Sequence[OptimizerInput],
        shots: Optional[int] = None,
        mode: Optional[ExecutionMode] = ExecutionMode.BATCH,
        provider_params: Optional[ProviderParams] = None,
    ) -> tuple[tuple[Result, ...], ...]:
        """Evaluate several parameter vectors in one provider batch when possible.

        Executions are ordered first by parameter vector, then by the original
        circuit/binding order. Mixed job types require one ZIP binding per type;
        results are reassembled in their original order.
        """
        if shots is not None and (type(shots) is not int or shots < 0):
            raise ValueError("shots must be a non-negative integer or None.")
        parameter_vectors = [
            self._parameters(parameters) for parameters in parameter_batch
        ]
        if not parameter_vectors:
            raise ValueError("parameter_batch must contain at least one vector.")

        # CircuitBinding requires homogeneous job types. Build one ZIP binding
        # for each type while retaining the global point-major result order.
        grouped: dict[JobType, list[_ExecutionRequest]] = {}
        width = len(self._prepared)
        for point_index, parameters in enumerate(parameter_vectors):
            for circuit_index, prepared in enumerate(self._prepared):
                measurement = self._measurement(prepared, shots)
                if measurement is None:
                    job_type = JobType.STATE_VECTOR
                elif isinstance(measurement, ExpectationMeasure):
                    job_type = JobType.OBSERVABLE
                elif isinstance(measurement, BasisMeasure):
                    job_type = JobType.SAMPLE
                else:
                    raise TypeError(
                        f"Unsupported VQA measurement: {type(measurement).__name__}."
                    )
                grouped.setdefault(job_type, []).append(
                    _ExecutionRequest(
                        index=point_index * width + circuit_index,
                        circuit=_isolated_execution_circuit(prepared.circuit),
                        values=self._binding_values(prepared, parameters),
                        measurement=measurement,
                    )
                )

        ordered: list[Optional[Result]] = [None] * (len(parameter_vectors) * width)
        execution_mode = mode or ExecutionMode.JOB
        for executions in grouped.values():
            indices = [execution.index for execution in executions]
            circuits = [execution.circuit for execution in executions]
            values = [execution.values for execution in executions]
            measurements = [execution.measurement for execution in executions]
            if all(measurement is None for measurement in measurements):
                binding = CircuitBinding(
                    circuits=list(circuits),
                    values=list(values),
                    mode=BindingMode.ZIP,
                )
            elif all(measurement is not None for measurement in measurements):
                typed_measurements = [
                    measurement
                    for measurement in measurements
                    if measurement is not None
                ]
                binding = CircuitBinding(
                    circuits=circuits,
                    values=values,
                    measurements=typed_measurements,
                    mode=BindingMode.ZIP,
                )
            else:
                raise ValueError("A VQA execution group has inconsistent measurements.")
            batch = run(
                binding,
                self.backend,
                mode=execution_mode,
                provider_params=provider_params,
            )
            if len(batch.results) != len(indices):
                raise RuntimeError("Provider returned an unexpected result count.")
            for index, result in zip(indices, batch.results):
                ordered[index] = result

        if any(result is None for result in ordered):
            raise RuntimeError("Provider batch did not return every VQA result.")
        complete = tuple(result for result in ordered if result is not None)
        return tuple(
            tuple(complete[offset : offset + width])
            for offset in range(0, len(complete), width)
        )

    def cost(
        self,
        current_params: OptimizerInput,
        shots: Optional[int] = None,
        mode: Optional[ExecutionMode] = ExecutionMode.JOB,
        provider_params: Optional[ProviderParams] = None,
    ) -> float:
        """Evaluate quantum results and the classical cost at one vector.

        Args:
            current_params: Values in the module's parameter order.
            shots: Shot count override. Defaults to each circuit's configuration.
            mode: Provider execution mode. ``None`` selects ``JOB``.
            provider_params: Provider-specific execution configuration.

        Returns:
            Custom cost, or the sum of expectation values when no cost is set.

        Raises:
            ValueError: If evaluation inputs are invalid or the cost is not finite.

        Note:
            Sampling and statevector results require a custom cost function.
            This method does not append to optimization history.
        """
        values = self._parameters(current_params)
        results = self.evaluate(
            values,
            shots=shots,
            mode=mode,
            provider_params=provider_params,
        )
        return self._cost_from_results(values, results)

    def _cost_from_results(
        self,
        parameters: npt.NDArray[np.float64],
        results: Sequence[Result],
    ) -> float:
        """Compute and validate a cost from already executed quantum results."""
        if self.cost_function is not None:
            loss = self.cost_function(parameters, results)
        else:
            loss = 0.0
            for result in results:
                expectations = result.expectation_values
                loss += (
                    sum(expectations.values())
                    if isinstance(expectations, dict)
                    else expectations
                )
        loss = float(loss)
        if not np.isfinite(loss):
            raise ValueError("Cost function returned a non-finite value.")
        return loss

    def minimize(
        self,
        optimizer_data: OptimizerData,
        eval_func: Optional[OptimizableFunc] = None,
        mode: Optional[ExecutionMode] = ExecutionMode.JOB,
        shots: Optional[int] = None,
        provider_params: Optional[ProviderParams] = None,
    ) -> VQAResult:
        """Optimize without mutating options or initial parameters.

        Args:
            optimizer_data: Configuration for this optimization stage.
            eval_func: Full objective override, bypassing quantum execution.
                Prefer ``cost_function`` for quantum post-processing.
            mode: Provider execution mode. ``None`` selects ``JOB``.
            shots: Shot count override. Defaults to each circuit's configuration.
            provider_params: Provider-specific execution configuration.

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
                else self.cost(
                    values,
                    shots=shots,
                    mode=mode,
                    provider_params=provider_params,
                )
            )
            self.result.loss = value
            self.result.loss_total.append(value)
            return value

        def batch_objective(
            candidates: Sequence[npt.NDArray[np.float64]],
        ) -> list[float]:
            vectors = [self._parameters(candidate) for candidate in candidates]
            batches = self.evaluate_batch(
                vectors,
                shots=shots,
                mode=ExecutionMode.BATCH,
                provider_params=provider_params,
            )
            losses = [
                self._cost_from_results(vector, results)
                for vector, results in zip(vectors, batches)
            ]
            self.result.loss_total.extend(losses)
            if losses:
                self.result.loss = losses[-1]
            return losses

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
                batch_eval=batch_objective if eval_func is None else None,
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


def minimize(
    optimizable: QCircuit | OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    device: Optional[AvailableDevice] = None,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[OptimizerOptions] = None,
    callback: Optional[OptimizerCallback] = None,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Compatibility wrapper around :class:`VQAModule` and ``run_optimizer``.

    New code should instantiate :class:`VQAModule` directly when optimizing a
    quantum circuit. Callable objectives keep the historical ``minimize`` API.
    """
    if isinstance(optimizable, QCircuit):
        if device is None:
            raise ValueError("A device is needed to optimize a circuit.")
        module = VQAModule(optimizable, device)
        if nb_params is not None and nb_params != len(module.variables):
            raise ValueError(
                f"Expected {len(module.variables)} circuit parameters, got {nb_params}."
            )
        result = module.minimize(
            OptimizerData(
                method=method,
                init_params=init_params,
                optimizer_options=optimizer_options,
                callback=callback,
            )
        )
        return result.loss, np.array(
            [result.angles[variable] for variable in module.variables], dtype=float
        )

    if init_params is None:
        if nb_params is None:
            raise ValueError(
                "Provide init_params or nb_params for a callable objective."
            )
        init_params = np.zeros(nb_params, dtype=float)
    initial = np.asarray(init_params, dtype=float)
    options = deepcopy(optimizer_options or {})
    if isinstance(method, Optimizer):
        return run_optimizer(
            optimizable,
            method,
            initial,
            options,
            callback=callback,
        )
    loss, parameters = method(optimizable, initial, options)
    return float(loss), np.asarray(parameters, dtype=float)
