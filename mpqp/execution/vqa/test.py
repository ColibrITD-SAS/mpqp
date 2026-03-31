from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, TypeVar, Union
from warnings import warn

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from mpqp import QCircuit
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.job import ExecutionMode
from mpqp.execution.result import Result
from mpqp.execution.runner import run
from mpqp.execution.vqa.optimizer import Optimizer
from mpqp.tools import OneOrMany

T1 = TypeVar("T1")
T2 = TypeVar("T2")
OptimizerInput = Union[list[float], npt.NDArray[np.float64]]
OptimizableFunc = Union[partial[float], Callable[[OptimizerInput], float]]
OptimizerOptions = dict[str, Any]
OptimizerCallable = Callable[
    [OptimizableFunc, Optional[OptimizerInput], Optional[OptimizerOptions]],
    tuple[float, OptimizerInput],
]
OptimizerCallback = Union[
    Callable[[OptimizeResult], None],
    Callable[[Union[list[float], npt.NDArray[np.float64], tuple[float, ...]]], None],
]
EvaluationFunc = Callable[[Sequence[Result]], float]


def minimize(
    optimizable: OneOrMany[QCircuit] | OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    device: Optional[AvailableDevice] = None,
    cost_function: Optional[EvaluationFunc] = None,
    mode: Optional[ExecutionMode] = ExecutionMode.JOB,
):
    # ) -> tuple[float, OptimizerInput]:
    _circuit_evaluation: Optional[OptimizableFunc] = None

    if cost_function is None and isinstance(optimizable, Sequence):
        raise ValueError(
            "In order to optimize over several circuits, a `cost_function` "
            "is necessary to turn the results into a single value."
        )
    if isinstance(optimizable, (QCircuit, Sequence)):
        if device is None:
            raise ValueError("A device is needed to optimize circuits.")
        if cost_function is None:

            def _cost_func(results: Sequence[Result]) -> float:
                r = results[0].expectation_values
                if TYPE_CHECKING:
                    assert isinstance(r, float)
                return r

            cost_function = _cost_func
        if not device.is_remote():
            if mode is not ExecutionMode.JOB:
                warn(
                    f"Optimisation on a local simulator ({device} in this case)"
                    f" with an execution mode different from {ExecutionMode.JOB}"
                    f" ({mode} in this case) doesn't make sense. The mode will "
                    "be ignored."
                )
                mode = ExecutionMode.JOB

        def _circ_eval(params: OptimizerInput) -> float:
            results = run(optimizable, device)
            results = [results] if isinstance(results, Result) else [r for r in results]
            return cost_function(results)

        _circuit_evaluation = _circ_eval

    if isinstance(optimizable, (QCircuit, Sequence)):
        assert _circuit_evaluation is not None
        func_to_optimize = _circuit_evaluation
    else:
        func_to_optimize = optimizable

    if mode is ExecutionMode.BATCH:
        raise NotImplementedError()

    # TODO: bellow, it would be better to do it with a context manager, so that
    # in case of crash, the session still closes
    if mode is ExecutionMode.SESSION:
        ...  # TODO: open session
    res = _minimize_job(func_to_optimize, method)
    if mode is ExecutionMode.SESSION:
        ...  # TODO: close session
    return res


def _minimize_job(
    optimizable: OptimizableFunc,
    method: Optimizer | OptimizerCallable,
) -> tuple[float, OptimizerInput]: ...
