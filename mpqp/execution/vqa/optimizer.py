"""For now, our minimizer is a wrapper around ``scipy``'s minimizer. The
:class:`Optimizer` enum lists all the methods validated with the rest of the
library."""

from enum import Enum
from copy import deepcopy
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

OptimizerInput = Union[list[float], npt.NDArray[np.float64]]
OptimizableFunc = Callable[[OptimizerInput], float]
OptimizerOptions = dict[str, Any]


class Optimizer(Enum):
    """Enum used to select the optimizer for the VQA."""

    BFGS = "BFGS"
    L_BFGS_B = "L-BFGS-B"
    COBYLA = "COBYLA"
    POWELL = "POWELL"
    NELDER_MEAD = "Nelder-Mead"
    SLSQP = "SLSQP"

    CMAES = "CMAES"


def run_optimizer(
    eval_func: OptimizableFunc,
    method: Optimizer,
    init_params: OptimizerInput,
    optimizer_options: Optional[OptimizerOptions] = None,
    callback: Optional[Callable[[OptimizerInput], None]] = None,
    batch_eval: Optional[
        Callable[[Sequence[npt.NDArray[np.float64]]], Sequence[float]]
    ] = None,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Minimize an objective using SciPy or the optional CMA-ES implementation.

    Args:
        eval_func: Objective evaluated at a parameter vector.
        method: Optimizer to run.
        init_params: Initial parameter vector.
        optimizer_options: Optimizer-specific options, copied before use.
            CMA-ES accepts ``sigma0`` for its initial search scale.
        callback: Function receiving the current parameters after each iteration.
        batch_eval: Optional population evaluator for CMA-ES. Results must
            follow the order of candidate vectors. Ignored by SciPy optimizers.

    Returns:
        Best loss reported by the optimizer and its corresponding parameters.

    Raises:
        ImportError: If CMA-ES is selected but the ``cma`` package is unavailable.

    Note:
        This function does not modify the supplied optimizer options.
    """

    optimizer_options = deepcopy(optimizer_options or {})

    x0 = np.asarray(init_params, dtype=float)

    if method == Optimizer.CMAES:
        import cma

        sigma0 = float(optimizer_options.pop("sigma0", 0.5))
        es = cma.CMAEvolutionStrategy([float(x) for x in x0], sigma0, optimizer_options)

        while not es.stop():
            solutions = es.ask()

            if batch_eval is not None:
                candidates = [np.asarray(x, dtype=float) for x in solutions]
                fit = [float(v) for v in batch_eval(candidates)]
            else:
                fit = [float(eval_func(np.asarray(x, dtype=float))) for x in solutions]

            es.tell(solutions, fit)
            es.disp()

            if callback is not None:
                callback(np.asarray(es.best.x, dtype=float))

        return float(es.result.fbest), np.asarray(es.result.xbest, dtype=float)

    result: OptimizeResult = scipy_minimize(
        eval_func,
        x0=x0,
        method=method.value,
        options=optimizer_options,
        callback=callback,
    )
    best_value = float(result.fun)
    best_params = np.asarray(result.x, dtype=float)

    return best_value, best_params
