"""For now, our minimizer is a wrapper around ``scipy``'s minimizer. The
:class:`Optimizer` enum lists all the methods validated with the rest of the
library."""

from enum import Enum
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

OptimizerInput = Union[list[float], npt.NDArray[np.float_]]
OptimizableFunc = Union[partial[float], Callable[[OptimizerInput], float]]
OptimizerOptions = dict[str, Any]
# OptimizerCallback = Callable[[OptimizerInput, float], None]


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
) -> tuple[float, npt.NDArray[np.float_]]:

    if optimizer_options is None:
        optimizer_options = {}

    x0 = np.asarray(init_params, dtype=float)

    if method == Optimizer.CMAES:
        import cma

        sigma0 = float(optimizer_options.pop("sigma0", 0.5))

        best_params, es = cma.fmin2(
            eval_func,
            x0=x0,
            sigma0=sigma0,
            options=optimizer_options,
        )
        best_value = float(es.result.fbest)

        return best_value, np.asarray(best_params, dtype=float)

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
