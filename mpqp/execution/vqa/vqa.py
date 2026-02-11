from __future__ import annotations

from functools import partial
from typing import Any, Callable, Collection, Optional, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction import ExpectationMeasure
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.job import ExecutionMode
from mpqp.execution.runner import run
from mpqp.execution.vqa.optimizer import Optimizer, run_optimizer

T1 = TypeVar("T1")
T2 = TypeVar("T2")


OptimizerInput = Union[list[float], npt.NDArray[np.float_]]
OptimizableFunc = Union[partial[float], Callable[[OptimizerInput], float]]
OptimizerOptions = dict[str, Any]
OptimizerCallable = Callable[
    [OptimizableFunc, Optional[OptimizerInput], Optional[OptimizerOptions]],
    tuple[float, OptimizerInput],
]
OptimizerCallback = Union[
    Callable[[OptimizeResult], None],
    Callable[[Union[list[float], npt.NDArray[np.float_], tuple[float, ...]]], None],
]


# TODO: all those functions with almost or exactly the same signature look like
#  a code smell to me.

# TODO: test the minimizer options


def _maps(l1: Collection[T1], l2: Collection[T2]) -> dict[T1, T2]:
    """Does like zip, but with a dictionary instead of a list of tuples"""
    if len(l1) != len(l2):
        raise ValueError(
            f"Length of the two collections are not equal ({len(l1)} and {len(l2)})."
        )
    return {e1: e2 for e1, e2 in zip(l1, l2)}


def minimize(
    optimizable: QCircuit | OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    device: Optional[AvailableDevice] = None,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[dict[str, Any]] = None,
    callback: Optional[OptimizerCallback] = None,
    mode: ExecutionMode = ExecutionMode.JOB,
) -> tuple[float, OptimizerInput]:
    """This function runs an optimization on the parameters of the circuit, in order to
    minimize the measured expectation value of observables associated with the given circuit.
    Note that this means that the latter should contain an ``ExpectationMeasure``.

    Args:
        optimizable: Either the circuit, containing symbols and an expectation
            measure, or the evaluation function.
        method: The method used to optimize most of those methods come from
            ``scipy``. If the choices offered in this package are not
            covering your needs, you can define your own optimizer. This should be
            a function taking as input a function representing the circuit, with
            as many inputs as the circuit has parameters, and any optional
            initialization parameters, and returning the optimal value reached
            and the parameters used to reach this value.
        device: The device on which the circuit should be run.
        init_params: The optional initialization parameters (the value
            attributed to the symbols in the first loop of the optimizer).
        nb_params: Number of variables to input in ``optimizable``. It is only
            useful if ``optimizable`` is a Callable and if ``init_params`` was
            not given. If not this argument is not taken into account.
        optimizer_options: Options used to configure the VQA optimizer (maximum
            iterations, convergence threshold, etc...). These options are passed
            as is to the minimizer.
        callback:  A callable called after each iteration.

    Returns:
        The optimal value reached and the parameters corresponding to this value.

    Examples:
        >>> alpha, beta = symbols("α β")
        >>> circuit = QCircuit([
        ...     H(0),
        ...     Rx(alpha, 1),
        ...     CNOT(1,0),
        ...     Rz(beta, 0),
        ...     ExpectationMeasure(
        ...         Observable(np.diag([1,2,-3,4])),
        ...         [0,1],
        ...         shots=0,
        ...     ),
        ... ])
        >>> minimize(
        ...     circuit,
        ...     Optimizer.BFGS,
        ...     ATOSDevice.MYQLM_PYLINALG,
        ...     optimizer_options={"maxiter":50},
        ... )
        (-0.9999999999999996, array([0., 0.]))


        >>> def cost_func(params):
        ...     run_res = run(
        ...         circuit,
        ...         ATOSDevice.MYQLM_PYLINALG,
        ...         {alpha: params[0], beta: params[1]}
        ...     )
        ...     return 1 - run_res.expectation_values ** 2
        >>> minimize(
        ...     cost_func,
        ...     Optimizer.BFGS,
        ...     nb_params=2,
        ...     optimizer_options={"maxiter":50},
        ... )
        (8.881784197001252e-16, array([0., 0.]))

    """
    if isinstance(optimizable, QCircuit):
        if device is None:
            raise ValueError("A device is needed to optimize a circuit")

        if device.is_remote():
            return _minimize_remote(
                optimizable,
                method,
                device,
                init_params,
                nb_params,
                optimizer_options,
                callback,
                mode,
            )

        return _minimize_local(
            optimizable,
            method,
            device,
            init_params,
            nb_params,
            optimizer_options,
            callback,
            mode,
        )

    if device is not None and device.is_remote():
        raise ValueError(
            "Remote execution is only supported when `optimizable` is a QCircuit."
        )

    return _minimize_local(
        optimizable,
        method,
        device,
        init_params,
        nb_params,
        optimizer_options,
        callback,
        mode,
    )


def _minimize_remote(
    optimizable: QCircuit | OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    device: Optional[AvailableDevice] = None,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[dict[str, Any]] = None,
    callback: Optional[OptimizerCallback] = None,
    mode: Optional[ExecutionMode] = None,
) -> tuple[float, OptimizerInput]:
    """This function runs an optimization on the parameters of the circuit, to
    minimize the expectation value of the measure of the circuit by it's
    observables. Note that this means that the circuit should contain an
    expectation measure.

    Args:
        optimizable: Either the circuit, containing symbols and an expectation
            measure, or the evaluation function.
        method: The method used to optimize most of those methods come from
            either scipy or cma. If the choice offered in this package are not
            covering your needs, you can define your own optimizer. It should be
            a function taking as input a function representing the circuit, with
            as many inputs as the circuit has parameters, as well as optional
            initialization parameters, and returning the optimal value reached
            and the parameters used to reach this value.
        device: The device on which the circuit should be run.
        init_params: The optional initialization parameters (the value
            attributed to the symbols in the first loop of the optimizer).
        nb_params: number of variables to input in ``optimizable``. It is only
            useful if ``optimizable`` is a Callable and if ``init_params`` was
            not given. If not this argument is not taken into account.
        optimizer_options: Options used to configure the VQA optimizer (maximum
            iterations, convergence threshold, etc...). These options are passed
            as is to the minimizer.
        callback:  A callable called after each iteration.

    Returns:
        The optimal value reached and the parameters used to reach this value.

    TODO to implement on QLM first
    """
    return _minimize_local(
        optimizable,
        method,
        device,
        init_params,
        nb_params,
        optimizer_options,
        callback,
        mode,
    )


def _minimize_local(
    optimizable: QCircuit | OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    device: Optional[AvailableDevice] = None,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[dict[str, Any]] = None,
    callback: Optional[OptimizerCallback] = None,
    mode: Optional[ExecutionMode] = None,
) -> tuple[float, OptimizerInput]:
    """This function runs an optimization on the parameters of the circuit, to
    minimize the expectation value of the measure of the circuit by it's
    observables. Note that this means that the circuit should contain an
    expectation measure.

    Args:
        optimizable: Either the circuit, containing symbols and an expectation
            measure, or the evaluation function.
        method: The method used to optimize most of those methods come from
            either scipy or cma. If the choice offered in this package are not
            covering your needs, you can define your own optimizer. It should be
            a function taking as input a function representing the circuit, with
            as many inputs as the circuit has parameters, as well as optional
            initialization parameters, and returning the optimal value reached
            and the parameters used to reach this value.
        device: The device on which the circuit should be run.
        init_params: The optional initialization parameters (the value
            attributed to the symbols in the first loop of the optimizer).
        nb_params: number of variables to input in ``optimizable``. It is only
            useful if ``optimizable`` is a Callable and if ``init_params`` was
            not given. If not this argument is not taken into account.
        optimizer_options: Options used to configure the VQA optimizer (maximum
            iterations, convergence threshold, etc...). These options are passed
            as is to the minimizer.
        callback:  A callable called after each iteration.

    Returns:
        the optimal value reached and the parameters used to reach this value.
    """
    if isinstance(optimizable, QCircuit):
        if device is None:
            raise ValueError("A device is needed to optimize a circuit")
        return _minimize_local_circ(
            optimizable,
            device,
            method,
            init_params,
            optimizer_options,
            callback,
            mode,
        )
    else:
        return _minimize_local_func(
            optimizable, method, init_params, nb_params, optimizer_options, callback
        )


def _minimize_local_circ(
    circ: QCircuit,
    device: AvailableDevice,
    method: Optimizer | OptimizerCallable,
    init_params: Optional[OptimizerInput] = None,
    optimizer_options: Optional[dict[str, Any]] = None,
    callback: Optional[OptimizerCallback] = None,
    mode: Optional[ExecutionMode] = None,
) -> tuple[float, OptimizerInput]:
    """This function runs an optimization on the parameters of the circuit, to
    minimize the expectation value of the measure of the circuit by its
    observable. This is equivalent to the run of a VQE. Note that this means
    that the circuit should contain an expectation measure, with only one measurement!

    Args:
        circ: Either the circuit, containing symbols and an expectation measure.
        method: The method used to optimize most of those methods come from
            either scipy or cma. If the choice offered in this package are not
            covering your needs, you can define your own optimizer. It should be
            a function taking as input a function representing the circuit, with
            as many inputs as the circuit has parameters, as well as optional
            initialization parameters, and returning the optimal value reached
            and the parameters used to reach this value.
        device: The device on which the circuit should be run.
        init_params: The optional initialization parameters (the value
            attributed to the symbols in the first loop of the optimizer).
        optimizer_options: Options used to configure the VQA optimizer (maximum
            iterations, convergence threshold, etc...). These options are passed
            as is to the minimizer.
        callback:  A callable called after each iteration.

    Returns:
        The optimal value reached and the parameters used to reach this value.
    """
    if len(circ.measurements) != 1:
        raise ValueError("Cannot optimize a circuit containing several measurements.")

    if not isinstance(circ.measurements[0], ExpectationMeasure):
        raise ValueError("Expected an ExpectationMeasure to optimize the circuit.")

    if len(circ.measurements[0].observables) > 1:
        raise ValueError(
            "Expected only one observable in the ExpectationMeasure but got"
            f" {len(circ.measurements[0].observables)}"
        )

    variables = sorted(circ.variables(), key=str)

    exec_mode = mode or ExecutionMode.JOB
    if exec_mode == ExecutionMode.BATCH and not (
        isinstance(method, Optimizer) and method == Optimizer.CMAES
    ):
        raise ValueError("Batch mode is supported with CMAES optimizer ")

    single_mode = (
        ExecutionMode.SESSION
        if exec_mode == ExecutionMode.SESSION
        else ExecutionMode.JOB
    )

    def eval_circ(params: OptimizerInput) -> float:
        params_fixed = [complex(x) for x in params]
        values = _maps(variables, params_fixed)
        res = run(circ, device, values, mode=single_mode)
        return float(res.expectation_values)

    batch_eval_fn: Optional[
        Callable[[Sequence[npt.NDArray[np.float_]]], Sequence[float]]
    ] = None

    if (
        isinstance(method, Optimizer)
        and method == Optimizer.CMAES
        and exec_mode == ExecutionMode.BATCH
    ):

        def _batch_eval(
            candidates: Sequence[npt.NDArray[np.float_]],
        ) -> Sequence[float]:
            values_list = [
                _maps(variables, [complex(float(x)) for x in cand])
                for cand in candidates
            ]
            batch_res = run(
                circ,
                device,
                values=values_list,
                mode=ExecutionMode.BATCH,
                display_breakpoints=False,
            )

            return [float(res.expectation_values) for res in batch_res]

        batch_eval_fn = _batch_eval

    return _minimize_local_func(
        eval_circ,
        method,
        init_params,
        len(variables),
        optimizer_options,
        callback,
        batch_eval=batch_eval_fn,
    )


def _minimize_local_func(
    eval_func: OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[OptimizerOptions] = None,
    callback: Optional[OptimizerCallback] = None,
    batch_eval: Optional[
        Callable[[Sequence[npt.NDArray[np.float_]]], Sequence[float]]
    ] = None,
) -> tuple[float, OptimizerInput]:
    """This function runs an optimization on the parameters of the circuit, to
    minimize the expectation value of the measure of the circuit by it's
    observables. Note that this means that the circuit should contain an
    expectation measure!

    Args:
        eval_func: Evaluation function.
        method: The method used to optimize most of those methods come from
            either scipy or cma. If the choice offered in this package are not
            covering your needs, you can define your own optimizer. It should be
            a function taking as input a function representing the circuit, with
            as many inputs as the circuit has parameters, as well as optional
            initialization parameters, and returning the optimal value reached
            and the parameters used to reach this value.
        init_params: The optional initialization parameters (the value
            attributed to the symbols in the first loop of the optimizer).
        nb_params: number of variables to input in ``optimizable``. It is only
            useful if ``init_params`` was not given. If not this argument is not
            taken into account.
        optimizer_options: Options used to configure the VQA optimizer (maximum
            iterations, convergence threshold, etc...). These options are passed
            as is to the minimizer.
        callback:  A callable called after each iteration.


    Returns:
        The optimal value reached and the parameters used to reach this value.
    """
    if init_params is None:
        if nb_params is None:
            raise ValueError(
                "Please provide either a set of initialization parameters or "
                "the number of parameters expected by the function."
            )
        else:
            init_params = [0.0] * nb_params

    def _optimizer_callback(x: OptimizerInput) -> None:
        if callback is None:
            return

        if isinstance(x, OptimizeResult):
            callback(x)
            return

        try:
            callback(OptimizeResult(x=np.array(x, dtype=float)))
        except Exception:
            callback(x)

    if isinstance(method, Optimizer):
        best_value, best_params = run_optimizer(
            eval_func,
            method,
            init_params,
            optimizer_options,
            _optimizer_callback if callback is not None else None,
            batch_eval=batch_eval,
        )
        return best_value, best_params

    best_value, best_params = method(eval_func, init_params, optimizer_options)

    return float(best_value), np.asarray(best_params, dtype=float)
