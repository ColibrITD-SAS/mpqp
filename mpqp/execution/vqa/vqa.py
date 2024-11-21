from __future__ import annotations

from typing import Any, Callable, Collection, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize
from sympy import Expr
from typeguard import typechecked

from mpqp.core.circuit import QCircuit
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.execution.vqa.optimizer import Optimizer

T1 = TypeVar("T1")
T2 = TypeVar("T2")
OptimizerInput = Union[list[float], npt.NDArray[np.float32]]
OptimizableFunc = Callable[[OptimizerInput], float]
OptimizerOptions = dict[str, Any]
OptimizerCallable = Callable[
    [OptimizableFunc, Optional[OptimizerInput], Optional[OptimizerOptions]],
    tuple[float, OptimizerInput],
]

# TODO: all those functions with almost or exactly the same signature look like
#  a code smell to me.

# TODO: test the minimizer options


def _maps(l1: Collection[T1], l2: Collection[T2]) -> dict[T1, T2]:
    """Does like zip, but with a dictionary instead of a list of tuples"""
    if len(l1) != len(l2):
        ValueError(
            f"Length of the two collections are not equal ({len(l1)} and {len(l2)})."
        )
    return {e1: e2 for e1, e2 in zip(l1, l2)}


@typechecked
def minimize(
    optimizable: QCircuit | OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    device: Optional[AvailableDevice] = None,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[dict[str, Any]] = None,
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
        ...     return 1 - run_res.expectation_value ** 2
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
        optimizer = _minimize_remote if device.is_remote() else _minimize_local
        return optimizer(optimizable, method, device, init_params, nb_params)
    else:
        # TODO: find a way to know if the job is remote or local from the function
        return _minimize_local(
            optimizable, method, device, init_params, nb_params, optimizer_options
        )


@typechecked
def _minimize_remote(
    optimizable: QCircuit | OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    device: Optional[AvailableDevice] = None,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[dict[str, Any]] = None,
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

    Returns:
        The optimal value reached and the parameters used to reach this value.

    TODO to implement on QLM first
    """
    raise NotImplementedError()


@typechecked
def _minimize_local(
    optimizable: QCircuit | OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    device: Optional[AvailableDevice] = None,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[dict[str, Any]] = None,
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

    Returns:
        the optimal value reached and the parameters used to reach this value.
    """
    if isinstance(optimizable, QCircuit):
        if device is None:
            raise ValueError("A device is needed to optimize a circuit")
        return _minimize_local_circ(
            optimizable, device, method, init_params, optimizer_options
        )
    else:
        return _minimize_local_func(
            optimizable, method, init_params, nb_params, optimizer_options
        )


@typechecked
def _minimize_local_circ(
    circ: QCircuit,
    device: AvailableDevice,
    method: Optimizer | OptimizerCallable,
    init_params: Optional[OptimizerInput] = None,
    optimizer_options: Optional[dict[str, Any]] = None,
) -> tuple[float, OptimizerInput]:
    """This function runs an optimization on the parameters of the circuit, to
    minimize the expectation value of the measure of the circuit by it's
    observables. Note that this means that the circuit should contain an
    expectation measure!

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

    Returns:
        The optimal value reached and the parameters used to reach this value.
    """
    # The sympy `free_symbols` method returns in fact sets of Basic, which
    # are theoretically different from Expr, but in our case the difference
    # is not relevant.
    # TODO: bellow might be a bug, check why we need this type ignore
    variables: set[Expr] = circ.variables()  # pyright: ignore[reportAssignmentType]

    def eval_circ(params: OptimizerInput):
        # pyright is bad with abstract numeric types:
        # "float" is incompatible with "Complex"
        return _run_single(
            circ,
            device,
            _maps(variables, params),  # pyright: ignore[reportArgumentType]
        ).expectation_value

    return _minimize_local_func(
        eval_circ, method, init_params, len(variables), optimizer_options
    )


@typechecked
def _minimize_local_func(
    eval_func: OptimizableFunc,
    method: Optimizer | OptimizerCallable,
    init_params: Optional[OptimizerInput] = None,
    nb_params: Optional[int] = None,
    optimizer_options: Optional[OptimizerOptions] = None,
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

    if isinstance(method, Optimizer):
        res: OptimizeResult = scipy_minimize(
            eval_func,
            x0=np.array(init_params),
            method=method.name.lower(),
            options=optimizer_options,
        )
        return res.fun, res.x
    else:
        return method(eval_func, init_params, optimizer_options)
