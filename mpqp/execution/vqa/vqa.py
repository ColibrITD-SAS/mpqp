from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

from mpqp.execution.result import Result
from mpqp.tools.generics import OneOrMany

if TYPE_CHECKING:
    from sympy import Expr

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction import ExpectationMeasure
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.job import ExecutionMode
from mpqp.execution.runner import run
from mpqp.execution.vqa.optimizer import Optimizer

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

# TODO: all those functions with almost or exactly the same signature look like
#  a code smell to me.

# TODO: test the minimizer options

# TODO: update doc with new arguments


@dataclass
class OptimizerData:
    method: Optimizer | OptimizerCallable
    init_params: Optional[OptimizerInput] = None
    optimizer_options: Optional[dict[str, Any]] = None
    callback: Optional[OptimizerCallback] = None


def minimize(
    optimizable: OneOrMany[QCircuit] | OptimizableFunc,
    optimizer_data: OptimizerData,
    device: Optional[AvailableDevice] = None,
    cost_function: Optional[EvaluationFunc] = None,
    nb_params: Optional[int] = None,
    mode: Optional[ExecutionMode] = ExecutionMode.JOB,
) -> tuple[float, OptimizerInput]:
    """This function runs an optimization on the parameters of the circuit, in
    order to minimize the measured expectation value of observables associated
    with the given circuit. Note that this means that the latter should contain
    an :class:`mpqp.core.instruction.measurement.expectation_value.ExpectationMeasure`.

    TODO: update doc

    Args:
        optimizable: Either the circuit, containing symbols and an expectation
            measure, or the evaluation function.
        method: The method used to optimize most of those methods come from
            ``scipy``. If the choices offered in this package are not
            covering your needs, you can define your own optimizer. This should
            be a function taking as input a function representing the circuit,
            with as many inputs as the circuit has parameters, and any optional
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
        >>> minimize( # doctest: +MYQLM
        ...     circuit,
        ...     Optimizer.BFGS,
        ...     ATOSDevice.MYQLM_PYLINALG,
        ...     optimizer_options={"maxiter":50},
        ... )
        (-0.9999999999999996, array([0., 0.]))


        >>> def cost_func(params): # doctest: +MYQLM
        ...     run_res = run(
        ...         circuit,
        ...         ATOSDevice.MYQLM_PYLINALG,
        ...         {alpha: params[0], beta: params[1]}
        ...     )
        ...     return 1 - run_res.expectation_values ** 2
        >>> minimize( # doctest: +MYQLM
        ...     cost_func,
        ...     Optimizer.BFGS,
        ...     nb_params=2,
        ...     optimizer_options={"maxiter":50},
        ... )
        (8.881784197001252e-16, array([0., 0.]))

    """
    if isinstance(optimizable, QCircuit):
        if device is None:
            raise ValueError("A device is needed to optimize a circuit.")
        # TODO: in case of remote take into account the job mode
        # TODO: enable the usage of the cost function here too
        optimizer = _minimize_remote if device.is_remote() else _minimize_local
        return optimizer(
            optimizable,
            optimizer_data,
            device,
            nb_params,
        )
    if isinstance(optimizable, Sequence):
        if device is None:
            raise ValueError("A device is needed to optimize circuits.")
        if cost_function is None:
            raise ValueError(
                "In order to optimize over several circuits, a `cost_function` "
                "is necessary to turn the results into a single value."
            )
        # TODO
        raise NotImplementedError()
    # TODO: find a way to know if the job is remote or local from the function
    return _minimize_local(
        optimizable,
        optimizer_data,
        device,
        nb_params,
    )


def _minimize_remote(
    optimizable: QCircuit | OptimizableFunc,
    optimizer_data: OptimizerData,
    device: Optional[AvailableDevice] = None,
    nb_params: Optional[int] = None,
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
    raise NotImplementedError()


def _minimize_local(
    optimizable: QCircuit | OptimizableFunc,
    optimizer_data: OptimizerData,
    device: Optional[AvailableDevice] = None,
    nb_params: Optional[int] = None,
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
        return _minimize_local_circ(optimizable, device, optimizer_data)
    else:
        return _minimize_local_func(optimizable, optimizer_data, nb_params)


def _minimize_local_circ(
    circ: QCircuit,
    device: AvailableDevice,
    optimizer_data: OptimizerData,
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
    # The sympy `free_symbols` method returns in fact sets of Basic, which
    # are theoretically different from Expr, but in our case the difference
    # is not relevant.
    variables: set["Expr"] = circ.variables()  # pyright: ignore[reportAssignmentType]

    if len(circ.measurements) != 1:
        raise ValueError("Cannot optimize a circuit containing several measurements.")

    if not isinstance(circ.measurements[0], ExpectationMeasure):
        raise ValueError("Expected an ExpectationMeasure to optimize the circuit.")
    else:
        if len(circ.measurements[0].observables) > 1:
            raise ValueError(
                "Expected only one observable in the ExpectationMeasure but got"
                f" {len(circ.measurements[0].observables)}"
            )

    def eval_circ(params: OptimizerInput):
        # pyright is bad with abstract numeric types:
        # "float" is incompatible with "Complex"
        from numbers import Complex

        params_fixed_type: Collection[Complex] = (
            params  # pyright: ignore[reportAssignmentType]
        )

        result = run(circ, device, dict(zip(variables, params_fixed_type)))
        if TYPE_CHECKING:
            assert isinstance(result.expectation_values, float)
        return result.expectation_values

    return _minimize_local_func(eval_circ, optimizer_data, len(variables))


def _minimize_local_func(
    eval_func: OptimizableFunc,
    optimizer_data: OptimizerData,
    nb_params: Optional[int] = None,
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
    if optimizer_data.init_params is None:
        if nb_params is None:
            raise ValueError(
                "Please provide either a set of initialization parameters or "
                "the number of parameters expected by the function."
            )
        optimizer_data.init_params = [0.0] * nb_params

    if isinstance(optimizer_data.method, Optimizer):
        res = scipy_minimize(
            eval_func,
            x0=np.array(optimizer_data.init_params),
            method=optimizer_data.method.name.lower(),
            options=optimizer_data.optimizer_options,
            callback=optimizer_data.callback,
        )
        if TYPE_CHECKING:
            assert isinstance(res, OptimizeResult)
        return float(res.fun), res.x
    else:
        return optimizer_data.method(
            eval_func, optimizer_data.init_params, optimizer_data.optimizer_options
        )


if __name__ == "__main__":
    # testing a all the possible combinations
    from sympy import symbols

    from mpqp import IBMDevice, Observable, Rx, Ry, pX, pZ

    try:
        print(
            minimize(
                QCircuit([Rx(symbols("theta"), 0), ExpectationMeasure(Observable(pZ))]),
                OptimizerData(Optimizer.BFGS),
            )
        )  # should error because no device & optimizable is a circuit
    except Exception as e:
        print(e)

    print(
        minimize(
            QCircuit([Rx(symbols("theta"), 0), ExpectationMeasure(Observable(pZ))]),
            OptimizerData(Optimizer.BFGS),
            device=IBMDevice.AER_SIMULATOR,
        )
    )

    try:
        print(
            minimize(
                QCircuit(
                    [
                        Rx(symbols("theta"), 0),
                        ExpectationMeasure([Observable(pZ), Observable(pX)]),
                    ]
                ),
                OptimizerData(Optimizer.BFGS),
                device=IBMDevice.AER_SIMULATOR,
            )
        )  # should error because two observables but no way to know what to do with it (no CF)
    except Exception as e:
        print(e)

    print(
        minimize(
            QCircuit(
                [
                    Rx(symbols("theta"), 0),
                    ExpectationMeasure([Observable(pZ), Observable(pX)]),
                ]
            ),
            OptimizerData(Optimizer.BFGS),
            device=IBMDevice.AER_SIMULATOR,
            cost_function=lambda rs: (
                rs[0].expectation_values ** 2  # type: ignore
                + rs[1].expectation_values ** 2  # type: ignore
            ),
        )
    )

    print(
        minimize(
            [
                QCircuit(
                    [Rx(symbols("theta0"), 0), ExpectationMeasure([Observable(pZ)])]
                ),
                QCircuit(
                    [Ry(symbols("theta1"), 0), ExpectationMeasure([Observable(pX)])]
                ),
            ],
            OptimizerData(Optimizer.BFGS),
            device=IBMDevice.AER_SIMULATOR,
            cost_function=lambda rs: (
                rs[0].expectation_values ** 2  # type: ignore
                + rs[1].expectation_values ** 2  # type: ignore
            ),
        )
    )

    print(
        minimize(
            QCircuit([Rx(symbols("theta0"), 0), ExpectationMeasure([Observable(pZ)])]),
            OptimizerData(Optimizer.BFGS),
            device=IBMDevice.AER_SIMULATOR,
            mode=ExecutionMode.SESSION,
        )
    )  # should raise a warning because local device + session mode

    print(
        minimize(
            QCircuit([Rx(symbols("theta"), 0), ExpectationMeasure(Observable(pZ))]),
            OptimizerData(Optimizer.BFGS),
            device=IBMDevice.IBM_LEAST_BUSY,
        )
    )

    print(
        minimize(
            QCircuit([Rx(symbols("theta"), 0), ExpectationMeasure(Observable(pZ))]),
            OptimizerData(Optimizer.BFGS),
            device=IBMDevice.IBM_LEAST_BUSY,
        )
    )

    try:
        print(
            minimize(
                QCircuit(
                    [
                        Rx(symbols("theta"), 0),
                        ExpectationMeasure([Observable(pZ), Observable(pX)]),
                    ]
                ),
                OptimizerData(Optimizer.BFGS),
                device=IBMDevice.IBM_LEAST_BUSY,
                mode=ExecutionMode.SESSION,
            )  # should error because two observables but no way to know what to do with it (no CF)
        )
    except Exception as e:
        print(e)
