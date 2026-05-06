from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
import itertools
from random import random
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
from sympy import Basic
from tqdm import tqdm

from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.expectation_value import Observable
from mpqp.core.languages import Language
from mpqp.execution.result import Result
from mpqp.tools.generics import OneOrMany

if TYPE_CHECKING:
    from sympy import Expr

from mpqp.core import QCircuit
from mpqp.core.instruction import ExpectationMeasure
from mpqp.execution.devices import AWSDevice, AvailableDevice, IBMDevice
from mpqp.execution.job import ExecutionMode
from mpqp.execution.runner import ValuesDict, run
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
    maxiter: Optional[int] = None
    optimizer_options: Optional[dict[str, Any]] = None
    callback: Optional[OptimizerCallback] = None


class RunMode(Enum):
    EXPECTATION = auto()
    SAMPLING = auto()
    STATEVECTOR = auto()


class VQAResult:
    def __init__(self):
        self.loss_total: list[float] = []
        self.angles: dict[Basic, float] = {}
        self.loss: float = 0.0
        self.optimizer_results: Any = None

    
    def __str__(self) -> str:
        return f"Loss: {self.loss} \nAngles: {self.angles}"


class VQAModule:
    def __init__(
        self,
        circuits: OneOrMany[QCircuit],
        device: AvailableDevice,
    ):
        self.backend = device
        if isinstance(circuits, QCircuit):
            circuits = [circuits]
        self.circuits = circuits

        for circ in self.circuits:
            circ.transpiled_for_device(self.backend)


    def _default_cost_function(self, current_params: OptimizerInput) -> float:
        results: list[Result] = []
        for circ in self.circuits:
            current_params_dict: ValuesDict = {var: current_params[self.variables.index(var)] for var in circ.variables()}
            results.append(run(circ, self.backend, values=current_params_dict, mode=self.mode))
        self.result.loss = sum(result.expectation_values for result in results)
        return self.result.loss

    def minimize(
        self,
        optimizer_data: OptimizerData,
        eval_func: Optional[OptimizableFunc] = None,
        mode: Optional[ExecutionMode] = ExecutionMode.JOB,
        shots: int = 0,
    ):
        self.result = VQAResult()
        tqdm_bar = None
        self.variables: list[Basic] = list(set([var for circ in self.circuits for var in circ.variables()]))
        if optimizer_data.init_params is None:
            optimizer_data.init_params = [0.0 for _ in range(len(self.variables))]
        if optimizer_data.callback is None and optimizer_data.maxiter is not None:
            
            pbar_desc = f"{optimizer_data.method}"
            tqdm_bar = tqdm(
                initial=1, total=optimizer_data.maxiter, desc=pbar_desc, unit="iter", leave=True
            )
            # iter_counter = itertools.count(1)
    
            def callback_update(xk, *args):
                """
                Callback some output data of optimizer (at each iter), save them, update the percentage bar and print the cost value

                Parameters:
                    xk: current parameter values

                """
                # iter  = next(iter_counter)
                self.result.loss_total.append(self.result.loss)
                tqdm_bar.update(1)
                tqdm_bar.set_postfix(cost=f"{float(self.result.loss):.6g}")
            
            optimizer_data.callback = callback_update

        self.run_mode = RunMode.EXPECTATION
        for circ in self.circuits:
            for inst in circ.instructions:
                if isinstance(inst, ExpectationMeasure):
                    inst.shots = shots

        self.mode = mode
        if eval_func is None:
            eval_func = self._default_cost_function

        if isinstance(optimizer_data.method, Optimizer):
            if optimizer_data.maxiter is not None:
                if optimizer_data.optimizer_options is None:
                    optimizer_data.optimizer_options = {}
                optimizer_data.optimizer_options["maxiter"] = optimizer_data.maxiter
            res = scipy_minimize(
                eval_func,
                x0=np.array(optimizer_data.init_params),
                method=optimizer_data.method.value,
                options=optimizer_data.optimizer_options,
                callback=optimizer_data.callback,
            )
            if tqdm_bar is not None:
                tqdm_bar.close()
            if TYPE_CHECKING:
                assert isinstance(res, OptimizeResult)
            self.result.loss = float(res.fun)
            self.result.angles = dict(zip(self.variables, res.x))
            return self.result
        else:
            return optimizer_data.method(
                eval_func, optimizer_data.init_params, optimizer_data.optimizer_options
            )
            return result 
