from typing import Any

import numpy as np
import pytest
from sympy import Expr, symbols

from mpqp import (
    ATOSDevice,
    AWSDevice,
    ExpectationMeasure,
    GOOGLEDevice,
    IBMDevice,
    Observable,
    Optimizer,
    QCircuit,
    minimize,
    run,
)
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.vqa.vqa import OptimizableFunc
from mpqp.gates import *

# the symbols function is a bit wacky, so some manual type definition is needed here
theta: Expr = symbols("θ")


def with_local_devices(args: tuple[Any, ...]):
    return (
        (*args, d)
        for d in list(IBMDevice)
        + list(ATOSDevice)
        + list(AWSDevice)
        + list(GOOGLEDevice)
        if not d.is_remote() and d.is_gate_based() and not d.has_reduced_gate_set()
    )


@pytest.mark.parametrize(
    "circ, minimum, device",
    with_local_devices(
        (
            QCircuit(
                [
                    P(theta, 0),
                    ExpectationMeasure(Observable(np.array([[0, 1], [1, 0]])), [0]),
                ]
            ),
            0,
        )
    ),
)
def test_optimizer_circuit(circ: QCircuit, minimum: float, device: AvailableDevice):
    def run():
        assert minimize(circ, Optimizer.BFGS, device)[0] - minimum < 0.05

    try:
        run()
    except (ValueError, NotImplementedError) as err:
        if "not handled" not in str(err):
            raise


@pytest.mark.parametrize(
    "eval_f, minimum, device",
    with_local_devices(
        (
            lambda params: (
                1
                - run(
                    QCircuit(
                        [
                            P(theta, 0),
                            ExpectationMeasure(
                                Observable(np.array([[0, 1], [1, 0]])), [0]
                            ),
                        ]
                    ),
                    ATOSDevice.MYQLM_PYLINALG,
                    {theta: params[0]},
                ).expectation_values  # pyright: ignore[reportOperatorIssue]
                ** 2
            ),
            1,
        )
    ),
)
def test_optimizer_func(
    eval_f: OptimizableFunc, minimum: float, device: AvailableDevice
):
    try:
        assert minimize(eval_f, Optimizer.BFGS, nb_params=1)[0] - minimum < 0.05
    except (ValueError, NotImplementedError) as err:
        if "not handled" not in str(err):
            raise
