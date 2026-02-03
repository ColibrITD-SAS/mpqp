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


def with_local_devices_qiskit(args: tuple[Any, ...]):
    return (
        (*args, d)
        for d in list(IBMDevice)
        if not d.is_remote() and d.is_gate_based() and not d.has_reduced_gate_set()
    )


def with_local_devices_braket(args: tuple[Any, ...]):
    return (
        (*args, d)
        for d in list(AWSDevice)
        if not d.is_remote() and d.is_gate_based() and not d.has_reduced_gate_set()
    )


def with_local_devices_myqlm(args: tuple[Any, ...]):
    return (
        (*args, d)
        for d in list(ATOSDevice)
        if not d.is_remote() and d.is_gate_based() and not d.has_reduced_gate_set()
    )


def with_local_devices_cirq(args: tuple[Any, ...]):
    return (
        (*args, d)
        for d in list(GOOGLEDevice)
        if not d.is_remote() and d.is_gate_based() and not d.has_reduced_gate_set()
    )


circuit_min = (
    QCircuit(
        [
            P(theta, 0),
            ExpectationMeasure(Observable(np.array([[0, 1], [1, 0]])), [0]),
        ]
    ),
    0,
)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "circ, minimum, device",
    with_local_devices_qiskit(circuit_min),
)
def test_optimizer_circuit_qiskit(
    circ: QCircuit, minimum: float, device: AvailableDevice
):
    exec_optimizer_circuit(circ, minimum, device)


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circ, minimum, device",
    with_local_devices_braket(circuit_min),
)
def test_optimizer_circuit_braket(
    circ: QCircuit, minimum: float, device: AvailableDevice
):
    exec_optimizer_circuit(circ, minimum, device)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize(
    "circ, minimum, device",
    with_local_devices_cirq(circuit_min),
)
def test_optimizer_circuit_cirq(
    circ: QCircuit, minimum: float, device: AvailableDevice
):
    exec_optimizer_circuit(circ, minimum, device)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize(
    "circ, minimum, device",
    with_local_devices_myqlm(circuit_min),
)
def test_optimizer_circuit_myqlm(
    circ: QCircuit, minimum: float, device: AvailableDevice
):
    exec_optimizer_circuit(circ, minimum, device)


def exec_optimizer_circuit(circ: QCircuit, minimum: float, device: AvailableDevice):
    def run():
        assert minimize(circ, Optimizer.BFGS, device)[0] - minimum < 0.05

    try:
        run()
    except (ValueError, NotImplementedError) as err:
        if "not handled" not in str(err):
            raise


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "eval_f, minimum, device",
    with_local_devices_qiskit(
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
                    IBMDevice.AER_SIMULATOR,
                    {theta: params[0]},
                ).expectation_values  # pyright: ignore[reportOperatorIssue]
                ** 2
            ),
            1,
        )
    ),
)
def test_optimizer_func_qiskit(
    eval_f: OptimizableFunc, minimum: float, device: AvailableDevice
):
    exec_optimizer_func(eval_f, minimum, device)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize(
    "eval_f, minimum, device",
    with_local_devices_myqlm(
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
def test_optimizer_func_myqlm(
    eval_f: OptimizableFunc, minimum: float, device: AvailableDevice
):
    exec_optimizer_func(eval_f, minimum, device)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize(
    "eval_f, minimum, device",
    with_local_devices_cirq(
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
                    GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
                    {theta: params[0]},
                ).expectation_values  # pyright: ignore[reportOperatorIssue]
                ** 2
            ),
            1,
        )
    ),
)
def test_optimizer_func_cirq(
    eval_f: OptimizableFunc, minimum: float, device: AvailableDevice
):
    exec_optimizer_func(eval_f, minimum, device)


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "eval_f, minimum, device",
    with_local_devices_braket(
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
                    AWSDevice.BRAKET_LOCAL_SIMULATOR,
                    {theta: params[0]},
                ).expectation_values  # pyright: ignore[reportOperatorIssue]
                ** 2
            ),
            1,
        )
    ),
)
def test_optimizer_func_braket(
    eval_f: OptimizableFunc, minimum: float, device: AvailableDevice
):
    exec_optimizer_func(eval_f, minimum, device)


def exec_optimizer_func(
    eval_f: OptimizableFunc, minimum: float, device: AvailableDevice
):
    try:
        assert minimize(eval_f, Optimizer.BFGS, nb_params=1)[0] - minimum < 0.05
    except (ValueError, NotImplementedError) as err:
        if "not handled" not in str(err):
            raise
