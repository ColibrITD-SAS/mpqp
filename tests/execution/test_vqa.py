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
    pZ,
)
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.vqa.vqa import (
    OptimizerData,
    OptimizerInput,
    RunMode,
    VQACircuit,
    VQAModule,
)
from mpqp.gates import *

# the symbols function is a bit wacky, so some manual type definition is needed here
theta: Expr = symbols("θ")


def with_local_devices_qiskit(args: tuple[Any, ...]):
    return ((*args, IBMDevice.AER_SIMULATOR),)


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


vqa_circuit = (
    VQACircuit(
        QCircuit(
            [
                P(theta, 0),
            ]
        ),
        ExpectationMeasure([Observable(pZ)]),
    ),
)


def exec_vqa_optimizer(vqa_circuit: VQACircuit, device: AvailableDevice):
    vqa = VQAModule(vqa_circuit, device)

    opt_data = OptimizerData(
        method=Optimizer.BFGS,
        init_params=[0.1],
        optimizer_options={"maxiter": 50},
    )

    try:
        val, _ = vqa.minimize(opt_data)

        # expected minimum ~ -1 or 0 depending formulation
        assert abs(val) < 1.1

    except (ValueError, NotImplementedError) as err:
        if "not handled" not in str(err):
            raise


# --------------------------------------------------
# TESTS per provider
# --------------------------------------------------


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("vqa_circuit, device", with_local_devices_qiskit(vqa_circuit))
def test_vqa_module_qiskit(vqa_circuit: VQACircuit, device: AvailableDevice):
    print(device)
    exec_vqa_optimizer(vqa_circuit, device)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("vqa_circuit, device", with_local_devices_braket(vqa_circuit))
def test_vqa_module_braket(vqa_circuit: VQACircuit, device: AvailableDevice):
    exec_vqa_optimizer(vqa_circuit, device)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("vqa_circuit, device", with_local_devices_cirq(vqa_circuit))
def test_vqa_module_cirq(vqa_circuit: VQACircuit, device: AvailableDevice):
    exec_vqa_optimizer(vqa_circuit, device)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("vqa_circuit, device", with_local_devices_myqlm(vqa_circuit))
def test_vqa_module_myqlm(vqa_circuit: VQACircuit, device: AvailableDevice):
    exec_vqa_optimizer(vqa_circuit, device)


def make_two_vqa_circuits():
    circ1 = QCircuit([P(theta, 0)])
    circ2 = QCircuit([P(theta, 0)])

    obs = ExpectationMeasure([Observable(pZ)])

    return [
        VQACircuit(circ1, obs),
        VQACircuit(circ2, obs),
    ]


def exec_vqa_multi(vqa_circuit: VQACircuit, device: AvailableDevice):
    vqa = VQAModule(vqa_circuit, device)

    opt_data = OptimizerData(
        method=Optimizer.BFGS,
        init_params=[0.1],
    )

    try:
        val, _ = vqa.minimize(opt_data)
        assert abs(val) < 2.0

    except (ValueError, NotImplementedError) as err:
        if "not handled" not in str(err):
            raise


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("vqa_circuit, device", with_local_devices_qiskit(vqa_circuit))
def test_vqa_multi_qiskit(vqa_circuit: VQACircuit, device: AvailableDevice):
    exec_vqa_multi(vqa_circuit, device)


def exec_vqa_modes(vqa_circuit: VQACircuit, device: AvailableDevice):
    vqa = VQAModule(vqa_circuit, device)

    opt_data = OptimizerData(
        method=Optimizer.BFGS,
        init_params=[0.2],
        optimizer_options={"maxiter": 1},
    )

    for mode in [RunMode.IDEAL, RunMode.OBS, RunMode.NOT_IDEAL, RunMode.STATE_VECTOR]:
        val, _ = vqa.minimize(
            opt_data,
            run_mode=mode,
            shots=10,
        )
        assert isinstance(val, float)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("vqa_circuit, device", with_local_devices_qiskit(vqa_circuit))
def test_vqa_modes_qiskit(vqa_circuit: VQACircuit, device: AvailableDevice):
    exec_vqa_modes(vqa_circuit, device)
