import sys

import pytest

from mpqp.execution import AvailableDevice, IBMDevice
from mpqp.execution.vqa.qaoa import QAOAMixerType, qaoa_solver
from mpqp.execution.vqa.qubo import Qubo, QuboAtom

x = QuboAtom('x')
y = QuboAtom('y')
z = QuboAtom('z')

x0_1 = QuboAtom('x0_1')
x0_2 = QuboAtom('x0_2')
x1_0 = QuboAtom('x1_0')
x1_2 = QuboAtom('x1_2')
x2_0 = QuboAtom('x2_0')
x2_1 = QuboAtom('x2_1')


@pytest.mark.parametrize(
    "expr, depth, mixer, device, optimizer, state",
    [
        (2 * x, 2, QAOAMixerType.MIXER_X, IBMDevice.AER_SIMULATOR, "Powell", "0"),
        (x * 2 + 2, 2, QAOAMixerType.MIXER_X, IBMDevice.AER_SIMULATOR, "Powell", "0"),
        (
            x * 2 + 3 * y,
            2,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "00",
        ),
        (
            x * 2 - 3 * y,
            2,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "01",
        ),
        (
            3 * x * y - 4 * x - 2 * y,
            2,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "10",
        ),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 1,
            3,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "101",
        ),
        (
            2 * x + y + 3 * x + 4 * z,
            3,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "000",
        ),
        (
            3 * x - 2 * y + 100 * (x & y),
            3,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "01",
        ),
        (
            3 * x + 2 * y - 100 * (x & y),
            2,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "11",
        ),
        (
            3 * x - 2 * y + 100 * (x | y),
            2,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "00",
        ),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 100 * (x | z),
            3,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "010",
        ),
        (
            -10 * x - 100 * y + 100 * z - 1000 * (x ^ y),
            4,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Powell",
            "010",
        ),
        (
            2 * x0_1
            + x0_2
            + 2 * x1_0
            + 4 * x1_2
            + x2_0
            + x2_1
            - 10 * ((x0_1 ^ x2_1) + (x1_0 ^ x2_0) + (x1_2 ^ x0_2))
            - 10 * ((x0_1 ^ x0_2) + (x1_0 ^ x1_2) + (x2_0 ^ x2_1)),
            5,
            QAOAMixerType.MIXER_X,
            IBMDevice.AER_SIMULATOR,
            "Nelder-Mead",
            "011001",
        ),
    ],
)
def qaoa(
    expr: Qubo,
    depth: int,
    mixer: QAOAMixerType,
    device: AvailableDevice,
    optimizer: str,
    state: str,
):
    assert qaoa_solver(expr, depth, mixer, device, optimizer) == state


if "--long-local" in sys.argv or "--long" in sys.argv:
    test_qaoa = qaoa
