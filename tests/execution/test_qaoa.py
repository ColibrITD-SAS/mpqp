import sys

import pytest

from mpqp.execution.vqa.qaoa import MixerType, qaoa_solver
from mpqp.execution.vqa.qubo import Qubo

x = Qubo('x')
y = Qubo('y')
z = Qubo('z')

x0_1 = Qubo('x0_1')
x0_2 = Qubo('x0_2')
x1_0 = Qubo('x1_0')
x1_2 = Qubo('x1_2')
x2_0 = Qubo('x2_0')
x2_1 = Qubo('x2_1')


@pytest.mark.parametrize(
    "expr, depth, mixer, optimizer, state",
    [
        (2 * x, 2, MixerType.MIXER_X, "Powell", "0"),
        (x * 2 + 2, 2, MixerType.MIXER_X, "Powell", "0"),
        (x * 2 + 3 * y, 2, MixerType.MIXER_X, "Powell", "00"),
        (x * 2 - 3 * y, 2, MixerType.MIXER_X, "Powell", "01"),
        (
            3 * x * y - 4 * x - 2 * y,
            2,
            MixerType.MIXER_X,
            "Powell",
            "10",
        ),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 1,
            3,
            MixerType.MIXER_X,
            "Powell",
            "101",
        ),
        (2 * x + y + 3 * x + 4 * z, 3, MixerType.MIXER_X, "Powell", "000"),
        (3 * x - 2 * y + 100 * (x & y), 3, MixerType.MIXER_X, "Powell", "01"),
        (3 * x + 2 * y - 100 * (x & y), 2, MixerType.MIXER_X, "Powell", "11"),
        (3 * x - 2 * y + 100 * (x | y), 2, MixerType.MIXER_X, "Powell", "00"),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 100 * (x | z),
            3,
            MixerType.MIXER_X,
            "Powell",
            "010",
        ),
        (
            -10 * x - 100 * y + 100 * z - 1000 * (x ^ y),
            4,
            MixerType.MIXER_X,
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
            MixerType.MIXER_X,
            "Nelder-Mead",
            "011001",
        ),
    ],
)
def qaoa(expr: Qubo, depth: int, mixer: MixerType, optimizer: str, state: str):
    assert qaoa_solver(expr, depth, mixer, optimizer) == state


if "--long-local" in sys.argv or "--long" in sys.argv:
    test_qaoa = qaoa
