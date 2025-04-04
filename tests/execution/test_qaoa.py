import pytest
from mpqp.execution.vqa.qaoa import qaoa_solver, MixerType
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
    "expr, depth, mixer, state",
    [
        (
            2 * x, 
            2, 
            MixerType.MIXER_X, 
            "0"
        ),
        (
            x * 2 + 2, 
            2, 
            MixerType.MIXER_X, 
            "0"
        ),
        (
            x * 2 + 3 * y, 
            2, 
            MixerType.MIXER_X, 
            "00"
        ),
        (
            x * 2 - 3 * y,
            2, 
            MixerType.MIXER_X, 
            "01"
        ),
        (
            3 * x * y - 4 * x - 2 * y,
            2,
            MixerType.MIXER_X,
            "10",
        ),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 1,
            3,
            MixerType.MIXER_X,
            "101",
        ),
        (
            2 * x + y + 3 * x + 3 * z, 
            3, 
            MixerType.MIXER_X, 
            "000"
        ),
        (
            3 * x - 2 * y + 100 * (x & y), 
            4, 
            MixerType.MIXER_X, 
            "01"
        ),
        (
            3 * x + 2 * y - 100 * (x & y), 
            2, 
            MixerType.MIXER_X, 
            "11"
        ),
        (
            3 * x - 2 * y + 100 * (x | y), 
            2, 
            MixerType.MIXER_X, 
            "00"
        ),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 100 * (x | z),
            3,
            MixerType.MIXER_X,
            "010",
        ),
        (
            -10 * x - 100 * y + 100 * z - 1000 * (x ^ y),
            4,
            MixerType.MIXER_X,
            "010",
        ),
        (
            2 * x0_1 + x0_2 + 2 * x1_0 + 4 * x1_2 + x2_0 + x2_1 - 100 * ((x0_1 ^ x2_1) + (x1_0 ^ x2_0) + (x1_2 ^ x0_2)) - 100 * ((x0_1 ^ x1_0) + (x0_2 ^ x2_0) + (x1_2 ^ x2_1)) - 100 * ((x0_1 ^ x0_2) + (x1_0 ^ x1_2) + (x2_0 ^ x2_1)),
            3,
            MixerType.MIXER_X,
            "010101",
        ),
    ],
)
def test_qaoa(expr: Qubo, depth: int, mixer: MixerType, state: str):

    res = qaoa_solver(expr, depth, mixer)
    assert res == state
