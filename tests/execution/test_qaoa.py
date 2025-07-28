import sys

import pytest
from networkx import Graph

from mpqp.core.instruction.measurement.expectation_value import Observable
from mpqp.execution import AvailableDevice, IBMDevice
from mpqp.execution.vqa.qaoa import QaoaMixer, QaoaMixerType, qaoa_solver
from mpqp.execution.vqa.qubo import Qubo, QuboAtom
from mpqp.execution.vqa import Optimizer

x = QuboAtom('x')
y = QuboAtom('y')
z = QuboAtom('z')

x0_1 = QuboAtom('x0_1')
x0_2 = QuboAtom('x0_2')
x1_0 = QuboAtom('x1_0')
x1_2 = QuboAtom('x1_2')
x2_0 = QuboAtom('x2_0')
x2_1 = QuboAtom('x2_1')

mixer_x = QaoaMixer(QaoaMixerType.MIXER_X)

graph_3 = Graph([(0, 1), (0, 2)])
mixer_xy_3 = QaoaMixer(QaoaMixerType.MIXER_XY, graph_3)
mixer_bitflip_3 = QaoaMixer(QaoaMixerType.MIXER_BITFLIP, graph_3, 0)


@pytest.mark.parametrize(
    "expr, depth, mixer, device, optimizer, init_params, state",
    [
        (2 * x, 2, mixer_x, IBMDevice.AER_SIMULATOR, Optimizer.POWELL, [1] * 4, "0"),
        (
            x * 2 + 2,
            2,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 4,
            "0",
        ),
        (
            x * 2 + 3 * y,
            2,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1.0] * 4,
            "00",
        ),
        (
            -3 * x - 2 * y + 4 * x * y - 3 * z,
            2,
            mixer_xy_3,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1.0] * 4,
            "101",
        ),
        (
            x * 2 - 3 * y,
            2,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1.0] * 4,
            "01",
        ),
        (
            -3 * x - 2 * y + 4 * x * y - z,
            2,
            mixer_bitflip_3,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1.0] * 4,
            "101",
        ),
        (
            3 * x * y - 4 * x - 2 * y,
            2,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1.0] * 4,
            "10",
        ),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 1,
            2,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 4,
            "101",
        ),
        (
            2 * x + y + 3 * x + 4 * z,
            2,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 4,
            "000",
        ),
        (
            3 * x - 2 * y + 100 * (x & y),
            3,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 6,
            "01",
        ),
        (
            3 * x - 2 * y + 100 * (x & y),
            3,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 6,
            "01",
        ),
        (
            3 * x + 2 * y - 100 * (x & y),
            2,
            Observable(mixer_x.generate_mixer_hamiltonian(2)),
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 4,
            "11",
        ),
        (
            3 * x - 2 * y + 100 * (x | y),
            2,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 4,
            "00",
        ),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 100 * (x | z),
            3,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 6,
            "010",
        ),
        (
            -10 * x - 100 * y + 100 * z - 1000 * (x ^ y),
            2,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 4,
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
            3,
            mixer_x,
            IBMDevice.AER_SIMULATOR,
            Optimizer.POWELL,
            [1] * 6,
            "011001",
        ),
    ],
)
def qaoa(
    expr: Qubo,
    depth: int,
    mixer: QaoaMixer,
    device: AvailableDevice,
    optimizer: Optimizer,
    init_params: list[float],
    state: str,
):
    result = qaoa_solver(expr, depth, mixer, device, optimizer, init_params)
    print(result.values)
    assert result.final_state == state


if "--long-local" in sys.argv or "--long" in sys.argv:
    test_qaoa = qaoa
