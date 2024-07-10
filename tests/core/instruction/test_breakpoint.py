import pytest
from pytest import CaptureFixture

from mpqp import Breakpoint, QCircuit
from mpqp.execution import run
from mpqp.execution.devices import ATOSDevice
from mpqp.gates import *


@pytest.mark.parametrize(
    "circuit, expected_out",
    [
        (
            QCircuit([H(0), CNOT(0, 1), Breakpoint(), Y(1)]),
            """\
DEBUG: After instruction 2, state is
       0.707|00⟩ + 0.707|11⟩
""",
        ),
        (
            QCircuit([H(0), CNOT(0, 1), Breakpoint(draw_circuit=True), Y(1)]),
            """\
DEBUG: After instruction 2, state is
       0.707|00⟩ + 0.707|11⟩
       and circuit is
            ┌───┐     
       q_0: ┤ H ├──■──
            └───┘┌─┴─┐
       q_1: ─────┤ X ├
                 └───┘
""",
        ),
        (
            QCircuit([H(0), Y(0), Breakpoint(draw_circuit=True), CNOT(0, 1)]),
            """\
DEBUG: After instruction 2, state is
       0.707j|00⟩ + 0.707j|10⟩
       and circuit is
            ┌───┐┌───┐
       q_0: ┤ H ├┤ Y ├
            └───┘└───┘
       q_1: ──────────
               
""",
        ),
        (
            QCircuit([H(0), CNOT(0, 1), Breakpoint(enabled=False), Y(1)]),
            "",
        ),
        (
            QCircuit([H(0), CNOT(0, 1), Breakpoint(label="Bell state"), Y(1)]),
            """\
DEBUG: After instruction 2, at breakpoint `Bell state`, state is
       0.707|00⟩ + 0.707|11⟩
""",
        ),
        (
            QCircuit(
                [
                    H(0),
                    Breakpoint(enabled=False),
                    CNOT(0, 1),
                    Breakpoint(draw_circuit=True, label="Bell state"),
                    Y(1),
                    Breakpoint(),
                ]
            ),
            """\
DEBUG: After instruction 2, at breakpoint `Bell state`, state is
       0.707|00⟩ + 0.707|11⟩
       and circuit is
            ┌───┐     
       q_0: ┤ H ├──■──
            └───┘┌─┴─┐
       q_1: ─────┤ X ├
                 └───┘
DEBUG: After instruction 3, state is
       0.707j|01⟩ - 0.707j|10⟩
""",
        ),
    ],
)
def test_capture(circuit: QCircuit, expected_out: str, capsys: CaptureFixture[str]):
    run(circuit, ATOSDevice.MYQLM_CLINALG)
    captured = capsys.readouterr()
    assert captured.out == expected_out
