import pytest
from pytest import CaptureFixture

from mpqp import CNOT, ATOSDevice, Breakpoint, H, QCircuit, Y, run
from mpqp.execution.devices import (
    AvailableDevice,
    IBMDevice,
    GOOGLEDevice,
    ATOSDevice,
    AWSDevice,
)

list_circuit_expected_out = [
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
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "circuit, expected_out",
    list_circuit_expected_out,
)
def test_capture_qiskit(
    circuit: QCircuit, expected_out: str, capsys: CaptureFixture[str]
):
    exec_capture(circuit, expected_out, capsys, IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circuit, expected_out",
    list_circuit_expected_out,
)
def test_capture_braket(
    circuit: QCircuit, expected_out: str, capsys: CaptureFixture[str]
):
    exec_capture(circuit, expected_out, capsys, AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize(
    "circuit, expected_out",
    list_circuit_expected_out,
)
def test_capture_cirq(
    circuit: QCircuit, expected_out: str, capsys: CaptureFixture[str]
):
    exec_capture(circuit, expected_out, capsys, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize(
    "circuit, expected_out",
    list_circuit_expected_out,
)
def test_capture_myqlm(
    circuit: QCircuit, expected_out: str, capsys: CaptureFixture[str]
):
    exec_capture(circuit, expected_out, capsys, ATOSDevice.MYQLM_CLINALG)


def exec_capture(
    circuit: QCircuit,
    expected_out: str,
    capsys: CaptureFixture[str],
    device: AvailableDevice,
):
    run(circuit, device)
    captured = capsys.readouterr()
    assert captured.out == expected_out
