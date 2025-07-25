from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import pytest
from braket.circuits import Circuit as BraketCircuit, Noise as BraketNoise
from braket.circuits.gate import Gate as BraketGate
from qiskit import QuantumCircuit as QiskitCircuit, QuantumRegister, ClassicalRegister
from typeguard import TypeCheckError
from typing import TYPE_CHECKING

from mpqp import Barrier, Instruction, Language, QCircuit
from mpqp.core.instruction.gates import native_gates
from mpqp.core.instruction.gates.gate import SingleQubitGate
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.core.instruction.measurement.pauli_string import I
from mpqp.core.instruction.measurement.pauli_string import Z as Pauli_Z
from mpqp.execution.devices import ATOSDevice, IBMDevice
from mpqp.execution.runner import run, Result
from mpqp.gates import (
    CNOT,
    CZ,
    SWAP,
    TOF,
    CRk,
    Gate,
    H,
    Id,
    Rx,
    Ry,
    Rz,
    S,
    T,
    X,
    Y,
    Z,
    U,
    P,
)
from mpqp.measures import BasisMeasure, ExpectationMeasure, Observable
from mpqp.noise.noise_model import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    PhaseDamping,
    NoiseModel,
)
from mpqp.tools import NumberQubitsError
from mpqp.tools.circuit import (
    compute_expected_matrix,
    random_circuit,
    statevector_from_random_circuit,
)
from mpqp.tools.display import one_lined_repr
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning, NonReversibleWarning
from mpqp.tools.generics import Matrix, OneOrMany
from mpqp.tools.maths import matrix_eq
import random

from qiskit.circuit.random import random_circuit as random_qiskit_circuit
from cirq.testing.random_circuit import random_circuit as random_cirq_circuit
from cirq.circuits.circuit import Circuit as cirq_Circuit
from cirq.circuits.moment import Moment
from qat.core.wrappers.circuit import Circuit as myQLM_Circuit


@pytest.fixture
def list_qiskit_funky_circuits() -> list[QiskitCircuit]:
    qreg_q = QuantumRegister(4, 'q')
    creg_c = ClassicalRegister(4, 'c')
    qiskit_circuit_1 = QiskitCircuit(qreg_q, creg_c)
    from qiskit.circuit.library import RC3XGate

    pi = np.pi

    qiskit_circuit_1.sxdg(qreg_q[0])
    qiskit_circuit_1.sx(qreg_q[1])
    qiskit_circuit_1.sdg(qreg_q[2])
    qiskit_circuit_1.s(qreg_q[3])
    qiskit_circuit_1.append(RC3XGate(), [qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3]])
    qiskit_circuit_1.z(qreg_q[0])
    qiskit_circuit_1.s(qreg_q[1])
    qiskit_circuit_1.rxx(pi / 2, qreg_q[2], qreg_q[3])
    qiskit_circuit_1.id(qreg_q[2])
    qiskit_circuit_1.rx(pi / 2, qreg_q[0])
    qiskit_circuit_1.rz(pi / 2, qreg_q[1])
    qiskit_circuit_1.p(pi / 2, qreg_q[0])
    qiskit_circuit_1.rzz(pi / 2, qreg_q[2], qreg_q[3])
    qiskit_circuit_1.rzz(pi / 2, qreg_q[1], qreg_q[2])
    qiskit_circuit_1.rxx(pi / 2, qreg_q[0], qreg_q[1])
    qiskit_circuit_1.append(RC3XGate(), [qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3]])
    qiskit_circuit_1.rccx(qreg_q[1], qreg_q[2], qreg_q[3])
    qiskit_circuit_1.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
    qiskit_circuit_1.swap(qreg_q[0], qreg_q[1])
    qiskit_circuit_1.cx(qreg_q[2], qreg_q[3])
    qiskit_circuit_1.x(qreg_q[1])
    qiskit_circuit_1.h(qreg_q[0])
    qiskit_circuit_1.id(qreg_q[2])
    qiskit_circuit_1.id(qreg_q[3])

    return [qiskit_circuit_1]


@pytest.fixture
def list_braket_funky_circuits() -> list[BraketCircuit]:
    braket_circuit1 = BraketCircuit().h(0).x(control=0, target=1)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit1.ry(angle=0.13, target=2, control=(0, 1))
    braket_circuit1.x(0, power=1 / 5)

    braket_circuit2 = BraketCircuit()
    braket_circuit2.ccnot(0, 1, 2)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.cnot(0, 1)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.cphaseshift(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.cphaseshift00(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.cphaseshift01(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.cphaseshift10(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.cswap(0, 1, 2)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.swap(0, 1)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.phaseshift(0, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.cy(0, 1)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.cz(0, 1)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.ecr(0, 1)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.rx(0, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.ry(0, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.rz(0, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.h(range(3))  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.i([0, 1, 2])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.iswap(0, 1)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.x([1, 2])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.y([1, 2])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.z([1, 2])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.s([0, 1, 2])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.si([0, 1])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.t([0, 1])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.ti([0, 1])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.v([0, 1, 2])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.vi([0, 1, 2])  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.xx(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.xy(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.yy(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.zz(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.gpi(0, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.gpi2(0, 0.15)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit2.ms(0, 1, 0.15, 0.15, 0.15)  # type: ignore[reportAttributeAccessIssue]

    my_unitary = np.array([[0, 1], [1, 0]])
    braket_circuit3 = BraketCircuit()
    braket_circuit3.unitary(matrix=my_unitary, targets=[0])  # type: ignore[reportAttributeAccessIssue]
    QCircuit.from_other_language(braket_circuit3)

    braket_circuit4 = BraketCircuit().h(0).cnot(0, 1).measure(0)  # type: ignore[reportAttributeAccessIssue]

    braket_circuit5 = BraketCircuit().x(0).x(1).depolarizing(0, probability=0.1)  # type: ignore[reportAttributeAccessIssue]

    noise = BraketNoise.PhaseDamping(gamma=0.1)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit6 = BraketCircuit().x(0).y(1).cnot(0, 2).x(1).z(2)  # type: ignore[reportAttributeAccessIssue]
    braket_circuit6.apply_gate_noise(noise, target_gates=BraketGate.X)  # type: ignore[reportAttributeAccessIssue]

    braket_circuit7 = BraketCircuit().pswap(0, 1, 0.15)  # type: ignore[reportAttributeAccessIssue]

    return [
        braket_circuit1,
        braket_circuit2,
        braket_circuit3,
        braket_circuit4,
        braket_circuit5,
        braket_circuit6,
        braket_circuit7,
    ]


@pytest.fixture
def list_cirq_funky_circuits() -> list[cirq_Circuit | Moment]:
    import cirq

    q0, q1, q2 = cirq.LineQubit.range(3)  # type: ignore[reportPrivateImportUsage]

    cirq_circuit_1 = cirq.Circuit()  # type: ignore[reportPrivateImportUsage]

    cirq_circuit_1.append(cirq.X(q0))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.Y(q0))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.Z(q0))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.H(q0))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.S(q0))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.S(q0) ** -1)  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.T(q0))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.T(q0) ** -1)  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.rx(np.pi / 4)(q0))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.ry(np.pi / 4)(q0))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.rz(np.pi / 4)(q0))  # type: ignore[reportPrivateImportUsage]

    cirq_circuit_1.append(cirq.CX(q0, q1))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.ControlledGate(cirq.Y).on(q0, q1))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.CZ(q0, q1))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.ControlledGate(cirq.H).on(q0, q1))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.SWAP(q0, q1))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.ControlledGate(cirq.rz(np.pi / 4)).on(q0, q1))  # type: ignore[reportPrivateImportUsage]

    cirq_circuit_1.append(cirq.CCX(q0, q1, q2))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_1.append(cirq.CSWAP(q0, q1, q2))  # type: ignore[reportPrivateImportUsage]

    qubit = cirq.LineQubit(0)  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_2 = cirq.Circuit()  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_2.append(cirq.H(qubit))  # type: ignore[reportPrivateImportUsage]
    cirq_circuit_2.append(cirq.measure(qubit))  # type: ignore[reportPrivateImportUsage]

    q0, q1 = cirq.LineQubit.range(2)  # type: ignore[reportPrivateImportUsage]
    moment = cirq.Moment([cirq.H(q0), cirq.H(q1)])  # type: ignore[reportPrivateImportUsage]

    return [cirq_circuit_1, cirq_circuit_2, moment]


@pytest.fixture
def list_myqlm_funky_circuits() -> list[myQLM_Circuit]:
    from qat.lang.AQASM import Program, H, X, Y, Z, S, T, RX, RY, RZ, CNOT, SWAP, CCNOT, I, PH  # type: ignore[reportAttributeAccessIssue]

    prog = Program()
    qbits = prog.qalloc(3)

    prog.apply(I, qbits[0])
    prog.apply(X, qbits[0])
    prog.apply(Y, qbits[0])
    prog.apply(Z, qbits[0])
    prog.apply(H, qbits[0])
    prog.apply(S, qbits[0])
    prog.apply(S.dag(), qbits[0])
    prog.apply(T, qbits[0])
    prog.apply(T.dag(), qbits[0])
    prog.apply(RX(np.pi / 4), qbits[0])
    prog.apply(RY(np.pi / 4), qbits[0])
    prog.apply(RZ(np.pi / 4), qbits[0])
    prog.apply(PH(np.pi / 3), qbits[0])
    prog.apply(CNOT, qbits[0], qbits[1])
    prog.apply(SWAP, qbits[0], qbits[1])
    prog.apply(CCNOT, qbits[0], qbits[1], qbits[2])
    myqlm_circuit_1 = prog.to_circ()

    prog = Program()
    qbits = prog.qalloc(3)
    prog.apply(Y.ctrl(), qbits[0], qbits[1])
    prog.apply(Z.ctrl(), qbits[0], qbits[1])
    prog.apply(H.ctrl(), qbits[0], qbits[1])
    prog.apply(RZ(np.pi / 4).ctrl(), qbits[0], qbits[1])
    prog.apply(PH(np.pi / 4).ctrl(), qbits[0], qbits[1])
    myqlm_circuit_2 = prog.to_circ()

    prog = Program()
    qbits = prog.qalloc(3)

    prog.apply(I, qbits[0])
    prog.apply(X, qbits[0])
    prog.apply(Y, qbits[0])
    prog.apply(Z, qbits[0])
    results = prog.calloc(2)
    prog.measure(qbits[0], results[0])
    myqlm_circuit_3 = prog.to_circ()

    return [myqlm_circuit_1, myqlm_circuit_2, myqlm_circuit_3]


@pytest.mark.parametrize(
    "init_param, printed_result_filename",
    [
        (0, "empty"),
        ([], "empty"),
        ([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1)], "cnots"),
        ([CNOT(0, 1), X(0), BasisMeasure([0, 1], shots=100)], "measure"),
        ([CNOT(1, 2), Barrier(), X(0)], "barrier"),
    ],
)
def test_init_right(
    init_param: int | Sequence[Instruction], printed_result_filename: str
):
    with open(
        f"tests/core/test_circuit/init-{printed_result_filename}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        assert str(QCircuit(init_param)) == f.read()


@pytest.mark.parametrize(
    "init_param",
    [1.0, X(0), -1],
)
def test_init_wrong(init_param: int | Sequence[Instruction]):
    with pytest.raises(TypeCheckError):
        QCircuit(init_param)


@pytest.mark.parametrize(
    "init_circuit, added_gates, printed_result_filename",
    [
        (QCircuit(2), X(0), "X"),
        (
            QCircuit([X(0)], nb_qubits=2),
            [CNOT(0, 1), BasisMeasure([0, 1], shots=100)],
            "list",
        ),
    ],
)
def test_add(
    init_circuit: QCircuit,
    added_gates: OneOrMany[Instruction],
    printed_result_filename: str,
):
    with open(
        f"tests/core/test_circuit/add-{printed_result_filename}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        init_circuit.add(added_gates)
        assert str(init_circuit) == f.read()


@pytest.mark.parametrize(
    "instructions, result",
    [
        ([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), CNOT(2, 3)], 3),
        ([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1)], 3),
        ([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), X(2)], 3),
        ([CNOT(0, 1), CNOT(1, 2), CNOT(0, 1), Barrier(), X(2)], 4),
        ([CNOT(0, 1), CNOT(1, 2), CNOT(2, 3), X(0), X(0)], 3),
    ],
)
def test_depth(instructions: list[Instruction], result: int):
    assert QCircuit(instructions).depth() == result


@pytest.mark.parametrize(
    "circuit, result",
    [
        (QCircuit([CNOT(0, 1), CNOT(1, 2)]), (3, 0)),
        (QCircuit(3, nb_cbits=2), (3, 2)),
    ],
)
def test_size(circuit: QCircuit, result: tuple[int, Optional[int]]):
    assert circuit.size() == result


@pytest.mark.parametrize(
    "circuit, result",
    [
        (QCircuit([CNOT(0, 1), CNOT(1, 2), X(1), CNOT(1, 2)]), 4),
    ],
)
def test_len(circuit: QCircuit, result: int):
    assert len(circuit) == result


@pytest.mark.parametrize(
    "inst_list_1, inst_list_2",
    [
        ([CNOT(0, 1), CNOT(1, 2)], [X(1), CNOT(1, 2)]),
    ],
)
def test_append(inst_list_1: list[Instruction], inst_list_2: list[Instruction]):
    circ = QCircuit(inst_list_1)
    circ.append(QCircuit(inst_list_2))
    assert str(circ) == str(QCircuit(inst_list_1 + inst_list_2))
    assert str(QCircuit(inst_list_1) + QCircuit(inst_list_2)) == str(
        QCircuit(inst_list_1 + inst_list_2)
    )


@pytest.mark.parametrize(
    "inst_list_1, inst_list_2, inst_list_result",
    [
        (
            [CNOT(0, 1), CNOT(1, 2)],
            [X(1), CNOT(1, 2)],
            [CNOT(0, 1), CNOT(1, 2), X(4), CNOT(4, 5)],
        ),
    ],
)
def test_tensor(
    inst_list_1: list[Instruction],
    inst_list_2: list[Instruction],
    inst_list_result: list[Instruction],
):
    assert str(QCircuit(inst_list_1).tensor(QCircuit(inst_list_2))) == str(
        QCircuit(inst_list_result)
    )
    assert str(QCircuit(inst_list_1) @ (QCircuit(inst_list_2))) == str(
        QCircuit(inst_list_result)
    )


@pytest.mark.parametrize(
    "state",
    [
        (np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])),
        (np.array([1, 0, 0, 1]) / np.sqrt(2)),
        (statevector_from_random_circuit(1)),
        (statevector_from_random_circuit(2)),
        (statevector_from_random_circuit(4)),
        (statevector_from_random_circuit(6)),
        (np.array([1 / 2, np.sqrt(3) / 2])),
        (np.array([1, 1j]) / np.sqrt(2)),
    ],
)
def test_initializer(state: npt.NDArray[np.complex128]):
    qc = QCircuit.initializer(state)
    res = run(qc, IBMDevice.AER_SIMULATOR_STATEVECTOR)
    if TYPE_CHECKING:
        assert isinstance(res, Result)
    state_vector_initialized = res.state_vector.vector
    assert matrix_eq(state, state_vector_initialized)


@pytest.mark.parametrize(
    "circuit, filter, count",
    [
        (
            QCircuit(
                [X(0), Y(1), Z(2), CNOT(0, 1), SWAP(0, 1), CZ(1, 2), X(2), X(1), X(0)]
            ),
            (),
            9,
        ),
        (
            QCircuit(
                [X(0), Y(1), Z(2), CNOT(0, 1), SWAP(0, 1), CZ(1, 2), X(2), X(1), X(0)]
            ),
            (X,),
            4,
        ),
    ],
)
def test_count(circuit: QCircuit, filter: tuple[type[Gate]], count: int):
    assert circuit.count_gates(*filter) == count


@pytest.mark.parametrize(
    "circuit, result_repr",
    [
        (
            QCircuit(
                [
                    BasisMeasure([0, 1], shots=1000),
                    ExpectationMeasure(
                        Observable(np.identity(2, dtype=np.complex128)), [1], shots=1000
                    ),
                ]
            ),
            "[BasisMeasure([0, 1], shots=1000), ExpectationMeasure("
            "Observable(array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]]), 'observable_0'), [1], shots=1000)]",
        )
    ],
)
def test_get_measurements(circuit: QCircuit, result_repr: str):
    assert one_lined_repr(circuit.measurements) == result_repr


@pytest.mark.parametrize(
    "circuit, printed_result_filename",
    [
        (QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)]), "all"),
    ],
)
def test_without_measurements(circuit: QCircuit, printed_result_filename: str):
    with open(
        f"tests/core/test_circuit/wo_meas-{printed_result_filename}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        assert str(circuit.without_measurements()) == f.read()


@pytest.mark.parametrize(
    "circuit, args, result_type, result_repr",
    [
        (
            QCircuit([X(0), CNOT(0, 1)]),
            (),
            QiskitCircuit,
            (
                "[CircuitInstruction(operation=Instruction(name='x', num_qubits=1,"
                " num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),),"
                " clbits=()), CircuitInstruction(operation=Instruction(name='cx',"
                " num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2,"
                " 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=())]"
            ),
        ),
        (
            QCircuit([X(0), CNOT(0, 1)]),
            (Language.QISKIT,),
            QiskitCircuit,
            (
                "[CircuitInstruction(operation=Instruction(name='x', num_qubits=1,"
                " num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),),"
                " clbits=()), CircuitInstruction(operation=Instruction(name='cx',"
                " num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2,"
                " 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=())]"
            ),
        ),
        (
            QCircuit([CNOT(0, 1), Depolarizing(0.5, [0, 1])]),
            (Language.BRAKET,),
            BraketCircuit,
            (
                """\
T  : │         0         │
            ┌───────────┐ 
q0 : ───●───┤ DEPO(0.5) ├─
        │   └───────────┘ 
      ┌─┴─┐ ┌───────────┐ 
q1 : ─┤ X ├─┤ DEPO(0.5) ├─
      └───┘ └───────────┘ 
T  : │         0         │"""
            ),
        ),
        (
            QCircuit([CNOT(0, 1), Depolarizing(0.5, [0, 1], dimension=2)]),
            (Language.BRAKET,),
            BraketCircuit,
            (
                """\
T  : │         0         │
            ┌───────────┐ 
q0 : ───●───┤ DEPO(0.5) ├─
        │   └─────┬─────┘ 
      ┌─┴─┐ ┌─────┴─────┐ 
q1 : ─┤ X ├─┤ DEPO(0.5) ├─
      └───┘ └───────────┘ 
T  : │         0         │"""
            ),
        ),
        (
            QCircuit(
                [CNOT(0, 1), Depolarizing(0.5, [0, 1], dimension=2, gates=[CNOT])]
            ),
            (Language.BRAKET,),
            BraketCircuit,
            (
                """\
T  : │         0         │
            ┌───────────┐ 
q0 : ───●───┤ DEPO(0.5) ├─
        │   └─────┬─────┘ 
      ┌─┴─┐ ┌─────┴─────┐ 
q1 : ─┤ X ├─┤ DEPO(0.5) ├─
      └───┘ └───────────┘ 
T  : │         0         │"""
            ),
        ),
    ],
)
def test_to_other_language(
    circuit: QCircuit, args: tuple[Language], result_type: type, result_repr: str
):
    language = Language.QISKIT if len(args) == 0 else args[0]
    # TODO: test other languages
    if language == Language.BRAKET:
        with pytest.warns(UnsupportedBraketFeaturesWarning) as record:
            converted_circuit = circuit.to_other_language(*args)
        assert len(record) == 1
    else:
        converted_circuit = circuit.to_other_language(*args)
    assert type(converted_circuit) == result_type
    if isinstance(converted_circuit, QiskitCircuit):
        assert repr(converted_circuit.data) == result_repr
    if isinstance(converted_circuit, BraketCircuit):
        assert str(converted_circuit) == result_repr


def _create_large_circuits_for_tests() -> tuple[QiskitCircuit, QiskitCircuit]:
    from qiskit import QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import RC3XGate

    qreg_q = QuantumRegister(4, 'q')
    creg_c = ClassicalRegister(4, 'c')
    circuit = QiskitCircuit(qreg_q, creg_c)

    circuit.sxdg(qreg_q[0])
    circuit.sx(qreg_q[1])
    circuit.sdg(qreg_q[2])
    circuit.s(qreg_q[3])
    circuit.append(RC3XGate(), [qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3]])
    circuit.z(qreg_q[0])
    circuit.s(qreg_q[1])
    circuit.rxx(np.pi / 2, qreg_q[2], qreg_q[3])
    circuit.id(qreg_q[2])
    circuit.rx(np.pi / 2, qreg_q[0])
    circuit.rz(np.pi / 2, qreg_q[1])
    circuit.p(np.pi / 2, qreg_q[0])
    circuit.rzz(np.pi / 2, qreg_q[2], qreg_q[3])
    circuit.rzz(np.pi / 2, qreg_q[1], qreg_q[2])
    circuit.rxx(np.pi / 2, qreg_q[0], qreg_q[1])
    circuit.append(RC3XGate(), [qreg_q[0], qreg_q[1], qreg_q[2], qreg_q[3]])
    circuit.rccx(qreg_q[1], qreg_q[2], qreg_q[3])
    circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[2])
    circuit.swap(qreg_q[0], qreg_q[1])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.x(qreg_q[1])
    circuit.h(qreg_q[0])
    circuit.id(qreg_q[2])
    circuit.id(qreg_q[3])

    circuit_2 = QiskitCircuit(qreg_q, creg_c)
    circuit_2.rzz(np.pi / 2, qreg_q[1], qreg_q[2]).c_if(creg_c, 0)

    return circuit, circuit_2


@pytest.mark.parametrize(
    "circuit, language, expected_output",
    [
        (random_qiskit_circuit(2, 5), Language.QISKIT, None),
        (random_qiskit_circuit(5, 5), Language.QISKIT, None),
        (random_qiskit_circuit(10, 5), Language.QISKIT, None),
        (
            _create_large_circuits_for_tests()[0],
            Language.QISKIT,
            None,
        ),
        (
            _create_large_circuits_for_tests()[1],
            Language.QISKIT,
            "\"If\" instructions aren't handled",
        ),
        (QCircuit([H(0), CNOT(0, 1)]), Language.QASM2, None),
        (random_circuit(None, 2), Language.QASM2, None),
        (random_circuit(None, 10), Language.QASM2, None),
        (
            "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];",
            Language.QASM2,
            QCircuit([H(0), CNOT(0, 1)]),
        ),
        (
            "// Generated from Cirq v1.3.0\n\nOPENQASM 2.0;\n\n// Qubits: [q0, q1]\nqreg q[2];\nh q[0];\ncx q[0],q[1];",
            Language.QASM2,
            QCircuit([H(0), CNOT(0, 1)]),
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[4];
            creg c[4];
            z q[0];
            s q[1];
            id q[2];
            rxx(pi/2) q[2], q[3];
            rzz(pi/2) q[0], q[1];
            rx(pi/2) q[0];
            rccx q[1], q[2], q[3];
            p(pi/2) q[0];
            rz(pi/2) q[1];
            tdg q[2];
            ccx q[0], q[1], q[2];
            swap q[0], q[1];
            cx q[2], q[3];
            x q[1];
            h q[0];
            id q[2];
            id q[3];""",
            Language.QASM2,
            QCircuit(
                [
                    Z(0),
                    S(1),
                    Id(2),
                    H(2),
                    H(3),
                    CNOT(2, 3),
                    Rz(np.pi / 2, 3),
                    CNOT(2, 3),
                    H(3),
                    H(2),
                    CNOT(0, 1),
                    Rz(np.pi / 2, 1),
                    CNOT(0, 1),
                    Rx(np.pi / 2, 0),
                    U(np.pi / 2, 0, np.pi, 3),
                    U(0, 0, np.pi / 4, 3),
                    CNOT(2, 3),
                    U(0, 0, -np.pi / 4, 3),
                    CNOT(1, 3),
                    U(0, 0, np.pi / 4, 3),
                    CNOT(2, 3),
                    U(0, 0, -np.pi / 4, 3),
                    U(np.pi / 2, 0, np.pi, 3),
                    P(np.pi / 2, 0),
                    Rz(np.pi / 2, 1),
                    P(-np.pi / 4, 2),
                    TOF([0, 1], 2),
                    SWAP(0, 1),
                    CNOT(2, 3),
                    X(1),
                    H(0),
                    Id(2),
                    Id(3),
                ]
            ),
        ),
        (random_cirq_circuit(2, 5, 0.5), Language.CIRQ, None),
        (random_cirq_circuit(10, 5, 0.5), Language.CIRQ, None),
        (QCircuit([H(0), CNOT(0, 1)]), Language.BRAKET, None),
        (random_circuit(None, 2), Language.BRAKET, None),
        (random_circuit(None, 10), Language.BRAKET, None),
        (QCircuit([H(0), CNOT(0, 1)]), Language.MY_QLM, None),
        (random_circuit(None, 2), Language.MY_QLM, None),
        (random_circuit(None, 10), Language.MY_QLM, None),
        (
            "OPENQASM 3.0;include \"stdgates.inc\";qubit[2] q;h q[0];cx q[0], q[1];",
            Language.QASM3,
            QCircuit([H(0), CNOT(0, 1)]),
        ),
        (
            "//Generated with Qiskit\n\nOPENQASM 3.0;include \"stdgates.inc\";\n//Qubits\nqubit[2] q;h q[0];cx q[0], q[1];",
            Language.QASM3,
            QCircuit([H(0), CNOT(0, 1)]),
        ),
    ],
)
def test_from_other_language(
    circuit: QiskitCircuit | QCircuit | cirq_Circuit | str,
    language: Language,
    expected_output: Optional[str | QCircuit],
):
    if isinstance(circuit, QiskitCircuit):
        from qiskit.quantum_info import Operator

        if not isinstance(expected_output, str):
            qcircuit = QCircuit.from_other_language(circuit)
            circuit = circuit.reverse_bits()
            matrix = Operator(circuit).data
            if TYPE_CHECKING:
                assert isinstance(matrix, np.ndarray)
            assert matrix_eq(matrix, qcircuit.to_matrix())
        else:
            with pytest.raises(ValueError, match=expected_output):
                QCircuit.from_other_language(circuit)

    elif isinstance(circuit, cirq_Circuit):
        from cirq.protocols.unitary_protocol import unitary

        qcircuit = QCircuit.from_other_language(circuit)
        cirq_circuit = qcircuit.to_other_language(language)
        assert matrix_eq(unitary(cirq_circuit), unitary(circuit))

    elif language == Language.QASM3:
        qcircuit = QCircuit.from_other_language(circuit)
        if TYPE_CHECKING:
            assert isinstance(expected_output, QCircuit)
        assert matrix_eq(qcircuit.to_matrix(), expected_output.to_matrix())

    elif isinstance(circuit, str):
        qcircuit = QCircuit.from_other_language(circuit)
        if TYPE_CHECKING:
            assert isinstance(expected_output, QCircuit)
        assert matrix_eq(qcircuit.to_matrix(), expected_output.to_matrix())

    else:
        circ_to_test = circuit.to_other_language(language)
        if TYPE_CHECKING:
            assert isinstance(circ_to_test, (BraketCircuit, str))
        qcircuit = QCircuit.from_other_language(circ_to_test)
        assert matrix_eq(qcircuit.to_matrix(), circuit.to_matrix())


def test_from_other_language_qiskit_circuits(
    list_qiskit_funky_circuits: list[QiskitCircuit],
):
    for qiskit_circuit in list_qiskit_funky_circuits:
        QCircuit.from_other_language(qiskit_circuit)


def test_from_other_language_braket_circuits(
    list_braket_funky_circuits: list[BraketCircuit],
):
    for i in range(len(list_braket_funky_circuits)):
        if i == 0:
            with pytest.raises(
                ValueError,
                match="Gates not defined/handled at the time of usage: ccry, ccry",
            ):
                QCircuit.from_other_language(list_braket_funky_circuits[i])
        elif i == 6:
            with pytest.raises(
                ValueError,
                match="Gates not defined/handled at the time of usage: pswap, pswap",
            ):
                QCircuit.from_other_language(list_braket_funky_circuits[i])
        else:
            QCircuit.from_other_language(list_braket_funky_circuits[i])


def test_from_other_language_cirq_circuits(
    list_cirq_funky_circuits: list[cirq_Circuit],
):
    for circ in list_cirq_funky_circuits:
        QCircuit.from_other_language(circ)


def test_from_other_language_myqlm_circuits(
    list_myqlm_funky_circuits: list[myQLM_Circuit],
):
    for circ in list_myqlm_funky_circuits:
        QCircuit.from_other_language(circ)


@pytest.mark.parametrize(
    "circuit, expected_str",
    [
        (
            QCircuit(
                [H(i) for i in range(3)]
                + [
                    PhaseDamping(0.32, list(range(3))),
                    PhaseDamping(0.45, [0, 1]),
                ]
            ),
            "[PhaseDamping(0.45, [0]), PhaseDamping(0.32, [0]), PhaseDamping(0.45, [1]), PhaseDamping(0.32, [1]), PhaseDamping(0.32, [2])]",
        )
    ],
)
def test_from_other_language_noise(circuit: QCircuit, expected_str: str):
    braket_circuit = circuit.to_other_language(Language.BRAKET)
    qc = QCircuit.from_other_language(braket_circuit)
    assert str(qc.noises) == expected_str


@pytest.mark.parametrize(
    "circuit, printed_result_filename",
    [(QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)]), "all")],
)
def test_to_qasm_2(circuit: QCircuit, printed_result_filename: str):
    with open(
        f"tests/core/test_circuit/{printed_result_filename}.qasm2",
        "r",
        encoding="utf-8",
    ) as f:
        qasm2 = circuit.to_other_language(Language.QASM2)
        assert isinstance(qasm2, str)
        assert qasm2 == f.read()


@pytest.mark.parametrize(
    "circuit, printed_result_filename",
    [
        (QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)]), "all"),
        (
            QCircuit(
                [
                    T(0),
                    CNOT(0, 1),
                    X(0),
                    H(1),
                    Z(2),
                    CZ(2, 1),
                    SWAP(2, 0),
                    CNOT(0, 2),
                    Ry(3.14 / 2, 2),
                    S(1),
                    H(3),
                    CNOT(1, 2),
                    Rx(3.14, 1),
                    CNOT(3, 0),
                    Rz(3.14, 0),
                    BasisMeasure([0, 1, 2, 3], shots=2000),
                ]
            ),
            "lot_of_gates",
        ),
    ],
)
def test_to_qasm_3(circuit: QCircuit, printed_result_filename: str):
    with open(
        f"tests/core/test_circuit/{printed_result_filename}.qasm3",
        "r",
        encoding="utf-8",
    ) as f:
        qasm3 = circuit.to_other_language(Language.QASM3, translation_warning=False)
        assert isinstance(qasm3, str)
        assert qasm3.strip() == f.read().strip()


@pytest.mark.parametrize(
    "measure",
    [BasisMeasure(), ExpectationMeasure(Observable(1 * I @ Pauli_Z + 1 * I @ I))],
)
def test_measure_no_target(measure: Measure):
    circuit = QCircuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(measure)

    if isinstance(measure, ExpectationMeasure):
        isinstance(run(circuit, ATOSDevice.MYQLM_PYLINALG).expectation_values, float)
    else:
        assert run(circuit, ATOSDevice.MYQLM_PYLINALG).job.measure.nb_qubits == circuit.nb_qubits  # type: ignore[AttributeAccessIssue]


@pytest.mark.parametrize(
    "component",
    [
        Depolarizing(0.3),
        BitFlip(0.05),
        AmplitudeDamping(0.2),
        Barrier(),
        BasisMeasure(),
    ],
)
def test_instruction_no_target(component: Instruction | NoiseModel):
    circuit = QCircuit(2)
    circuit.add(component)

    qubits = list(range(circuit.nb_qubits))
    for instruction in circuit.instructions:
        assert qubits == instruction.targets
    for noise in circuit.noises:
        assert qubits == noise.targets

    circuit.nb_qubits += 1
    qubits = list(range(circuit.nb_qubits))
    for instruction in circuit.instructions:
        assert qubits == instruction.targets
    for noise in circuit.noises:
        assert qubits == noise.targets


@pytest.mark.parametrize(
    "circuit, expected_matrix",
    [
        (QCircuit([H(0)]), H(0).to_matrix()),
        (QCircuit([CNOT(0, 1)]), CNOT(0, 1).to_matrix()),
        (
            QCircuit([H(0), CNOT(0, 1)]),
            np.dot(
                CNOT(0, 1).to_matrix(), np.kron(H(0).to_matrix(), Id(0).to_matrix())
            ),
        ),
        (QCircuit([Id(0)]), Id(0).to_matrix()),
        (QCircuit([SWAP(0, 1)]), SWAP(0, 1).to_matrix()),
        (QCircuit([]), np.array([[1]])),
        (
            QCircuit([Rz(np.pi / 4, 0)]),
            Rz(np.pi / 4, 0).to_matrix(),
        ),
        (
            QCircuit([Ry(np.pi / 2, 0)]),
            Ry(np.pi / 2, 0).to_matrix(),
        ),
        (
            QCircuit([TOF([0, 1], 2)]),
            TOF([0, 1], 2).to_matrix(),
        ),
        (
            QCircuit([TOF([0, 2], 1)]),
            TOF([0, 2], 1).to_matrix(),
        ),
    ],
)
def test_to_matrix(circuit: QCircuit, expected_matrix: Matrix):
    matrix_eq(circuit.to_matrix(), expected_matrix)


def test_to_matrix_random():
    gates = [
        gate for gate in native_gates.NATIVE_GATES if issubclass(gate, SingleQubitGate)
    ]
    for _ in range(10):
        qcircuit = random_circuit(gates, nb_qubits=4)
        expected_matrix = compute_expected_matrix(qcircuit)
        matrix_eq(qcircuit.to_matrix(), expected_matrix)


def test_to_matrix_gphase():
    gates = [
        gate for gate in native_gates.NATIVE_GATES if issubclass(gate, SingleQubitGate)
    ]
    for _ in range(10):
        qcircuit = random_circuit(gates, nb_qubits=4)
        qcircuit.gphase = random.random()
        expected_matrix = compute_expected_matrix(qcircuit)
        assert matrix_eq(
            qcircuit.to_matrix(), expected_matrix * np.exp(1j * qcircuit.gphase)
        )


@pytest.mark.parametrize(
    "circuit, expected_inverse",
    [
        (
            QCircuit([H(0), CNOT(0, 1)]),
            QCircuit([CNOT(0, 1), H(0)]),
        ),
        (
            QCircuit([S(0), CZ(0, 1), H(1), Ry(4.56, 1)]),
            QCircuit(
                [
                    Ry(4.56, 1),
                    H(1),
                    CZ(0, 1),
                    S(0),
                ]
            ),
        ),
        (
            QCircuit([S(0), CZ(0, 1), H(1), BasisMeasure([0, 1, 2, 3], shots=2000)]),
            QCircuit(
                [
                    H(1),
                    CZ(0, 1),
                    S(0),
                    BasisMeasure([0, 1, 2, 3], shots=2000),
                ]
            ),
        ),
        (
            QCircuit([S(0), CRk(2, 1, 2), Barrier(), H(1), Ry(4.56, 1)]),
            QCircuit(
                [
                    Ry(4.56, 1),
                    H(1),
                    Barrier(),
                    CRk(2, 1, 2),
                    S(0),
                ]
            ),
        ),
    ],
)
def test_inverse(circuit: QCircuit, expected_inverse: QCircuit):
    if any(not isinstance(inst, (Gate, Barrier)) for inst in circuit.instructions):
        with pytest.warns(NonReversibleWarning):
            inverse_circuit = circuit.inverse()
    else:
        inverse_circuit = circuit.inverse()
    for inverse_inst, expected_inst in zip(
        inverse_circuit.instructions, expected_inverse.instructions
    ):
        if isinstance(expected_inst, Gate) and isinstance(inverse_inst, Gate):
            assert matrix_eq(
                expected_inst.to_matrix().transpose().conjugate(),
                inverse_inst.to_matrix(),
            ), f"Expected {repr(expected_inst)}, but got {repr(inverse_inst)}"
        else:
            assert (
                expected_inst == inverse_inst
            ), f"Expected {repr(expected_inst)}, but got {repr(inverse_inst)}"


def test_inverse_random():
    for _ in range(10):
        qcircuit = random_circuit(nb_qubits=4)
        inverse_circuit = qcircuit.inverse()
        for inverse_inst, expected_inst in zip(
            inverse_circuit.instructions, reversed(qcircuit.instructions)
        ):
            if isinstance(expected_inst, Gate) and isinstance(inverse_inst, Gate):
                assert matrix_eq(
                    expected_inst.to_matrix().transpose().conjugate(),
                    inverse_inst.to_matrix(),
                ), f"Expected {repr(expected_inst)}, but got {repr(inverse_inst)}"
            else:
                assert expected_inst == inverse_inst


@pytest.mark.parametrize(
    "circuit, expected_qubits",
    [
        (QCircuit(), 0),
        (QCircuit([H(0)]), 1),
        (QCircuit([H(1)]), 2),
        (QCircuit([S(0), CZ(0, 2), H(1), Ry(4.56, 1)]), 3),
        (QCircuit([S(0), CZ(0, 1), H(1), BasisMeasure([0, 1, 2, 3], shots=2000)]), 4),
        (
            QCircuit(
                [S(0), CRk(2, 1, 2), Barrier(), H(1), Ry(4.56, 1), BasisMeasure()]
            ),
            3,
        ),
    ],
)
def test_qubits_dynamic_circuit(circuit: QCircuit, expected_qubits: int):
    assert circuit.nb_qubits == expected_qubits
    circuit.add(H(expected_qubits))
    assert circuit.nb_qubits == expected_qubits + 1


@pytest.mark.parametrize(
    "circuit, expected_qubits",
    [
        (QCircuit(0), 0),
        (QCircuit([H(0)], nb_qubits=1), 1),
        (QCircuit([H(1)], nb_qubits=2), 2),
        (QCircuit([S(0), CZ(0, 2), H(1), Ry(4.56, 1)], nb_qubits=3), 3),
        (
            QCircuit(
                [S(0), CZ(0, 1), H(1), BasisMeasure([0, 1, 2, 3], shots=2000)],
                nb_qubits=4,
            ),
            4,
        ),
        (
            QCircuit(
                [S(0), CRk(2, 1, 2), Barrier(), H(1), Ry(4.56, 1), BasisMeasure()],
                nb_qubits=3,
            ),
            3,
        ),
    ],
)
def test_qubits_not_dynamic_circuit(circuit: QCircuit, expected_qubits: int):
    assert circuit.nb_qubits == expected_qubits
    with pytest.raises(NumberQubitsError):
        circuit.add(H(expected_qubits))


@pytest.mark.parametrize(
    "circuit, expected_qubits",
    [
        (QCircuit(), 0),
        (QCircuit([H(1)]), 2),
        (QCircuit([S(0), CZ(0, 2), H(1), Ry(4.56, 1)]), 3),
        (QCircuit([S(0), CZ(0, 1), H(1), BasisMeasure([0, 1, 2, 3], shots=2000)]), 4),
        (
            QCircuit(
                [S(0), CRk(2, 1, 2), Barrier(), H(1), Ry(4.56, 1), BasisMeasure()]
            ),
            3,
        ),
    ],
)
def test_qubits_dynamic_to_not_dynamic_circuit(circuit: QCircuit, expected_qubits: int):
    assert circuit.nb_qubits == expected_qubits
    circuit.nb_qubits = expected_qubits
    with pytest.raises(NumberQubitsError):
        circuit.add(H(expected_qubits))


@pytest.mark.parametrize(
    "circuit, expected_cbits",
    [
        (QCircuit(), 0),
        (QCircuit([BasisMeasure([0], [0], shots=2000)]), 1),
        (QCircuit([BasisMeasure([0, 1], [0, 1], shots=2000)]), 2),
        (
            QCircuit(
                [
                    BasisMeasure([0], [0], shots=2000),
                    BasisMeasure([1, 2], [1, 2], shots=2000),
                ]
            ),
            3,
        ),
        (
            QCircuit(
                [
                    BasisMeasure([0], [0], shots=2000),
                    BasisMeasure([0, 1], [0, 1], shots=2000),
                ]
            ),
            2,
        ),
        (
            QCircuit([BasisMeasure([0, 1, 2, 3], [0, 1, 2, 3], shots=2000)]),
            4,
        ),
        (
            QCircuit([BasisMeasure()], nb_qubits=3),
            3,
        ),
    ],
)
def test_cbits_dynamic_circuit(circuit: QCircuit, expected_cbits: int):
    assert circuit.nb_qubits == expected_cbits
    circuit.add(BasisMeasure([0], [expected_cbits]))
    assert circuit.nb_cbits == expected_cbits + 1


@pytest.mark.parametrize(
    "circuit, expected_cbits",
    [
        (QCircuit(nb_cbits=0), 0),
        (QCircuit([BasisMeasure([0], [0], shots=2000)], nb_cbits=1), 1),
        (QCircuit([BasisMeasure([0, 1], [0, 1], shots=2000)], nb_cbits=2), 2),
        (
            QCircuit(
                [
                    BasisMeasure([0], [0], shots=2000),
                    BasisMeasure([1, 2], [1, 2], shots=2000),
                ],
                nb_cbits=3,
            ),
            3,
        ),
        (
            QCircuit(
                [
                    BasisMeasure([0], [0], shots=2000),
                    BasisMeasure([0, 1], [0, 1], shots=2000),
                ],
                nb_cbits=2,
            ),
            2,
        ),
        (
            QCircuit(
                [BasisMeasure([0, 1, 2, 3], [0, 1, 2, 3], shots=2000)],
                nb_cbits=4,
            ),
            4,
        ),
        (
            QCircuit(
                [BasisMeasure()],
                nb_cbits=3,
            ),
            3,
        ),
    ],
)
def test_cbits_undersized_static_circuit(circuit: QCircuit, expected_cbits: int):
    assert circuit.nb_cbits == expected_cbits
    with pytest.raises(ValueError):
        circuit.add(BasisMeasure([expected_cbits], [expected_cbits]))


@pytest.mark.parametrize(
    "circuit, expected_cbits",
    [
        (QCircuit(), 0),
        (QCircuit([BasisMeasure([0], [0], shots=2000)]), 1),
        (QCircuit([BasisMeasure([0, 1], [0, 1], shots=2000)]), 2),
        (
            QCircuit(
                [
                    BasisMeasure([0], [0], shots=2000),
                    BasisMeasure([1, 2], [1, 2], shots=2000),
                ]
            ),
            3,
        ),
        (
            QCircuit(
                [
                    BasisMeasure([0], [0], shots=2000),
                    BasisMeasure([0, 1], [0, 1], shots=2000),
                ]
            ),
            2,
        ),
        (
            QCircuit([BasisMeasure([0, 1, 2, 3], [0, 1, 2, 3], shots=2000)]),
            4,
        ),
        (
            QCircuit([BasisMeasure()], nb_qubits=3),
            3,
        ),
    ],
)
def test_cbits_dynamic_toggled_off_undersized_circuit(
    circuit: QCircuit, expected_cbits: int
):
    assert circuit.nb_qubits == expected_cbits
    circuit.nb_qubits = expected_cbits
    with pytest.raises(ValueError):
        circuit.add(BasisMeasure([expected_cbits], [expected_cbits]))
