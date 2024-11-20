from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pytest
from braket.circuits import Circuit as BraketCircuit
from qiskit import QuantumCircuit as QiskitCircuit
from typeguard import TypeCheckError

from mpqp import Barrier, Instruction, Language, QCircuit
from mpqp.core.instruction.gates import native_gates
from mpqp.core.instruction.gates.gate import SingleQubitGate
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.core.instruction.measurement.pauli_string import I
from mpqp.core.instruction.measurement.pauli_string import Z as Pauli_Z
from mpqp.execution.devices import ATOSDevice
from mpqp.execution.runner import run
from mpqp.gates import CNOT, CZ, SWAP, TOF, CRk, Gate, H, Id, Rx, Ry, Rz, S, T, X, Y, Z
from mpqp.measures import BasisMeasure, ExpectationMeasure, Observable
from mpqp.noise.noise_model import AmplitudeDamping, BitFlip, Depolarizing, NoiseModel
from mpqp.tools.circuit import compute_expected_matrix, random_circuit
from mpqp.tools.display import one_lined_repr
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning, NonReversibleWarning
from mpqp.tools.generics import Matrix, OneOrMany
from mpqp.tools.maths import matrix_eq


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
                        Observable(np.identity(2, dtype=np.complex64)), [1], shots=1000
                    ),
                ]
            ),
            "[BasisMeasure([0, 1], shots=1000), ExpectationMeasure("
            "Observable(array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]], dtype=complex64)), [1], shots=1000)]",
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
        isinstance(run(circuit, ATOSDevice.MYQLM_PYLINALG).expectation_value, float)  # type: ignore[AttributeAccessIssue]
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
