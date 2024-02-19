from __future__ import annotations
from typing import Optional, Sequence

import pytest
import numpy as np
from qiskit import QuantumCircuit
from typeguard import TypeCheckError

from mpqp import QCircuit, Instruction, Barrier, Language
from mpqp.gates import Gate, CNOT, X, Y, Z, CZ, SWAP, T, H, Rx, S, Ry, Rz
from mpqp.measures import BasisMeasure, ExpectationMeasure, Observable
from mpqp.tools.generics import ListOrSingle, one_lined_repr

# 3M-TODO: a lot of these tests use str to test circuit equivalence, it would be
# preferable to define a __eq__ in the QCircuit class, or at least an
# `equivalent` method


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
    added_gates: ListOrSingle[Instruction],
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
                        [1], Observable(np.identity(2, dtype=np.complex64)), shots=1000
                    ),
                ]
            ),
            "[BasisMeasure([0, 1], shots=1000), ExpectationMeasure([1], "
            "Observable(array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]], dtype=complex64)), shots=1000)]",
        )
    ],
)
def test_get_measurements(circuit: QCircuit, result_repr: str):
    assert one_lined_repr(circuit.get_measurements()) == result_repr


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
            QuantumCircuit,
            (
                "[CircuitInstruction(operation=Instruction(name='x', num_qubits=1,"
                " num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),),"
                " clbits=()), CircuitInstruction(operation=Instruction(name='cx',"
                " num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2,"
                " 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=())]"
            ),
        ),
    ],
)
def test_to_other_language(
    circuit: QCircuit, args: tuple[Language], result_type: type, result_repr: str
):
    qiskit_circuit = circuit.to_other_language(*args)
    assert type(qiskit_circuit) == QuantumCircuit
    assert repr(qiskit_circuit.data) == result_repr


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
        assert circuit.to_qasm2() == f.read()


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
        assert circuit.to_qasm3().strip() == f.read().strip()
