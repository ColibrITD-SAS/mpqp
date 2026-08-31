import numpy as np
import pytest

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.gate_decomposition import (
    resolve_composed_gate,
    resolve_gate,
)
from mpqp.core.languages import Language
from mpqp.gates import *
from mpqp.tools.errors import UnsupportedGateError
from mpqp.tools.maths import matrix_eq

COMPOSED_GATES = [
    Rxx(np.pi / 2, 0, 1),
    Rzz(np.pi / 2, 0, 1),
    Ryy(np.pi / 2, 0, 1),
    PRX(np.pi / 3, 1, 0),
]


@pytest.mark.parametrize(
    "gate ,language",
    [
        (Rxx(np.pi / 2, 0, 1), Language.QISKIT),
        (Ryy(np.pi / 2, 0, 1), Language.QISKIT),
        (Rzz(np.pi / 2, 0, 1), Language.BRAKET),
        (PRX(np.pi / 3, 1, 0), Language.BRAKET),
    ],
)
def test_composedgate_compatible(gate: Gate, language: Language) -> None:
    translated = QCircuit([gate]).to_other_language(language)
    assert translated is not None


@pytest.mark.parametrize(
    "gate,gate_set,expected",
    [
        (PRX(1.0, 0.5, 0), {Rx, Rz}, [Rz, Rx, Rz]),
        (Rzz(1.0, 0, 1), {CNOT, Rz}, [CNOT, Rz, CNOT]),
    ],
)
def test_composed_gate_is_decomposed(
    gate: Gate,
    gate_set: set[type[Gate]],
    expected: list[type[Gate]],
) -> None:
    resolved = resolve_gate(gate, gate_set)

    assert [type(item) for item in resolved] == expected


def test_braket_translation_does_not_pad_sparse_circuit():
    circuit = QCircuit([H(1), CNOT(1, 3)], nb_qubits=4)

    translated = circuit.to_other_language(Language.BRAKET)

    assert all(
        instruction.operator.name != "I" for instruction in translated.instructions
    )


@pytest.mark.parametrize(
    "language, provider, gate_set_getter, gate, native_gates",
    [
        (
            Language.QISKIT,
            "qiskit",
            "get_qiskit_gate_set",
            Rxx(np.pi / 2, 0, 1),
            {Rx},
        ),
        (
            Language.QISKIT,
            "qiskit",
            "get_qiskit_gate_set",
            Ryy(np.pi / 2, 0, 1),
            {Rx, Rz},
        ),
        (
            Language.BRAKET,
            "braket",
            "get_braket_gate_set",
            Rzz(np.pi / 2, 0, 1),
            {Rz},
        ),
        (
            Language.CIRQ,
            "cirq",
            "get_cirq_gate_set",
            PRX(np.pi / 3, 1, 0),
            {Rx},
        ),
    ],
)
def test_composedgate_not_compatible_with_provider(
    monkeypatch: pytest.MonkeyPatch,
    language: Language,
    provider: str,
    gate_set_getter: str,
    gate: Gate,
    native_gates: set[type[Gate]],
) -> None:
    monkeypatch.setattr(
        f"mpqp.translation.{provider}.{gate_set_getter}",
        lambda: native_gates,
    )

    with pytest.raises(ValueError):
        QCircuit([gate]).to_other_language(language)


def define_parameters():
    return [
        (gate, language)
        for gate in COMPOSED_GATES
        for language in [
            Language.QISKIT,
            Language.BRAKET,
            Language.CIRQ,
            Language.MY_QLM,
            Language.QASM2,
            Language.QASM3,
        ]
    ]


@pytest.mark.parametrize(
    "gate, language",
    define_parameters(),
)
def test_composedgate_translation_no_decomposition(gate: Gate, language: Language):
    c = QCircuit()
    c.add(gate)
    translated = c.to_other_language(language)
    c_re = QCircuit().from_other_language(translated)
    assert matrix_eq(c_re.to_matrix(), c.to_matrix())


@pytest.mark.parametrize(
    "gate, language",
    define_parameters(),
)
def test_composedgate_translation_decomposition(gate: Gate, language: Language):
    c = QCircuit()
    c.add(gate)
    translated = c.to_other_language(language)
    c_re = QCircuit().from_other_language(translated)
    assert matrix_eq(c_re.to_matrix(), c.to_matrix(), 1e-5, 1e-5)


@pytest.mark.parametrize(
    "gate",
    COMPOSED_GATES,
)
def test_composedgates_decomposition(gate: ComposedGate):
    c = QCircuit(gate.decompose())
    assert matrix_eq(c.to_matrix(), gate.to_matrix())


@pytest.mark.parametrize("gate", [Rxx(np.pi / 2, 0, 1)])
def test_composedgates_error(gate: Gate):
    with pytest.raises(
        UnsupportedGateError,
        match=r"Rxx cannot be represented.*Missing gates: CNOT",
    ):
        resolve_composed_gate(gate, {Rx})
