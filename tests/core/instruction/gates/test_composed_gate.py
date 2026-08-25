import numpy as np
import pytest

from mpqp.core.circuit import QCircuit
from mpqp.core.languages import Language
from mpqp.gates import *
from mpqp.tools.maths import matrix_eq

COMPOSED_GATES = [
    Rxx(np.pi / 2, 0, 1),
    Rzz(np.pi / 2, 0, 1),
    Ryy(np.pi / 2, 0, 1),
    PRX(np.pi / 3, 1, 0),
]


@pytest.mark.parametrize(
    "gate",
    [
        (Rxx(np.pi / 2, 0, 1)),
        (Rxx(np.pi / 2, 0, 1)),
        (Rxx(np.pi / 2, 0, 1)),
        (Ryy(np.pi / 2, 0, 1)),
        (Ryy(np.pi / 2, 0, 1)),
        (Ryy(np.pi / 2, 0, 1)),
        (Rzz(np.pi / 2, 0, 1)),
        (Rzz(np.pi / 2, 0, 1)),
        (Rzz(np.pi / 2, 0, 1)),
        (PRX(np.pi / 3, 1, 0)),
        (PRX(np.pi / 3, 1, 0)),
        (PRX(np.pi / 3, 1, 0)),
    ],
)
def test_composedgate_compatible(gate: Gate) -> None:
    QCircuit([gate]).to_other_language(Language.QISKIT)


@pytest.mark.parametrize(
    "language, gate_set_getter, gate, native_gates",
    [
        (
            Language.QISKIT,
            "get_qiskit_gate_set",
            Rxx(np.pi / 2, 0, 1),
            {Rx},
        ),
        (
            Language.QISKIT,
            "get_qiskit_gate_set",
            Ryy(np.pi / 2, 0, 1),
            {Rx, Rz},
        ),
        (
            Language.BRAKET,
            "get_braket_gate_set",
            Rzz(np.pi / 2, 0, 1),
            {Rz},
        ),
        (
            Language.CIRQ,
            "get_cirq_gate_set",
            PRX(np.pi / 3, 1, 0),
            {Rx},
        ),
    ],
)
def test_composedgate_not_compatible_with_provider(
    monkeypatch: pytest.MonkeyPatch,
    language: Language,
    gate_set_getter: str,
    gate: Gate,
    native_gates: set[type[Gate]],
) -> None:
    monkeypatch.setattr(
        f"mpqp.tools.circuit.{gate_set_getter}",
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
