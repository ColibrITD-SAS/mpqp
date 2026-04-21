import numpy as np
import pytest

from mpqp.core.circuit import QCircuit
from mpqp.core.languages import Language
from mpqp.gates import *
from mpqp.tools.maths import matrix_eq

COMPOSED_GATES = [Rzz(np.pi / 2, 0, 1), PRX(np.pi / 3, 1, 0)]


@pytest.mark.parametrize(
    "gate, gate_set",
    [
        (Rzz(np.pi / 2, 0, 1), {}),
        (Rzz(np.pi / 2, 0, 1), {Rzz}),
        (Rzz(np.pi / 2, 0, 1), {Rz, CNOT}),
        (PRX(np.pi / 3, 1, 0), {}),
        (PRX(np.pi / 3, 1, 0), {PRX}),
        (PRX(np.pi / 3, 1, 0), {Rx, Rz}),
    ],
)
def test_composedgate_compatible(gate: Gate, gate_set: set[type[Gate]]) -> None:
    QCircuit([gate]).to_other_language(Language.QISKIT, authorized_gates=gate_set)


@pytest.mark.parametrize(
    "gate, gate_set",
    [
        (Rzz(np.pi / 2, 0, 1), {CNOT}),
        (Rzz(np.pi / 2, 0, 1), {Rz}),
        (PRX(np.pi / 3, 1, 0), {Rx}),
        (PRX(np.pi / 3, 1, 0), {Rz}),
    ],
)
def test_composedgate_notcompatible(gate: Gate, gate_set: set[type[Gate]]) -> None:
    with pytest.raises(ValueError):
        QCircuit([gate]).to_other_language(Language.QISKIT, authorized_gates=gate_set)


def define_parameters(decomposition: bool):
    result = []
    providers = [
        Language.QISKIT,
        Language.BRAKET,
        Language.CIRQ,
        Language.MY_QLM,
        Language.QASM2,
        Language.QASM3,
    ]
    for gate in COMPOSED_GATES:
        for provider in providers:
            if decomposition:
                result.append((gate, provider, {}))
            else:
                result.append((gate, provider, {type(gate)}))
    return result


@pytest.mark.parametrize(
    "gate, language, authorized_gates",
    define_parameters(False),
)
def test_composedgate_translation_no_decomposition(
    gate: Gate, language: Language, authorized_gates: set[type[Gate]]
):
    c = QCircuit()
    c.add(gate)
    translated = c.to_other_language(language, authorized_gates=authorized_gates)
    c_re = QCircuit().from_other_language(translated)
    assert matrix_eq(c_re.to_matrix(), c.to_matrix())


@pytest.mark.parametrize(
    "gate, language, authorized_gates",
    define_parameters(True),
)
def test_composedgate_translation_decomposition(
    gate: Gate, language: Language, authorized_gates: set[type[Gate]]
):
    c = QCircuit()
    c.add(gate)
    translated = c.to_other_language(language, authorized_gates=authorized_gates)
    c_re = QCircuit().from_other_language(translated)
    assert matrix_eq(c_re.to_matrix(), c.to_matrix())


@pytest.mark.parametrize(
    "gate",
    [
        (Rzz(np.pi / 2, 0, 1)),
        (PRX(np.pi / 3, 1, 0)),
    ],
)
def test_composedgates_decomposition(gate: ComposedGate):
    c = QCircuit(gate.decompose())
    assert matrix_eq(c.to_matrix(), gate.to_matrix())
