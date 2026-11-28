import numpy as np
import pytest
from sympy import Symbol

from mpqp import (
    PRX,
    BasisMeasure,
    CircuitBinding,
    CRk,
    Gate,
    H,
    QCircuit,
    QUANTINUUMDevice,
    Rk,
    Rxx,
    Ry,
    Rz,
    U,
    X,
)
from mpqp.core._types import BindingParameters
from mpqp.tools.maths import matrix_eq

pytestmark = pytest.mark.provider("quantinuum")


def test_mpqp_to_tket_preserves_symbolic_expressions_after_binding():
    from mpqp.translation import mpqp_to_tket
    from mpqp.translation.tket import bind_tket_parameters

    theta = Symbol("theta")
    phi = Symbol("phi")
    circuit = QCircuit([Ry(2 * theta + phi, 0)])
    translated = mpqp_to_tket(circuit)

    assert {str(symbol) for symbol in translated.free_symbols()} == {
        "phi",
        "theta",
    }
    assert circuit.variables() == {phi, theta}

    bound = bind_tket_parameters(
        translated,
        {theta: np.pi / 4, phi: 0.0},
    )

    assert bound.free_symbols() == set()
    assert float(bound.get_commands()[0].op.params[0]) == pytest.approx(0.5)
    assert {str(symbol) for symbol in translated.free_symbols()} == {
        "phi",
        "theta",
    }


@pytest.mark.parametrize(
    "circuit",
    [
        QCircuit([Rk(Symbol("k"), 0)]),
        QCircuit([CRk(Symbol("k"), 0, 1)]),
    ],
    ids=["Rk", "CRk"],
)
def test_tket_translation_preserves_symbolic_power_expressions(circuit: QCircuit):
    from mpqp.translation.tket import bind_tket_parameters, mpqp_to_tket

    translated = mpqp_to_tket(circuit)

    assert {str(symbol) for symbol in translated.free_symbols()} == {"k"}
    assert circuit.variables() == {Symbol("k")}

    bound = bind_tket_parameters(translated, {"k": 2})

    assert bound.free_symbols() == set()
    assert float(bound.get_commands()[0].op.params[0]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "symbolic_gate, values, expected_gate",
    [
        (
            U(Symbol("theta"), Symbol("phi"), Symbol("gamma"), 0),
            {"theta": 0.2, "phi": -0.3, "gamma": 0.7},
            U(0.2, -0.3, 0.7, 0),
        ),
        (
            PRX(Symbol("theta"), Symbol("phi"), 0),
            {"theta": 0.2, "phi": -0.3},
            PRX(0.2, -0.3, 0),
        ),
        (
            CRk(Symbol("k"), 0, 1),
            {"k": 3.0},
            CRk(3, 0, 1),
        ),
        (
            Rxx(Symbol("theta"), 0, 1),
            {"theta": 0.2},
            Rxx(0.2, 0, 1),
        ),
    ],
    ids=["U", "PRX", "CRk", "Rxx"],
)
def test_symbolic_gate_matrix_after_tket_translation(
    symbolic_gate: Gate,
    values: BindingParameters,
    expected_gate: Gate,
):
    from mpqp.translation.tket import bind_tket_parameters, mpqp_to_tket

    translated = mpqp_to_tket(QCircuit([symbolic_gate]))
    bound = bind_tket_parameters(translated, values)

    expected_matrix = QCircuit([expected_gate]).to_matrix()

    assert matrix_eq(bound.get_unitary(), expected_matrix)


def test_bind_tket_parameters_rejects_missing_values():
    from mpqp.translation.tket import bind_tket_parameters, mpqp_to_tket

    theta = Symbol("theta")
    translated = mpqp_to_tket(QCircuit([Ry(theta, 0)]))

    with pytest.raises(ValueError, match="Missing values.*theta"):
        bind_tket_parameters(translated, None)


def test_circuit_binding_to_tket_reuses_circuit_for_parameter_values():
    from mpqp.translation.tket import circuitbinding_to_tket

    theta = Symbol("theta")
    binding = CircuitBinding(
        QCircuit([Ry(theta, 0)]),
        values=[{theta: 0.0}, {theta: np.pi}],
        measurements=BasisMeasure(shots=10),
    )

    translated = circuitbinding_to_tket(
        binding,
        QUANTINUUMDevice.TKET_AER_SIMULATOR,
    )

    assert len(translated) == 1
    symbolic_circuit, contexts = translated[0]
    assert {str(symbol) for symbol in symbolic_circuit.free_symbols()} == {"theta"}
    assert [context.index for context in contexts] == [0, 1]
    assert [context.job.values for context in contexts] == [
        {theta: 0.0},
        {theta: np.pi},
    ]


def test_mpqp_to_tket_preserves_gate_order_with_symbolic_parameters():
    from mpqp.translation.tket import bind_tket_parameters, mpqp_to_tket

    theta = Symbol("theta")
    translated = mpqp_to_tket(QCircuit([H(0), Ry(theta, 0), X(0), Rz(np.pi / 2, 0)]))

    assert [command.op.type.name for command in translated.get_commands()] == [
        "H",
        "Ry",
        "X",
        "Rz",
    ]
    bound = bind_tket_parameters(translated, {theta: np.pi})
    assert bound.free_symbols() == set()
