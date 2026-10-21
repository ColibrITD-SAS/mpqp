"""Provider translations preserve expressions instead of renaming parameters."""

import numpy as np
import pytest
from sympy import symbols

from mpqp import QCircuit
from mpqp.core.languages import Language
from mpqp.gates import Ry


@pytest.mark.provider("qiskit")
def test_qiskit_expression_preserves_symbols():
    from qiskit.quantum_info import Statevector

    theta, phi = symbols("theta phi")
    translated = QCircuit([Ry(2 * theta + phi, 0)]).to_other_language(Language.QISKIT)
    assert {parameter.name for parameter in translated.parameters} == {"theta", "phi"}
    values = {"theta": 0.2, "phi": 0.1}
    bound = translated.assign_parameters(
        {parameter: values[parameter.name] for parameter in translated.parameters}
    )
    assert Statevector(bound).probabilities()[1] == pytest.approx(np.sin(0.25) ** 2)
    assert len(translated.parameters) == 2


@pytest.mark.provider("braket")
def test_braket_expression_preserves_symbols():
    theta, phi = symbols("theta phi")
    translated = QCircuit([Ry(2 * theta + phi, 0)]).to_other_language(Language.BRAKET)
    assert {str(parameter) for parameter in translated.parameters} == {"theta", "phi"}
    bound = translated.make_bound_circuit({"theta": 0.2, "phi": 0.1})
    assert float(bound.instructions[0].operator.angle) == pytest.approx(0.5)
    assert len(translated.parameters) == 2
