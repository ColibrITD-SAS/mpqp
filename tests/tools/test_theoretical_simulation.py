import numpy as np

from mpqp import CNOT, BasisMeasure, H, QCircuit
from mpqp.tools.theoretical_simulation import theoretical_probs


def test_simulation():
    circuit = QCircuit(
        [
            H(0),
            CNOT(0, 1),
            BasisMeasure([0, 1], shots=1024),
        ],
        label="Noise-Testing",
    )

    assert np.allclose(np.array([0.5, 0, 0, 0.5]), theoretical_probs(circuit))
