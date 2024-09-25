from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import jensenshannon

from mpqp.all import *


def theoretical_probs(
    circ: QCircuit,
    p: float,
) -> npt.NDArray[np.float32]:
    d: int = 2 ** (circ.nb_qubits)

    state = np.zeros(d, dtype=np.complex64)
    state[0] = 1

    for gate in circ.instructions:
        if isinstance(gate, Gate):
            state @= gate.to_matrix(circ.nb_qubits).astype(np.complex64).T

    noisy_density_matrix = (1 - p) * np.outer(state, np.conj(state)) + p * np.eye(d) / d

    return noisy_density_matrix.diagonal().astype(np.float32)


def validate(
    mpqp_counts: list[int],
    theoretical_probabilities: npt.NDArray[np.float32],
    alpha: float = 0.1,
) -> bool:

    dist = jensenshannon(mpqp_counts, theoretical_probabilities * sum(mpqp_counts))

    return float(dist) < alpha
