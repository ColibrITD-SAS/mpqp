from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chisquare
from scipy.stats._stats_py import Power_divergenceResult

from mpqp.all import *


def run_experiment_probs(
    circ: QCircuit,
    p: float,
    shots: int,
) -> npt.NDArray[np.int32]:
    d: int = 2 ** (circ.nb_qubits)

    gates = [i for i in circ.instructions if isinstance(i, Gate)]
    state = np.zeros(d, dtype=np.complex64)
    state[0] = 1

    for gate in gates:
        state @= gate.to_matrix(circ.nb_qubits)

    state_density_matrix: npt.NDArray[np.complex64] = np.outer(state, np.conj(state))
    noisy_density_matrix = (1 - p) * state_density_matrix + p * np.eye(d) / d
    return np.abs(noisy_density_matrix.flatten()) ** 2


def run_experiment(
    circ: QCircuit,
    p: float,
    shots: int,
) -> dict[int, int]:
    probs = run_experiment_probs(circ, p, shots)
    print(probs)
    counter = Counter(
        [int(np.random.choice(len(probs), p=probs)) for _ in range(shots)]
    )
    return {state: counter[state] for state in range(2**circ.nb_qubits)}


def chi_square_test(
    mpqp_counts: list[int],
    theoretical_counts: list[int],
    shots: int,
    alpha: float = 0.05,
) -> ChiSquareResult:

    if not mpqp_counts or not theoretical_counts:
        return ChiSquareResult(
            expected_counts=[],
            chi_square_stat=np.nan,
            p_value=1.0,
            significant=False,
        )

    theoretical_probabilities = [count / shots for count in theoretical_counts]
    expected_counts = [int(tp * shots) for tp in theoretical_probabilities]

    chi_square_stat, p_value = chisquare(mpqp_counts, expected_counts)

    return ChiSquareResult(
        expected_counts=expected_counts,
        chi_square_stat=chi_square_stat,
        p_value=p_value,
        significant=p_value < alpha,
    )


@dataclass
class ChiSquareResult:
    expected_counts: list[int]
    chi_square_stat: Power_divergenceResult | float
    p_value: float
    significant: bool
