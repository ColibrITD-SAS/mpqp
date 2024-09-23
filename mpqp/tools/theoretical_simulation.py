from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import chisquare

from mpqp.all import *
from mpqp.tools.generics import Matrix

noise_proba = 0.7
shots = 1024

ElementalGate = tuple[Optional[str], list[int]]
GatesMatrix = dict[Optional[str], Matrix]


def run_experiment(
    circ: QCircuit,
    p: float,
    shots: int,
) -> dict[int, int]:
    d: int = 2 ** (circ.nb_qubits)

    gates = [i for i in circ.instructions if isinstance(i, Gate)]
    state = np.zeros(d, dtype=np.complex64)
    state[0] = 1

    for gate in gates:
        state @= gate.to_matrix(circ.nb_qubits)

    state_density_matrix: npt.NDArray[np.complex64] = np.outer(state, np.conj(state))
    noisy_density_matrix = (1 - p) * state_density_matrix + p * np.eye(d) / d
    probs = np.abs(noisy_density_matrix.flatten()) ** 2

    counter = Counter(
        [int(np.random.choice(len(probs), p=probs)) for _ in range(shots)]
    )
    return {state: counter[state] for state in range(2**circ.nb_qubits)}


def print_results(results_dict: dict[str, int], num_qubits: int, shots: int):
    counts = [results_dict.get(f"{i:0{num_qubits}b}", 0) for i in range(2**num_qubits)]
    probabilities = [count / shots for count in counts]

    print("Result: Theoretical - Without SDK")
    print("Counts:", counts)
    print("Probabilities:", probabilities)
    print("Samples:")
    for i, (count, probability) in enumerate(zip(counts, probabilities)):
        state = f"{i:0{num_qubits}b}"
        print(
            f"  State: |{state}>, Index: {i}, Count: {count}, Probability: {probability:.8f}"
        )


def plot_results(results: dict[int, int], num_qubits: int):
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    states, counts = zip(*sorted_results)

    x = np.arange(len(states))  # plot the counts for each state
    states = [f"|{state:0{num_qubits}b}⟩" for state in states]

    plt.bar(x, counts, align="center", width=0.6)
    plt.xticks(x, states)
    plt.xlabel("States")
    plt.ylabel("Counts")
    plt.title("Results without SDK")
    plt.show()


def chi_square_test(
    mpqp_counts: list[int],
    theoretical_counts: list[int],
    shots: int,
    alpha: float = 0.05,
) -> ChiSquareResult:

    if not mpqp_counts or not theoretical_counts:
        return ChiSquareResult(
            expected_counts=[],
            chisquare_stat=np.nan,
            p_value=1.0,
            significant=False,
        )

    theoretical_probabilities = [count / shots for count in theoretical_counts]
    expected_counts = [int(tp * shots) for tp in theoretical_probabilities]

    chisquare_stat, p_value = chisquare(mpqp_counts, expected_counts)

    return ChiSquareResult(
        expected_counts=expected_counts,
        chisquare_stat=chisquare_stat,
        p_value=p_value,
        significant=p_value < alpha,
    )


@dataclass
class ChiSquareResult:
    expected_counts: list[int]
    chisquare_stat: Any
    p_value: float
    significant: bool
