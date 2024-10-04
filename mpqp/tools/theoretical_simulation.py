from __future__ import annotations

from functools import reduce
from itertools import product
from math import log2

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import jensenshannon

from mpqp import QCircuit
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.execution import AWSDevice
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.gates import Gate
from mpqp.noise import Depolarizing


def apply_global_dephasing_noise(state: npt.NDArray[np.complex64], p: float, d: int):
    return (1 - p) * state + p * np.eye(d) / d


def apply_global_bitflip_noise(state: npt.NDArray[np.complex64], p: float, d: int):
    n = int(log2(d))
    I = np.eye(2)
    X = np.ones(2) - I
    all_combinations = product([I, X], repeat=n)
    I_count_and_flip = [
        (comb.count(I), reduce(np.kron, comb)) for comb in all_combinations
    ]
    return sum(
        (
            (1 - p) ** count * p ** (n - count) * flip @ state @ flip
            for count, flip in I_count_and_flip
        ),
        start=np.zeros((d, d)),
    )


def theoretical_probs(
    circ: QCircuit,
    p: float,
) -> npt.NDArray[np.float32]:
    """Computes the theoretical probabilities of a circuit execution with a
    dephasing noise happening with a specific probability.

    Args:
        circ: The circuit to run.
        p: The probability of the dephasing noise.

    Returns:
        The probabilities corresponding to each basis state.
    """
    d: int = 2 ** (circ.nb_qubits)

    state = np.zeros(d, dtype=np.complex64)
    state[0] = 1

    for gate in circ.instructions:
        if isinstance(gate, Gate):
            state @= gate.to_matrix(circ.nb_qubits).astype(np.complex64).T

    dephased_state = apply_global_dephasing_noise(np.outer(state, np.conj(state)), p, d)
    bitfliped_state = apply_global_bitflip_noise(dephased_state, p, d)

    return bitfliped_state.diagonal().astype(np.float32)


def validate(
    mpqp_counts: list[int],
    theoretical_probabilities: npt.NDArray[np.float32],
    alpha: float = 0.1,
) -> bool:
    dist = jensenshannon(mpqp_counts, theoretical_probabilities * sum(mpqp_counts))

    return float(dist) < alpha


def dist_alpha_matching(alpha: float):
    """The trust interval is computed from the distance between the circuit
    without noise and the noisy circuits probability distributions. This
    interval depends on this distance in a non linear manner. This function
    gives the relation between the two.

    Args:
        alpha: The distance Jensen-Shannon between the non noisy distribution
            and the noisy distribution.

    Returns:
        The trust interval for the distance.
    """
    return 2 * (np.sqrt(alpha) - alpha)


# TODO: this bound it coming out of my ass, see if these results already in
# statistics. Some useful resources might be: jensen inequality, chernoff bound


def trust_int(noiseless_circuit: QCircuit, p: float):
    """Given a circuit, this computes the trust interval for the output samples
    for a specific noise level.

    Args:
        noiseless_circuit: The circuit without any noise.
        p: The probably of the dephasing noise happening.

    Returns:
        The size of the trust interval (related to the Jensen-Shannon distance).
    """
    noiseless_probs = theoretical_probs(noiseless_circuit, 0)
    noisy_probs = theoretical_probs(noiseless_circuit, p)
    return dist_alpha_matching(float(jensenshannon(noiseless_probs, noisy_probs)))


def exp_id_dist(noiseless_circuit: QCircuit, p: float, shots: int = 1024):
    """This function computes Jensen-Shannon the distance between the non noisy
    distribution and the noisy distribution.

    Args:
        noiseless_circuit: The circuit without any noise.
        p: The probably of the dephasing noise happening.
        shots: Number of shots in the basis measurement.

    Returns:
        The distance between the non noisy distribution and the noisy
        distribution.
    """
    noisy_probs = theoretical_probs(noiseless_circuit, p)

    noisy_circuit = noiseless_circuit.without_measurements()
    noisy_circuit.add([BasisMeasure(shots=shots), Depolarizing(p)])
    mpqp_counts = _run_single(
        noisy_circuit.hard_copy(), AWSDevice.BRAKET_LOCAL_SIMULATOR, {}
    ).counts

    return float(jensenshannon(mpqp_counts, noisy_probs * sum(mpqp_counts)))


def validate_noisy_circuit(
    noiseless_circuit: QCircuit, p: float, shots: int = 1024
) -> bool:
    """Validates our noise pipeline for a circuit.

    Args:
        noiseless_circuit: The circuit without any noise.
        p: The probably of the dephasing noise happening.
        shots: Number of shots in the basis measurement.

    Returns:
        Weather our noise pipeline matches the theory or not.
    """
    return exp_id_dist(noiseless_circuit, p, shots) <= trust_int(noiseless_circuit, p)


def exp_id_dist_excess(
    noiseless_circuit: QCircuit, p: float, shots: int = 1024
) -> float:
    """Computes the gap between theory and our noise pipeline for a circuit.

    Args:
        noiseless_circuit: The circuit without any noise.
        p: The probably of the dephasing noise happening.
        shots: Number of shots in the basis measurement.

    Returns:
        The gap between:

        1. the distance between theory and our results;
        2. and the trust interval.
    """
    return exp_id_dist(noiseless_circuit, p, shots) - trust_int(noiseless_circuit, p)


if __name__ == "__main__":
    from mpqp.all import *

    c = QCircuit(
        [
            H(0),
            X(1),
            Y(2),
            Z(0),
            S(1),
            T(0),
            Rx(1.2324, 2),
            Ry(-2.43, 0),
            Rz(1.04, 1),
            Rk(-1, 0),
            P(-323, 2),
            U(1.2, 2.3, 3.4, 2),
            SWAP(2, 1),
            CNOT(0, 1),
            CZ(1, 2),
        ]
    )

    probs = [0.001, 0.01, 0.1, 0.1, 0.2, 0.3]
    xs = sum(([elt] * 6 for elt in probs), start=[])
    shots_vals = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
    ys = shots_vals * 6
    dists = np.array(
        [exp_id_dist_excess(c, prob, shots) for prob in probs for shots in shots_vals]
    )

    ax = Axes3D(plt.figure())
    surf = ax.plot_trisurf(xs, np.log(ys), dists)

    ax.set_xlabel('probs')
    ax.set_ylabel('shots')
    ax.set_yticks(np.log(ys), map(str, ys))
    ax.set_zlabel('dists')

    plt.show()
