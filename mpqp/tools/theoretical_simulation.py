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
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.noise import Depolarizing
from mpqp.noise.noise_model import NoiseModel


# TODO: for now the noise is applied on the whole state and not only on the
# targets, fix this
def apply_global_depolarizing_noise(
    state: npt.NDArray[np.complex64], p: float, targets: set[int]
):
    d = len(state)
    return (1 - p) * state + p * np.eye(d) / d


def apply_global_bitflip_noise(
    state: npt.NDArray[np.complex64], p: float, targets: set[int]
):
    d = len(state)
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
        start=np.zeros((d, d), dtype=np.complex64),
    )


def theoretical_probs(
    circ: QCircuit,
) -> npt.NDArray[np.float32]:
    """Computes the theoretical probabilities of a (potentially noisy circuit
    execution.

    Args:
        circ: The circuit to run.

    Returns:
        The probabilities corresponding to each basis state.
    """
    d: int = 2 ** (circ.nb_qubits)

    state = np.zeros((d, d), dtype=np.complex64)
    state[0] = 1

    for gate in circ.get_gates():
        g = gate.to_matrix(circ.nb_qubits).astype(np.complex64)
        state = g @ state @ g.T
        for noise in circ.noises:
            if (
                len(noise.gates) == 0
                or type(gate) in noise.gates
                and gate.connections().issubset(noise.targets)
            ):
                if isinstance(noise, Depolarizing):
                    state = apply_global_depolarizing_noise(
                        state, noise.prob, gate.connections()
                    )
                elif isinstance(noise, BitFlip):
                    state = apply_global_bitflip_noise(
                        state, noise.prob, gate.connections()
                    )
                else:
                    raise NotImplementedError(f"{noise} not yest implemented.")

    connected_qubits = set().union(gate.connections() for gate in circ.get_gates())
    unconnected_qubits = set(range(circ.nb_qubits)).difference(connected_qubits)
    for noise in circ.noises:
        if len(noise.gates) == 0:
            if isinstance(noise, Depolarizing):
                state = apply_global_depolarizing_noise(
                    state, noise.prob, unconnected_qubits
                )
            elif isinstance(noise, BitFlip):
                state = apply_global_bitflip_noise(
                    state, noise.prob, unconnected_qubits
                )
            else:
                raise NotImplementedError(f"{noise} not yest implemented.")

    return state.diagonal().astype(np.float32)


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


def trust_int(circuit: QCircuit):
    """Given a circuit, this computes the trust interval for the output samples
    given into consideration the noise in the circuit.

    Args:
        circuit: The circuit.

    Returns:
        The size of the trust interval (related to the Jensen-Shannon distance).
    """
    noiseless_circuit = QCircuit(
        [inst for inst in circuit.instructions if not isinstance(inst, NoiseModel)]
    )
    noiseless_probs = theoretical_probs(noiseless_circuit)
    noisy_probs = theoretical_probs(circuit)
    return dist_alpha_matching(float(jensenshannon(noiseless_probs, noisy_probs)))


def exp_id_dist(
    circuit: QCircuit,
    shots: int = 1024,
    device: AvailableDevice = AWSDevice.BRAKET_LOCAL_SIMULATOR,
):
    """This function computes Jensen-Shannon the distance between the non noisy
    distribution and the noisy distribution.

    Args:
        circuit: The circuit.
        shots: Number of shots in the basis measurement.
        device: The device to be tested.

    Returns:
        The distance between the non noisy distribution and the noisy
        distribution.
    """
    noisy_probs = theoretical_probs(circuit)

    noisy_circuit = circuit.without_measurements()
    noisy_circuit.add(BasisMeasure(shots=shots))
    mpqp_counts = _run_single(noisy_circuit.hard_copy(), device, {}).counts

    return float(jensenshannon(mpqp_counts, noisy_probs * sum(mpqp_counts)))


def validate_noisy_circuit(
    circuit: QCircuit,
    shots: int = 1024,
    device: AvailableDevice = AWSDevice.BRAKET_LOCAL_SIMULATOR,
) -> bool:
    """Validates our noise pipeline for a circuit.

    Args:
        circuit: The circuit (with potential noises).
        shots: Number of shots in the basis measurement.
        device: The device to be tested.

    Returns:
        Weather our noise pipeline matches the theory or not.
    """
    return exp_id_dist(circuit, shots, device) <= trust_int(circuit)


def exp_id_dist_excess(circuit: QCircuit, shots: int = 1024) -> float:
    """Computes the gap between theory and our noise pipeline for a circuit.

    Args:
        circuit: The circuit (with potential noises).
        shots: Number of shots in the basis measurement.

    Returns:
        The gap between:

        1. the distance between theory and our results;
        2. and the trust interval.
    """
    return exp_id_dist(circuit, shots) - trust_int(circuit)


if __name__ == "__main__":
    from mpqp.all import *

    gates = [
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

    probs = [0.001, 0.01, 0.1, 0.1, 0.2, 0.3]
    xs = sum(([elt] * 6 for elt in probs), start=[])
    shots_vals = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
    ys = shots_vals * 6
    dists = np.array(
        [
            exp_id_dist_excess(QCircuit(gates + [Depolarizing(prob)]), shots)
            for prob in probs
            for shots in shots_vals
        ]
    )

    ax = Axes3D(plt.figure())
    surf = ax.plot_trisurf(xs, np.log(ys), dists)

    ax.set_xlabel('probs')
    ax.set_ylabel('shots')
    ax.set_yticks(np.log(ys), map(str, ys))
    ax.set_zlabel('dists')

    plt.show()
