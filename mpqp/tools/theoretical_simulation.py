"""This module contains the tools to check if a result obtained by running the 
circuit on one of our provider's device produce a result compatible with the
theory. (Which could itself be useful to find problems either in our code or in
our provider's code.)

In order to do so, :func:`theoretical_probs` performs a theoretical simulation 
of a :class:`~mpqp.core.circuit.QCircuit`. The noise models in the circuit will
be taken into account (but not the shot noise).

This theoretical simulation will be compared against the execution on a
:class:`~mpqp.execution.devices.AvailableDevice` in :func:`exp_id_dist`. This
comparison is performed using a distance between statistical distribution called
the Jensen-Shannon distance.

If the distance is small enough, :func:`validate_noisy_circuit` will return
``True``, meaning that the empirical data is within the expected range. 

The size of the trust interval is computed in :func:`trust_int` by computing the
Jensen-Shannon distance between a non noisy circuit and a noisy one. The idea
being that the more noise you have in a circuit, the bigger your trust interval
needs to be (the uncertainty is bigger).

The interval size is not linearly related with this distance so this needs to be
passed in a re-normalizing function before being used. This is the role
:func:`dist_alpha_matching` is playing."""

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import jensenshannon
from typeguard import typechecked

from mpqp import QCircuit
from mpqp.execution import AvailableDevice, AWSDevice
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.measures import BasisMeasure


@typechecked
def amplitude(
    circ: QCircuit,
) -> npt.NDArray[np.complex64]:
    """Computes the theoretical probabilities of a (potentially) noisy circuit
    execution.

    Args:
        circ: The circuit to run.

    Returns:
        The probabilities corresponding to each basis state.
    """
    d: int = 2**circ.nb_qubits

    state = np.zeros((d), dtype=np.complex64)
    state[0] = 1
    gates = circ.gates
    print(state)

    for gate in gates:
        g = gate.to_matrix(circ.nb_qubits).astype(np.complex64)
        print(g)
        state = g @ state
        print(state)
        for noise in circ.noises:
            if (
                len(noise.gates) == 0
                or type(gate) in noise.gates
                and gate.connections().issubset(noise.targets)
            ):
                state = sum(
                    (
                        k @ state
                        for k in noise.to_adjusted_kraus_operators(
                            gate.connections(), circ.nb_qubits
                        )
                    ),
                    start=np.zeros(d, dtype=np.complex64),
                )

    connected_qubits = set().union(*[gate.connections() for gate in gates])
    unconnected_qubits = set(range(circ.nb_qubits)).difference(connected_qubits)
    for noise in circ.noises:
        if len(noise.gates) == 0:
            state = sum(
                (
                    k @ state
                    for k in noise.to_adjusted_kraus_operators(
                        unconnected_qubits, circ.nb_qubits
                    )
                ),
                start=np.zeros(d, dtype=np.complex64),
            )

    return state


@typechecked
def theoretical_probs(
    circ: QCircuit,
) -> npt.NDArray[np.float32]:
    """Computes the theoretical probabilities of a (potentially) noisy circuit
    execution.

    Args:
        circ: The circuit to run.

    Returns:
        The probabilities corresponding to each basis state.
    """
    d: int = 2**circ.nb_qubits

    state = np.zeros((d, d), dtype=np.complex64)
    state[0, 0] = 1
    gates = circ.gates

    for gate in gates:
        g = gate.to_matrix(circ.nb_qubits).astype(np.complex64)
        state = g @ state @ g.T.conj()
        for noise in circ.noises:
            if (
                len(noise.gates) == 0
                or type(gate) in noise.gates
                and gate.connections().issubset(noise.targets)
            ):
                state = sum(
                    (
                        k @ state @ k.T.conj()
                        for k in noise.to_adjusted_kraus_operators(
                            gate.connections(), circ.nb_qubits
                        )
                    ),
                    start=np.zeros((d, d), dtype=np.complex64),
                )

    connected_qubits = set().union(*[gate.connections() for gate in gates])
    unconnected_qubits = set(range(circ.nb_qubits)).difference(connected_qubits)
    for noise in circ.noises:
        if len(noise.gates) == 0:
            state = sum(
                (
                    k @ state @ k.T.conj()
                    for k in noise.to_adjusted_kraus_operators(
                        unconnected_qubits, circ.nb_qubits
                    )
                ),
                start=np.zeros((d, d), dtype=np.complex64),
            )

    return state.diagonal().real


@typechecked
def dist_alpha_matching(alpha: float):
    """The trust interval is computed from the distance between the circuit
    without noise and the noisy circuits probability distributions. This
    interval depends on this distance in a non linear manner. This function
    gives the relation between the two.

    Args:
        alpha: The distance Jensen-Shannon between the non noisy distribution
            and the noisy distribution.

    Returns:
        Diameter of the trust interval for the distance.
    """
    return np.sqrt(alpha)


# TODO: this bound it coming out of my ass, see if these results already in
# statistics. Some useful resources might be: Jensen inequality, Chernoff bound,
# ...


@typechecked
def trust_int(circuit: QCircuit):
    """Given a circuit, this computes the diameter of the trust interval for the
    output samples given into consideration the noise in the circuit.

    Args:
        circuit: The circuit.

    Returns:
        The size of the trust interval (related to the Jensen-Shannon distance).
    """
    noiseless_circuit = circuit.without_noises()
    noiseless_probs = theoretical_probs(noiseless_circuit)
    noisy_probs = theoretical_probs(circuit)
    return dist_alpha_matching(float(jensenshannon(noiseless_probs, noisy_probs)))


@typechecked
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
    mpqp_counts = _run_single(noisy_circuit, device, {}).counts

    return float(jensenshannon(mpqp_counts, noisy_probs * sum(mpqp_counts)))


# @typechecked
# FIXME: Resolve type-checking errors encountered during test execution.
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


@typechecked
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
    return abs(exp_id_dist(circuit, shots) - trust_int(circuit))


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
