from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

I = np.eye(2)
H = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
X = np.array([[0, 1], [1, 0]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

noise_proba = 0.7
shots = 1024


def apply_gate(state, gate, qubit, num_qubits):
    gate_matrix = np.eye(1)
    for i in range(num_qubits):
        gate_matrix = np.kron(gate_matrix, gate if i == qubit else I)
    return gate_matrix @ state


def apply_cnot(state, control, target):
    control_mask = 1 << control
    target_mask = 1 << target
    new_state = state.copy()
    for i in range(len(state)):
        if (i & control_mask) and not (i & target_mask):
            j = i ^ target_mask
            new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state


def depolarizing_noise(density_matrix, p):
    d = len(density_matrix)
    identity = np.eye(d)
    return (1 - p) * density_matrix + p / d * identity


def measure_state(state):
    probabilities = np.abs(state) ** 2
    cumulative_prob = np.cumsum(probabilities)
    random_number = np.random.rand()
    return np.searchsorted(cumulative_prob, random_number)


def run_experiment(initial_state, gates, noise_proba, shots):
    num_qubits = int(np.log2(len(initial_state)))
    measurement_results = []

    for _ in range(shots):
        state = np.array(initial_state, dtype=complex)
        for gate, qubit in gates:
            if gate == "H":
                state = apply_gate(state, H, qubit, num_qubits)
            elif gate == "X":
                state = apply_gate(state, X, qubit, num_qubits)
            elif gate == "CNOT":
                state = apply_cnot(state, qubit[0], qubit[1])

        density_matrix = np.outer(state, np.conj(state))

        noisy_density_matrix = depolarizing_noise(density_matrix, noise_proba)

        # collapse to state vector for measurement
        probabilities = np.abs(noisy_density_matrix.flatten()) ** 2
        probabilities /= np.sum(probabilities)
        state = np.random.choice(len(probabilities), p=probabilities)
        measured_state = format(state, f"0{num_qubits}b")
        measurement_results.append(measured_state)

    return measurement_results


def results_to_dict(measurement_results, num_qubits):
    counter = Counter(measurement_results)
    max_value = max(map(int, counter.keys()))
    width = max(num_qubits, len(bin(max_value)) - 2)

    results_dict = {}
    for x, count in counter.items():
        binary_str = format(int(x), f"0{width}b")
        results_dict[binary_str[-num_qubits:]] = count
    sorted_results = sorted(results_dict.items(), key=lambda x: int(x[0], 2))
    return dict(sorted_results)


def plot_results(measurement_results, num_qubits):
    results_dict = results_to_dict(measurement_results, num_qubits)

    sorted_results = sorted(results_dict.items(), key=lambda x: int(x[0], 2))
    states, counts = zip(*sorted_results)

    x = np.arange(len(states))  # to plot the counts for each state

    plt.bar(x, counts, align="center", width=0.4)
    plt.xticks(x, states)
    plt.xlabel("States")
    plt.ylabel("Counts")
    plt.title("Measurement Results")
    plt.show()