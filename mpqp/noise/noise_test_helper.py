from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

from mpqp.all import *

noise_proba = 0.7
shots = 1024


def process_qcircuit(circuit: QCircuit):

    # gate_classes = [
    #     type(instr) for instr in circuit.instructions if isinstance(instr, Gate)
    # ]
    gate_operations = [
        instr for instr in circuit.instructions if issubclass(type(instr), Gate)
    ]

    gate_map = {gate.label: gate.to_matrix() for gate in gate_operations}

    gates = [
        (
            (gate.label, gate.controls + gate.targets)
            if gate.label == "CNOT"
            else (gate.label, gate.targets)
        )
        for gate in gate_operations
    ]

    return gate_map, gates


def apply_gate(state, gate, qubits, num_qubits):

    if len(qubits) == 1:
        qubit = qubits[0]
        gate_matrix = np.eye(1)
        for i in range(num_qubits):
            gate_matrix = np.kron(gate_matrix, gate if i == qubit else np.eye(2))
        return gate_matrix @ state
    elif len(qubits) == 2:
        full_gate = np.eye(1 << num_qubits, dtype=complex)
        control, target = qubits
        for i in range(1 << num_qubits):
            for j in range(1 << num_qubits):
                bin_i = format(i, f"0{num_qubits}b")
                bin_j = format(j, f"0{num_qubits}b")
                if all(
                    bin_i[k] == bin_j[k] for k in range(num_qubits) if k not in qubits
                ):
                    sub_i = int("".join(bin_i[k] for k in qubits), 2)
                    sub_j = int("".join(bin_j[k] for k in qubits), 2)
                    full_gate[i, j] = gate[sub_i, sub_j]
        return full_gate @ state
    else:
        raise ValueError("Only single-qubit and two-qubit gates are supported.")


def depolarizing_noise(density_matrix, p):
    d = len(density_matrix)
    identity = np.eye(d)
    return (1 - p) * density_matrix + p / d * identity


def run_experiment(initial_state, gates, gate_map, noise_proba, shots):
    num_qubits = int(np.log2(len(initial_state)))
    measurement_results = []

    for _ in range(shots):
        state = np.array(initial_state, dtype=complex)
        for gate, qubits in gates:
            if gate in gate_map:
                state = apply_gate(state, gate_map[gate], qubits, num_qubits)

        density_matrix = np.outer(state, np.conj(state))

        noisy_density_matrix = depolarizing_noise(density_matrix, noise_proba)

        probabilities = np.abs(noisy_density_matrix.flatten()) ** 2
        probabilities /= np.sum(probabilities)

        probabilities = np.asarray(probabilities, dtype=np.float64)

        state = np.random.choice(len(probabilities), p=probabilities)
        measured_state = format(state, f"0{num_qubits}b")
        measurement_results.append(measured_state)

    return measurement_results


def results_to_dict(measurement_results, num_qubits, shots):
    counter = Counter(measurement_results)
    max_value = max(map(int, counter.keys()))
    width = max(num_qubits, len(bin(max_value)) - 2)

    results_dict = {}
    for x, count in counter.items():
        binary_str = format(int(x), f"0{width}b")
        results_dict[binary_str[-num_qubits:]] = count

    total_counts = sum(results_dict.values())
    for state in results_dict:
        results_dict[state] = round(results_dict[state] * shots / total_counts)

    max_count_state = max(results_dict, key=results_dict.get)
    results_dict[max_count_state] += shots - sum(results_dict.values())

    sorted_results = sorted(results_dict.items(), key=lambda x: int(x[0], 2))
    return dict(sorted_results)


def print_results(results_dict, num_qubits, shots):
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


def plot_results(measurement_results, num_qubits):
    results_dict = results_to_dict(measurement_results, num_qubits, shots)

    sorted_results = sorted(results_dict.items(), key=lambda x: int(x[0], 2))
    states, counts = zip(*sorted_results)

    x = np.arange(len(states))  # plot the counts for each state
    states = [f"|{state}>" for state in states]  # "|00>", "|01>", "|10>", "|11>"

    plt.bar(x, counts, align="center", width=0.6)
    plt.xticks(x, states)
    plt.xlabel("States")
    plt.ylabel("Counts")
    plt.title("Results without SDK")
    plt.show()


def chisquare_test(mpqp_counts, theoretical_counts, shots, alpha=0.05):

    if not mpqp_counts or not theoretical_counts:
        return {
            "expected_counts": [],
            "chisquare_stat": np.nan,
            "p_value": 1.0,
            "significant": False,
        }

    theoretical_probabilities = [count / shots for count in theoretical_counts]
    expected_counts = [int(tp * shots) for tp in theoretical_probabilities]

    chisquare_stat, p_value = chisquare(mpqp_counts, expected_counts)

    return {
        "expected_counts": expected_counts,
        "chisquare_stat": chisquare_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
    }
