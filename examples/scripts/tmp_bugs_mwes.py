"""Local reproductions for the eight findings in the review of MPQP PR #197.

Run all cases: python tmp_bugs_mwes.py
Run selected cases: python tmp_bugs_mwes.py 1 3 6

Requires MPQP's Qiskit extra. Every execution uses local Aer; no credentials or
remote jobs are needed. Observations below were checked on commit 3ba38e49.
Noise counts vary between runs; their expected probabilities are hand-computed.
This is a temporary review aid, not a regression test or a proposed library fix.
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable

import numpy as np
from sympy import Symbol

from mpqp import (
    BasisMeasure,
    BindingMode,
    BitFlip,
    CircuitBinding,
    ExpectationMeasure,
    IBMDevice,
    Observable,
    QCircuit,
    Ry,
    X,
    pZ,
    run,
)

DEVICE = IBMDevice.AER_SIMULATOR
SHOTS = 2048


def sample_counts(binding: CircuitBinding) -> list[int]:
    return run(binding, DEVICE).results[0].counts


def noisy_binding(circuit: QCircuit | None = None) -> CircuitBinding:
    # Explicit targets isolate caching/nesting from issue 1's default targets.
    return CircuitBinding(
        QCircuit([X(0)]) if circuit is None else circuit,
        measurements=BasisMeasure(shots=SHOTS),
        noises=[BitFlip(0.5, targets=[0])],
    )


def approximately_half(counts: list[int]) -> bool:
    # Broad tolerance: this demonstration does not depend on an exact random draw.
    return sum(counts) == SHOTS and 0.35 < counts[0] / SHOTS < 0.65


def issue_1_default_noise_targets() -> (
    bool
):  # osef, le noise sera retravaillé plus tard
    # Expected: X prepares |1>; BitFlip(0.5) then yields P(0) = P(1) = 0.5.
    # Omitting targets means all qubits, so both bindings should have that law.
    # Actual: default targets give [0, 2048]; explicit [0] gives roughly 50/50.
    # Cause: transpiled_circuits assigns c.noises directly instead of using
    # QCircuit.add to expand dynamic noise targets. The empty target list reaches
    # generate_qiskit_noise_model, which consequently attaches no errors.
    default_targets = CircuitBinding(
        QCircuit([X(0)]),
        measurements=BasisMeasure(shots=SHOTS),
        noises=[BitFlip(0.5)],
    )
    default_counts = sample_counts(default_targets)
    explicit_counts = sample_counts(noisy_binding())
    print("Expected: both counts approximately [1024, 1024].")
    print(f"Actual default targets: {default_counts}")
    print(f"Actual explicit targets: {explicit_counts}")
    return default_counts == [0, SHOTS] and approximately_half(explicit_counts)


def issue_2_nested_noise() -> bool:  # osef, le noise sera retravaillé plus tard
    # Expected: wrapping a noisy binding changes neither its circuit nor its
    # noise. Both forms should yield approximately [1024, 1024].
    # Actual: the flat binding is noisy; the nested binding gives [0, 2048].
    # Cause: recursive transpilation sets the child's transpiled_noise_model,
    # but run_aer reads only the parent's model, which remains None.
    flat_counts = sample_counts(noisy_binding())
    child = noisy_binding()
    parent = CircuitBinding(child)
    nested_counts = sample_counts(parent)
    print("Expected: flat and nested counts approximately [1024, 1024].")
    print(f"Actual flat: {flat_counts}; nested: {nested_counts}")
    print(f"Child has noise model: {child.transpiled_noise_model is not None}")
    print(f"Parent has noise model: {parent.transpiled_noise_model is not None}")
    return approximately_half(flat_counts) and nested_counts == [0, SHOTS]


def issue_3_cached_clean_circuit() -> (
    bool
):  # osef, la transpilation sera retravaillé plus tard
    # Expected: previous clean execution must not disable a new binding's noise.
    # Both fresh and reused circuits should yield approximately [1024, 1024].
    # Actual: the reused circuit gives [0, 2048].
    # Cause: transpiled_circuits generates a noise model only when the circuit's
    # transpiled_circuit is None. A previous clean binding populated that cache,
    # so the new noisy binding runs with no noise model.
    fresh_counts = sample_counts(noisy_binding())
    circuit = QCircuit([X(0)])
    run(CircuitBinding(circuit), DEVICE)
    print(f"Clean execution populated cache: {circuit.transpiled_circuit is not None}")
    reused = noisy_binding(circuit)
    reused_counts = sample_counts(reused)
    print("Expected: fresh and reused counts approximately [1024, 1024].")
    print(f"Actual fresh: {fresh_counts}; reused: {reused_counts}")
    print(
        f"Reused binding has noise model: {reused.transpiled_noise_model is not None}"
    )
    return approximately_half(fresh_counts) and reused_counts == [0, SHOTS]


def issue_4_zero_shots() -> (
    bool
):  # vrai pbm (le truc ne devrai pas être classifié comme sample)
    # Expected: BasisMeasure(shots=0) requests a state vector, as it does for an
    # ordinary QCircuit. X|0> should return amplitudes [0, 1].
    # Actual: the binding is classified SAMPLE, then Aer result extraction raises
    # QiskitError: 'No counts for experiment "0"'.
    # Cause: CircuitBinding classifies every BasisMeasure as SAMPLE without
    # checking shots, unlike QCircuit.job_type.
    from qiskit.exceptions import QiskitError

    direct = run(QCircuit([X(0), BasisMeasure(shots=0)]), DEVICE)
    print(f"Expected (ordinary circuit): {direct.amplitudes}")
    binding = CircuitBinding(QCircuit([X(0)]), measurements=BasisMeasure(shots=0))
    print(f"Actual binding job type: {binding.job_type.name}")
    try:
        result = run(binding, DEVICE)
    except QiskitError as error:
        print(f"Actual exception: {type(error).__name__}: {error}")
        return "No counts" in str(error)
    print(f"Actual amplitudes: {result.results[0].amplitudes}")
    return False


def issue_5_zip_singleton_broadcasting() -> bool:  # bebou hallucine, intended behavior
    # Expected: the documented singleton broadcasting yields two executions:
    # Ry(0)|0> = |0> and Ry(pi)|0> = |1>, using the same basis measurement.
    # Actual: construction raises ValueError because measurement length 1 differs
    # from parameter-set length 2.
    # Cause: the constructor requires equal lengths before unroll's singleton
    # broadcasting code can run.
    theta = Symbol("theta")
    print("Expected: two executions, counts [16, 0] then [0, 16].")
    try:
        binding = CircuitBinding(
            QCircuit([Ry(theta, 0)]),
            values=[{theta: 0.0}, {theta: float(np.pi)}],
            measurements=BasisMeasure(shots=16),
            mode=BindingMode.ZIP,
        )
    except ValueError as error:
        print(f"Actual exception: {type(error).__name__}: {error}")
        return "number of measurements" in str(error)
    print(f"Actual counts: {[r.counts for r in run(binding, DEVICE).results]}")
    return False


def issue_6_grouped_result_order() -> (
    bool
):  # vrai pbm, mathieu veux changer comment le result marche pour que ça ne dépende plus de l'ordre
    # Expected: results follow unroll/input order: X, identity, X. Measuring Z
    # gives [-1, +1, -1], independently of shots (we use exact expectations).
    # Actual: [-1, -1, +1], corresponding to X, X, identity.
    # Cause: circuitbinding_to_pubs groups executions by circuit identity, and
    # compute_expectation_value flattens PUB results without restoring indices.
    # Contract note: a new PR test explicitly expects grouped order. This example
    # exposes its difference from input/unroll order; deciding to retain grouped
    # order would require an explicit public ordering contract instead.
    first = QCircuit([X(0)])
    second = QCircuit(1)
    binding = CircuitBinding(
        [first, second, first],
        measurements=ExpectationMeasure(Observable(pZ)),
        mode=BindingMode.ZIP,
    )
    names = {id(first): "X", id(second): "identity"}
    print(f"Unroll order: {[names[id(c)] for c, _, _ in binding.unroll()]}")
    actual = [float(r.expectation_values) for r in run(binding, DEVICE).results]
    print("Expected in input/unroll order: [-1.0, 1.0, -1.0]")
    print(f"Actual: {actual}")
    return bool(np.allclose(actual, [-1, -1, 1]))


def issue_7_mixed_circuit_widths() -> (
    bool
):  # vrai pbm, interdire les width différentes quand on a des mesures non dynamiques
    # Expected if mixed widths are supported: Z on target [0] gives -1 for the
    # one-qubit X circuit and +1 for the two-qubit all-zero circuit.
    # Actual: construction succeeds, but execution raises a dimension mismatch.
    # Cause: the constructor pads the measurement to the maximum circuit width
    # once, so the one-qubit circuit receives a two-qubit observable.
    # Contract note: an author comment on GitHub says widths should be equal.
    # Under that intended restriction, the expected behavior is an informative
    # constructor rejection, rather than accepting the binding and failing later.
    print("Expected if supported: [-1.0, 1.0]; otherwise reject at construction.")
    binding = CircuitBinding(
        [QCircuit([X(0)]), QCircuit(2)],
        measurements=ExpectationMeasure(Observable(pZ), targets=[0]),
    )
    print("Actual: constructor accepted the binding.")
    try:
        result = run(binding, DEVICE)
    except ValueError as error:
        print(f"Actual execution exception: {type(error).__name__}: {error}")
        return "number of qubits" in str(error)
    print(f"Actual expectations: {[r.expectation_values for r in result.results]}")
    return False


def issue_8_measurement_mutation() -> bool:  # vrai pbm, copier la mesure
    # Expected: a binding's shot override leaves the caller's measurement at its
    # original 1024 shots. Reusing it should still request 1024 shots.
    # Actual: constructing the first binding changes the original to 16 shots;
    # a second binding using it silently inherits 16 shots.
    # Cause: the constructor executes measure.shots = self.shots on the original
    # object before making any copy.
    measure = BasisMeasure()
    original_shots = measure.shots
    CircuitBinding(QCircuit(1), measurements=measure, shots=16)
    reused = CircuitBinding(QCircuit(1), measurements=measure)
    print(f"Expected original and reused shots: {original_shots}")
    print(f"Actual original measurement shots: {measure.shots}")
    print(f"Actual reused binding shots: {reused.shots}")
    return measure.shots == 16 and reused.shots == 16


CASES: dict[int, Callable[[], bool]] = {
    1: issue_1_default_noise_targets,
    2: issue_2_nested_noise,
    3: issue_3_cached_clean_circuit,
    4: issue_4_zero_shots,
    5: issue_5_zip_singleton_broadcasting,
    6: issue_6_grouped_result_order,
    7: issue_7_mixed_circuit_widths,
    8: issue_8_measurement_mutation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("issues", nargs="*", type=int, help="Issue numbers (1-8).")
    args = parser.parse_args()
    selected = args.issues or list(CASES)
    if any(number not in CASES for number in selected):
        parser.error("Issue numbers must be between 1 and 8.")
    reproduced = 0
    unexpected_errors = 0
    for number in selected:
        case = CASES[number]
        print(f"\n--- Issue {number}: {case.__name__} ---", flush=True)
        try:
            observed = case()
        except Exception:
            unexpected_errors += 1
            print("UNEXPECTED ERROR (not counted as a reproduction):", flush=True)
            traceback.print_exc()
        else:
            reproduced += int(observed)
            print("REPRODUCED" if observed else "REPORTED SYMPTOM NOT REPRODUCED")
    print(f"\n{reproduced}/{len(selected)} reported symptoms reproduced.")
    if unexpected_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
