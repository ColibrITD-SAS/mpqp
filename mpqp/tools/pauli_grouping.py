from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import numpy.typing as npt

from mpqp.core.instruction.instruction import Instruction
from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
from mpqp.core.instruction.measurement.pauli_string import (
    CommutingTypes,
    PauliStringMonomial,
)
from mpqp.execution.job import Job, JobStatus
from mpqp.execution.result import Result


def find_qubitwise_rotations(
    group: list[PauliStringMonomial], targets: Optional[list[int]] = None
) -> list[Instruction]:
    """Returns the single qubit rotations to handle multi observables in case of
    QWC grouping. This function is used in conjunction with the observables
    grouping, it rotates each qubits into the shared eigenbasis of the elements
    of the group.

    Args:
        group: The grouped up pauli monomials from which we find the common eigenbasis
        targets: Optional list of targets to give context where the instructions should be placed on the circuit.

    Returns:
        A list of single qubit instructions.
    """
    if targets is None:
        targets = list(range(group[0].nb_qubits))
    result = []
    for i, atoms in enumerate(group[0].atoms):
        if atoms.label == "I":
            all_identity = True
            for monomial in group:
                all_identity &= monomial.atoms[i].label == "I"
                if not all_identity:
                    for base in monomial.atoms[i].get_basis_change():
                        result.append(base(targets[i]))
                    break
            continue
        for base in atoms.get_basis_change():
            result.append(base(targets[i]))
    return result


def pauli_grouping_greedy(
    monomials: list[PauliStringMonomial], type: CommutingTypes
) -> list[list[PauliStringMonomial]]:
    """Regroups the Pauli operators in parameters into groups of
    mutual commuting Pauli operators using a greedy approach.

    Args:
        monomials: list of Pauli monomials to regroup.
        type: the commuting type by which we want to group the Pauli operators.

    Returns:
        A list of list of Pauli monomials containing the fully commutativity grouping.


    Examples:
        >>> pauli_grouping_greedy(
        ...     [pI@pX@pX, pY@pY@pZ, pI@pI@pI, -3*pZ@pY@pX, pY@pX@pY, -pZ@pZ@pY, 2*pX@pX@pY],
        ...     CommutingTypes.FULL,
        ... )
        [[pI@pX@pX, pY@pY@pZ, pI@pI@pI], [-3*pZ@pY@pX, -1*pZ@pZ@pY], [pY@pX@pY], [2*pX@pX@pY]]
    """
    groups: list[list[PauliStringMonomial]] = []

    for monomial in monomials:
        added = False
        for group in groups:
            found = False
            for monoms in group:
                if monoms.name == monomial.name:
                    found = True
                    break
            if found:
                added = True
                break
            if all(monomial.commutes_with(m_g, type) for m_g in group):
                group.append(monomial)
                added = True
                break

        if not added:
            groups.append([monomial])

    return groups


def pauli_monomial_eigenvalues(monom: PauliStringMonomial) -> npt.NDArray[np.float64]:
    """Returns the eigenvalues of a Pauli monomial as a numpy array."""
    import numpy as np

    first_half = np.array([1], dtype=np.float64)
    n = len(monom.atoms) // 2
    for i in range(n):
        first_half = np.kron(first_half, monom.atoms[i].eigen_values)
    second_half = np.array([1], dtype=np.float64)
    for j in range(n, len(monom.atoms)):
        second_half = np.kron(second_half, monom.atoms[j].eigen_values)
    return np.kron(first_half, second_half)  # pyright: ignore[reportReturnType]


def full_commutation_pauli_grouping_ibm_clique(monomials: list[PauliStringMonomial]):
    raise NotImplementedError


def reconstruct_full_counts(counts: dict[str, float], length: int):
    """Helper function for Braket's pauli grouping.
    When a count is requested from Braket it only outputs states that were measured.

    Args:
        counts: dictionary taking as keys the binary state that was measured
    """
    sorted_values: list[float] = []
    for i in range(length):
        binary_state = f"{bin(i)[2:].zfill(len(bin(length))- 3)}"
        if binary_state in counts:
            sorted_values.append(counts[binary_state].real)
        else:
            sorted_values.append(0)
    return sorted_values


def build_pauli_monomial_expectation_value(
    measure: ExpectationMeasure, monomial_values: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    exp_values: dict[str, float] = {}
    error: dict[str, float] = {}
    for i, obs in enumerate(measure.observables):
        string = obs.pauli_string
        local: float = 0
        for monoms in string.monomials:
            if TYPE_CHECKING:
                assert isinstance(monoms.coef, (int, float))
            local += monomial_values[monoms.name] * monoms.coef
        exp_values.update({f"observable_{i}": local})
        error.update({f"observable_{i}": 0.0})
    return (exp_values, error)


def build_pauli_grouping_result(
    job: Job, expectation_values: dict[str, float], variance: dict[str, Any]
) -> Result:
    results, errors = {}, {}
    if TYPE_CHECKING:
        assert isinstance(job.measure, ExpectationMeasure)
    for i, obs in enumerate(job.measure.observables):
        string = obs.pauli_string
        local: float = 0
        var = {}
        for monoms in string.monomials:
            if TYPE_CHECKING:
                assert isinstance(monoms.coef, (int, float))
            local += expectation_values[monoms.name] * monoms.coef
            var.update({monoms.name: variance[monoms.name]})
        results.update({f"observable_{i}": local})
        errors.update({f"observable_{i}": var})
    job.status = JobStatus.DONE
    if len(results) == 1:
        return Result(job, results["observable_0"], shots=job.measure.shots)
    return Result(job, results, errors=errors, shots=job.measure.shots)


def get_all_monomials(measure: ExpectationMeasure) -> list[PauliStringMonomial]:
    """Returns a list of all of the non repeating monomials present in the observables of the measure.

    Args:
        measure: The expectation measure from which we want to extract the monomials.

    Returns:
        A list of all the monomials in the measure.
    """
    monomials: list[PauliStringMonomial] = []
    for obs in measure.observables:
        if not obs.is_pauli_set:
            from warnings import warn

            warn(
                "The observable has not a set Pauli String, the inputted matrix is going to be converted into a Pauli String, this may lead to low performances."
            )
        for monom in obs.pauli_string.monomials:
            found = False
            for m in monomials:
                if monom.name == m.name:
                    found = True
                    break
            if not found:
                monomials.append(monom)
    return monomials
