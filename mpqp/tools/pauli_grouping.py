import numpy as np
import numpy.typing as npt

from mpqp.core.instruction.instruction import Instruction
from mpqp.core.instruction.measurement.pauli_string import (
    CommutingTypes,
    PauliStringMonomial,
)


def find_qubitwise_rotations(group: list[PauliStringMonomial]) -> list[Instruction]:
    """Returns the single qubit rotations to handle multi observables in case of
    QWC grouping. This function is used in conjunction with the observables
    grouping, it rotates each qubits into the shared eigenbasis of the elements
    of the group.

    Returns:
        A list of single qubit instructions.
    """
    result = []
    for i, atoms in enumerate(group[0].atoms):
        if atoms.label == "I":
            all_identity = True
            for monomial in group:
                all_identity &= monomial.atoms[i].label == "I"
                if not all_identity:
                    for base in monomial.atoms[i].get_basis_change():
                        result.append(base(i))
                    break
            continue
        for base in atoms.get_basis_change():
            result.append(base(i))
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
    result = np.array([1], dtype=np.float64)
    for atom in monom.atoms:
        result = np.kron(result, atom.eigen_values)
    return result


def full_commutation_pauli_grouping_ibm_clique(monomials: list[PauliStringMonomial]):
    pass
