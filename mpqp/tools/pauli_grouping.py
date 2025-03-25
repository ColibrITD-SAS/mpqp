from mpqp.core.instruction.measurement.pauli_string import PauliStringMonomial


def full_commutation_pauli_grouping_greedy(
    monomials: list[PauliStringMonomial],
) -> list[list[PauliStringMonomial]]:
    """Regroups the Pauli operators in parameters into groups of
    mutual fully commuting Pauli operators using a greedy approach.

    Args:
        monomials: list of Pauli monomials to regroup.

    Returns:
        A list of list of Pauli monomials containing the fully commutativity grouping.


    Examples:
        >>> from mpqp.measures import I, X, Y, Z
        >>> full_commutation_pauli_grouping_greedy([I@X@X, Y@Y@Z, I@I@I, -3*Z@Y@X, Y@X@Y, -Z@Z@Y, 2*X@X@Y])
        [[I@X@X, Y@Y@Z, I@I@I], [-3*Z@Y@X, -1*Z@Z@Y], [Y@X@Y], [2*X@X@Y]]
    """

    groups = []

    for m in monomials:
        added = False
        for group in groups:
            if all(m.commutes_with(m_g) for m_g in group):
                group.append(m)
                added = True
                break

        if not added:
            groups.append([m])

    return groups


def full_commutation_pauli_grouping_ibm_clique(monomials: list[PauliStringMonomial]):
    pass


def qubit_wise_commutation_pauli_grouping(monomials: list[PauliStringMonomial]):
    pass
