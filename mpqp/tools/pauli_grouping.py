from mpqp.core.instruction.measurement.pauli_string import PauliStringMonomial


def full_commutation_pauli_grouping_greedy(monomials: set[PauliStringMonomial]):
    """
    TODO: comment
    Args:
        monomials:

    Returns:

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


def full_commutation_pauli_grouping_ibm_clique(monomials: set[PauliStringMonomial]):
    pass


def qubit_wise_commutation_pauli_grouping(monomials: set[PauliStringMonomial]):
    pass
