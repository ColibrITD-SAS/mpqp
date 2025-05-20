from enum import Enum, auto
from typing import Union

import numpy as np
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.native_gates import Rx, Ry
from mpqp.core.instruction.instruction import Instruction
from mpqp.core.instruction.measurement.expectation_value import Observable
from mpqp.core.instruction.measurement.pauli_string import (
    I,
    X,
    Y,
    PauliString,
    PauliStringMonomial,
)
from mpqp.execution.devices import AvailableDevice, IBMDevice
from mpqp.execution.result import Result
from mpqp.execution.runner import run
import numpy.typing as npt


class CommutingTypes(Enum):
    FULL = auto()
    QUBITWISE = auto()


class GroupingMethods(Enum):
    GREEDY = auto()
    COLORING_GREEDY = auto()
    COLORING_SF = auto()
    COLORING_LF = auto()
    COLORING_RECURSIVE_LF = auto()
    COLORING_DB = auto()
    COLORING_DSATUR = auto()
    CLIQUE_REMOVING = auto()


def find_qubitwise_rotations(group: list[PauliStringMonomial]) -> list[Instruction]:
    """Returns the single qubit rotations to handle multi observables in case of qwc grouping.
    This function is used in conjunction with the observables grouping it rotates each qubits
    into the shared eigenbasis of the elements of the group.

    Returns:
        A list of single qubit instructions.
    """
    result = []
    for i, atoms in enumerate(group[0].atoms):
        if atoms == X:
            result.append(Ry(-np.pi / 2, i))
        elif atoms == Y:
            result.append(Rx(-np.pi / 2, i))
    return result


def pauli_grouping_greedy(monomials: list[PauliStringMonomial], type: CommutingTypes):
    """Regroups the Pauli operators in parameters into groups of
    mutual commuting Pauli operators using a greedy approach.

    Args:
        monomials: list of Pauli monomials to regroup.
        type: the commuting type by which we want to group the Pauli operators.

    Returns:
        A list of list of Pauli monomials containing the fully commutativity grouping.


    Examples:
        >>> from mpqp.measures import I, X, Y, Z
        >>> full_commutation_pauli_grouping_greedy([I@X@X, Y@Y@Z, I@I@I, -3*Z@Y@X, Y@X@Y, -Z@Z@Y, 2*X@X@Y])
        [[I@X@X, Y@Y@Z, I@I@I], [-3*Z@Y@X, -1*Z@Z@Y], [Y@X@Y], [2*X@X@Y]]
    """
    groups: list[list[PauliStringMonomial]] = []
    for monomial in monomials:
        added = False
        for group in groups:
            if type == CommutingTypes.QUBITWISE:
                if all(monomial.qubit_wise_commutes_with(m_g) for m_g in group):
                    group.append(monomial)
                    added = True
                    break
            elif type == CommutingTypes.FULL:
                if all(monomial.commutes_with(m_g) for m_g in group):
                    group.append(monomial)
                    added = True
                    break

        if not added:
            groups.append([monomial])

    return groups


def pauli_monomial_eigenvalues(monom: PauliStringMonomial) -> npt.NDArray[np.float64]:
    result = np.array([1], dtype=np.float64)
    eigen_I = np.array([1, 1])
    eigen_XYZ = np.array([1, -1])
    for atom in monom.atoms:
        if atom == I:
            result = np.kron(result, eigen_I)
        else:
            result = np.kron(result, eigen_XYZ)
    return result


def run_optimized_multi_observables(
    circuit: QCircuit,
    observable: Union[PauliString, Observable],
    device: AvailableDevice = IBMDevice.AER_SIMULATOR,
    commuting_type: CommutingTypes = CommutingTypes.QUBITWISE,
    grouping_method: GroupingMethods = GroupingMethods.GREEDY,
) -> float:
    """This function performs the Pauli grouping and returns the expectation value of the observable.

    Args:
        circuit: The quantum circuit to measure
        observable: The observable by which the circuit is measured, either a Pauli string or a matrix.
        device: The device on which the circuit should be ran.
        commuting_type: The type of commuting used for the Pauli grouping.
        grouping_method: The method used to group the commuting monomials.

    Returns:
        The expectation value of the circuit by the observable.
    """
    if isinstance(observable, Observable):
        observable = observable.pauli_string
    if grouping_method == GroupingMethods.GREEDY:
        grouping = pauli_grouping_greedy(observable.monomials, commuting_type)
    else:
        raise ValueError("This type of grouping is not currently supported.")
    result = 0
    for group in grouping:
        local_result = run(circuit + QCircuit(find_qubitwise_rotations(group)), device)
        assert isinstance(local_result, Result)
        for monom in group:
            expectation_value = np.dot(
                pauli_monomial_eigenvalues(monom), local_result.probabilities
            )
            result += expectation_value * monom.coef
    return result


def full_commutation_pauli_grouping_ibm_clique(monomials: list[PauliStringMonomial]):
    pass
