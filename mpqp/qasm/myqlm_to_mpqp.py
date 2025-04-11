from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from qat.core.wrappers.circuit import Circuit as my_QLM_Circuit

from mpqp import QCircuit
from mpqp.core.instruction.gates import *

Gates = Union[
    type[H],
    type[X],
    type[Y],
    type[Z],
    type[Id],
    type[S],
    type[T],
    type[Rx],
    type[Ry],
    type[Rz],
    type[P],
    type[CNOT],
    type[CZ],
    type[SWAP],
    type[U],
]

MyQLM_Gate = Tuple[str, List[int], List[int]]

def _define_parameters(gate: MyQLM_Gate) -> tuple[int, int, int, int, int, int]:
    theta = phi = gamma = target = control_1 = control_2 = 0
    if gate[1] != []:
        if len(gate[1]) >= 1:
            theta = gate[1][0]
        if len(gate[1]) >= 2:
            phi = gate[1][1]
        if len(gate[1]) >= 3:
            gamma = gate[1][2]

    if len(gate[2]) == 1:
        target = gate[2][0]
    elif len(gate[2]) == 2:
        control_1 = gate[2][0]
        target = gate[2][1]
    elif len(gate[2]) == 3:
        control_2 = gate[2][0]
        control_1 = gate[2][1]
        target = gate[2][2]
    else:
        raise SyntaxError(f"Unhandled Gate: {str(gate[0])}")

    return theta, phi, gamma, target, control_1, control_2


def _find_mpqp_gates(
    gate: str, mpqp_gates: Sequence[tuple[str, Gates | None]]
) -> tuple[str, Gates | None]:

    idx = 0
    for idx in range(len(mpqp_gates)):
        if gate == mpqp_gates[idx][0]:
            return mpqp_gates[idx]
    raise SyntaxError(f"Unknown Gate: {str(gate[0])}")


def from_myqlm_to_mpqp(circuit: my_QLM_Circuit) -> QCircuit:
    """
    Parses a MyQLM circuit's instructions and returns a MPQP circuit.

    Args:
        circuit: the MyQLM circuit to be transformed.

    Returns:
        QCircuit object representing the parsed MyQLM circuit.

    Raises:
        SyntaxError: If the input circuit contains gates that are not handled or that have syntax error.

    Example:
        >>> from qat.lang.AQASM import Program, H, CNOT
        >>> prog = Program()
        >>> qbits = prog.qalloc(2)
        >>> _ = H(qbits[0])
        >>> _ = CNOT(qbits[0], qbits[1])
        >>> myqlm_circuit = prog.to_circ()
        >>> qcircuit = from_myqlm_to_mpqp(myqlm_circuit)
        >>> print(qcircuit) # doctest: +NORMALIZE_WHITESPACE
             ┌───┐┌───┐
        q_0: ┤ I ├┤ H ├──■──
             ├───┤└───┘┌─┴─┐
        q_1: ┤ I ├─────┤ X ├
             └───┘     └───┘
    """
    qc = QCircuit()
    for i in range(circuit.nbqbits):
        qc.add(Id(i))

    gates = [
        ('H', H),
        ('X', X),
        ('Y', Y),
        ('Z', Z),
        ('I', Id),
        ('S', S),
        ('T', T),
        ('RX', Rx),
        ('RY', Ry),
        ('RZ', Rz),
        ('PH', P),
        ('CNOT', CNOT),
        ('CSIGN', CZ),
        ('SWAP', SWAP),
        ('U', U),
        ('U1', P),
        ('ISWAP', None),
        ('SQRTSWAP', None),
        ('MEASURE', None),
        ('CCNOT', None),
    ]

    for gate in circuit.iterate_simple():

        theta, phi, gamma, target, control_1, control_2 = _define_parameters(gate)

        mpqp_gate = _find_mpqp_gates(gate[0], gates)[1]

        if mpqp_gate is None:
            if gate[0] == 'ISWAP':
                qc += QCircuit(
                    [
                        S(control_1),
                        S(target),
                        H(control_1),
                        CNOT(control_1, target),
                        CNOT(target, control_1),
                        H(target),
                    ]
                )
            elif gate[0] == 'SQRTSWAP':
                qc += QCircuit(
                    [
                        CNOT(control_1, target),
                        H(control_1),
                        CP(np.pi / 2, target, control_1),
                        H(control_1),
                        CNOT(control_1, target),
                    ]
                )
            elif gate[0] == 'CCNOT':
                qc.add(TOF([control_2, control_1], target))
            elif gate[0] == 'MEASURE':
                break

        elif issubclass(mpqp_gate, (H, X, Y, Z, Id, S, T)):
            qc.add(mpqp_gate(target))
        elif issubclass(mpqp_gate, (Rx, Ry, Rz, P)):
            qc.add(mpqp_gate(theta, target))
        elif issubclass(mpqp_gate, (CNOT, CZ, SWAP)):
            qc.add(mpqp_gate(control_1, target))
        else:
            qc.add(U(theta, phi, gamma, target))

    return qc