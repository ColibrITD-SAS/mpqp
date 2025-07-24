from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qat.core.wrappers.circuit import Circuit as my_QLM_Circuit

from mpqp import QCircuit
from mpqp.gates import *

MyQLM_Gate = Tuple[str, List[int], List[int]]


def _define_parameters(gate: MyQLM_Gate) -> tuple[int, int, int, int, int, int]:
    theta = phi = gamma = target = control_1 = control_2 = 0
    if gate[1] != []:
        if len(gate[1]) == 1:
            theta = gate[1][0]
        elif len(gate[1]) == 2:
            theta = gate[1][0]
            phi = gate[1][1]
        elif len(gate[1]) == 3:
            theta = gate[1][0]
            phi = gate[1][1]
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


def from_myqlm_to_mpqp(circuit: my_QLM_Circuit) -> QCircuit:
    """
    Parses a MyQLM circuit's instructions and returns a MPQP circuit.

    Args:
        circuit: the MyQLM circuit to be translated.

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
             ┌───┐
        q_0: ┤ H ├──■──
             └───┘┌─┴─┐
        q_1: ─────┤ X ├
                  └───┘
    """
    from mpqp.core.instruction.gates.native_gates import (
        NoParameterGate,
        OneQubitNoParamGate,
        RotationGate,
    )
    from mpqp.core.instruction.gates.custom_controlled_gate import CustomControlledGate

    qc = QCircuit(circuit.nbqbits)

    gates = {
        'H': H,
        'X': X,
        'Y': Y,
        'Z': Z,
        'I': Id,
        'S': S,
        'D-S': S_dagger,
        'T': T,
        'D-T': None,
        'RX': Rx,
        'RY': Ry,
        'RZ': Rz,
        'PH': P,
        'CNOT': CNOT,
        'CSIGN': CZ,
        'SWAP': SWAP,
        'U': U,
        'U1': P,
        'ISWAP': None,
        'SQRTSWAP': None,
        'MEASURE': None,
        'CCNOT': None,
    }

    for gate in circuit.iterate_simple():

        theta, phi, gamma, target, control_1, control_2 = _define_parameters(gate)

        controlled_gate = False
        if gate[0] not in gates:
            if gate[0].startswith("C-"):
                controlled_gate = True
            else:
                raise ValueError(f"{gate[0]} not handled yet.")

        if controlled_gate:
            mpqp_gate = gates[gate[0].split("-")[1]]
        else:
            mpqp_gate = gates[gate[0]]

        if mpqp_gate is None:
            if gate[0] == 'ISWAP':
                qc.add(
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
                qc.add(
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
            elif gate[0] == 'D-T':
                qc.add(P(-np.pi / 4, target))
            elif gate[0] == 'MEASURE':
                break

        elif issubclass(mpqp_gate, OneQubitNoParamGate):
            if controlled_gate:
                qc.add(CustomControlledGate(control_1, mpqp_gate(target)))
            else:
                qc.add(mpqp_gate(target))
        elif issubclass(mpqp_gate, RotationGate):
            if controlled_gate:
                qc.add(CustomControlledGate(control_1, mpqp_gate(theta, target)))
            else:
                qc.add(mpqp_gate(theta, target))
        elif issubclass(mpqp_gate, NoParameterGate):
            if controlled_gate:
                qc.add(CustomControlledGate(control_2, mpqp_gate(control_1, target)))
            else:
                qc.add(mpqp_gate(control_1, target))
        else:
            if controlled_gate:
                qc.add(
                    CustomControlledGate(
                        control_1, mpqp_gate(theta, phi, gamma, target)
                    )
                )
            else:
                qc.add(U(theta, phi, gamma, target))

    return qc
