from __future__ import annotations

from typing import Union, Tuple, List

import numpy as np

from mpqp import QCircuit
from mpqp.core.instruction.gates import *
from qat.core.wrappers.circuit import Circuit as my_QLM_Circuit

Gates = Union[
        type[H], type[X], type[Y], type[Z], type[Id], type[S], type[T],
        type[Rx], type[Ry], type[Rz], type[P], type[CNOT], type[CZ], type[SWAP]
    ]

MyQLM_Gate = Tuple[str, List[int], List[int]]

def define_parameters(
        gate: MyQLM_Gate
    ) -> tuple[int, int, int, int, int, int]:
    
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



def from_myqlm_to_mpqp(circuit: my_QLM_Circuit) -> QCircuit:
    qc = QCircuit()
    for i in range(circuit.nbqbits):
        qc.add(Id(i))

    one_qubit_gates = ['H', 'X', 'Y', 'Z', 'I', 'S', 'T']
    mpqp_one_qubit_gates = [H, X, Y, Z, Id, S, T]
    two_qubits_gates = ['RX', 'RY', 'RZ', 'PH']
    mpqp_two_qubits_gates = [Rx, Ry, Rz, P]
    controlled_gates = ['CNOT', 'CSIGN', 'SWAP']
    mpqp_controlled_gates = [CNOT, CZ, SWAP]
    idx = 0
    
    for gate in circuit.iterate_simple():

        theta, phi, gamma, target, control_1, control_2 = define_parameters(gate)

        if gate[0] == 'CCNOT':
            qc.add(TOF([control_2, control_1], target))

        elif gate[0] in one_qubit_gates:
            idx = 0
            for idx in range(len(one_qubit_gates)):
                if gate[0] == one_qubit_gates[idx]:
                    break
            qc.add(mpqp_one_qubit_gates[idx](target))

        elif gate[0] in two_qubits_gates:
            for idx in range(len(two_qubits_gates)):
                if gate[0] == two_qubits_gates[idx]:
                    break
            qc.add(mpqp_two_qubits_gates[idx](theta, target))

        elif gate[0] in controlled_gates:
            for idx in range(len(controlled_gates)):
                if gate[0] == controlled_gates[idx]:
                    break
            qc.add(mpqp_controlled_gates[idx](control_1, target))

        else:
            if gate[0] == 'U':
                qc.add(U(theta, phi, gamma, target))
            elif gate[0] == 'U1':
                qc.add(P(theta, target))
            elif gate[0] == 'ISWAP':
                qc.add(S(target))
                qc.add(S(target))
                qc.add(H(target))
                qc.add(CNOT(control_1, target))
                qc.add(CNOT(control_1, target))
                qc.add(H(target))
            elif gate[0] == 'SQRTSWAP':
                qc.add(CNOT(control_1, target))
                qc.add(H(target))
                qc.add(CP(np.pi/2, control_1, target))
                qc.add(H(target))
                qc.add(CNOT(control_1, target))
            elif gate[0] == 'MEASURE':
                break

            else:
                raise SyntaxError(f"Unknown Gate: {str(gate[0])}")
            
    return qc 