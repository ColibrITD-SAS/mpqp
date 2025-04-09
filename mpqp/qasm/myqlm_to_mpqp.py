from __future__ import annotations

from mpqp import QCircuit
from mpqp.core.instruction.gates import *
from qat.core.wrappers.circuit import Circuit as my_QLM_Circuit


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
        if gate[0] == 'CCNOT':
            qc.add(TOF([gate[2][0], gate[2][1]], gate[2][2]))

        elif gate[0] in one_qubit_gates:
            idx = 0
            for idx in range(len(one_qubit_gates)):
                if gate[0] == one_qubit_gates[idx]:
                    break
            qc.add(mpqp_one_qubit_gates[idx](gate[2][0]))

        elif gate[0] in two_qubits_gates:
            for idx in range(len(two_qubits_gates)):
                if gate[0] == two_qubits_gates[idx]:
                    break
            qc.add(mpqp_two_qubits_gates[idx](gate[1][0], gate[2][0]))

        elif gate[0] in controlled_gates:
            for idx in range(len(controlled_gates)):
                if gate[0] == controlled_gates[idx]:
                    break
            qc.add(mpqp_controlled_gates[idx](gate[2][0], gate[2][1]))

        else:
            if gate[0] == 'U':
                qc.add(U(gate[1][0], gate[1][1], gate[1][2], gate[2][0]))
            elif gate[0] == 'U1':
                qc.add(P(gate[1][0], gate[2][0]))
            elif gate[0] == 'ISWAP':
                qc.add(S(0))
                qc.add(S(1))
                qc.add(H(0))
                qc.add(CNOT(0, 1))
                qc.add(CNOT(1, 0))
                qc.add(H(1))
            elif gate[0] == 'SQRTSWAP':
                qc.add(CNOT(0, 1))
                qc.add(H(0))
                qc.add(CP(np.pi/2, 1, 0))
                qc.add(H(0))
                qc.add(CNOT(0, 1))
            elif gate[0] == 'MEASURE':
                break

            else:
                raise SyntaxError(f"Unknown Gate: {str(gate[0])}")
            
    return qc 