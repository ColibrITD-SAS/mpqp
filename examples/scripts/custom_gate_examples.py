"""CustomGate examples"""

import numpy as np

from mpqp import Language, QCircuit
from mpqp.execution import *
from mpqp.gates import *

unitary = UnitaryMatrix(
    np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
)
custom_gate = CustomGate(unitary, [1, 2], label="CG1")

circuit = QCircuit([H(0), Rz(1.2, 1), Rx(3.2, 2), CNOT(0, 1), custom_gate])
print(circuit)
print(
    run(
        circuit,
        [
            IBMDevice.AER_SIMULATOR,
            ATOSDevice.MYQLM_PYLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        ],
    )
)

print(circuit.to_other_language(Language.QISKIT))
print(circuit.to_other_language(Language.QASM2))
print(circuit.to_other_language(Language.QASM3))
print(circuit.gphase)
