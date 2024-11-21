"""Example 3: Using native gates"""

from mpqp import QCircuit, Language
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.execution import run
from mpqp.execution.devices import ATOSDevice, AWSDevice, GOOGLEDevice, IBMDevice
from mpqp.gates import *

# Declaration of the circuit with the right size
circuit = QCircuit(3, label="Test native gates")
# Constructing the circuit by adding gates and measurements
circuit.add([H(0), X(1), Y(2), Z(0), S(1), T(0)])
circuit.add([Rx(1.2324, 2), Ry(-2.43, 0), Rz(1.04, 1), Rk(-1, 1), P(-323, 2)])
circuit.add(U(1.2, 2.3, 3.4, 2))
circuit.add(SWAP(2, 0))
circuit.add([CNOT(0, 1), CRk(4, 2, 1), CZ(1, 2)])
circuit.add(TOF([0, 1], 2))

# no measure, we want the state vector

print(circuit)
print(circuit.to_other_language(Language.QASM2))
print(circuit.to_other_language(Language.QASM3))

result = run(
    circuit,
    [
        ATOSDevice.MYQLM_PYLINALG,
        IBMDevice.AER_SIMULATOR_STATEVECTOR,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
    ],
)
print(result)

circuit.add(BasisMeasure())

result = run(
    circuit,
    [
        ATOSDevice.MYQLM_PYLINALG,
        IBMDevice.AER_SIMULATOR_STATEVECTOR,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
    ],
)
print(result)
