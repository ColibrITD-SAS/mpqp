import numpy as np
from braket.devices import LocalSimulator

from mpqp import QCircuit
from mpqp.core.instruction.measurement import ExpectationMeasure, Observable
from mpqp.execution import run
from mpqp.execution.devices import ATOSDevice, AWSDevice
from mpqp.gates import *
from mpqp.measures import BasisMeasure
from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit

device = LocalSimulator()


qasm_str = """OPENQASM 3.0;
include 'stdgates.inc';
qubit[2] q;
bit c;
h q[0];
cx q[0],q[1];
c[0] = measure q[0];
c[1] = measure q[1];"""


circuit = qasm3_to_braket_Circuit(qasm_str)
print(circuit)
result = device.run(circuit, shots=100).result()
print(f"Measurement counts: {result.measurement_counts}")  # pyright: ignore

#####################################################

# Declaration of the circuit with the right size
circuit = QCircuit(4)

# Constructing the circuit by adding gates
circuit.add(T(0))
circuit.add(CNOT(0, 1))
circuit.add(X(0))
circuit.add(H(1))
circuit.add(Z(2))
circuit.add(CZ(2, 1))
circuit.add(SWAP(2, 0))
circuit.add(CNOT(0, 2))
circuit.add(Ry(3.14 / 2, 2))
circuit.add(S(1))
circuit.add(H(3))
circuit.add(CNOT(1, 2))
circuit.add(Rx(3.14, 1))
circuit.add(CNOT(3, 0))
circuit.add(Rz(3.14, 0))
# Add measurement for all qubits (implicit target)
circuit.add(BasisMeasure(shots=2000))

res = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)
print(res)

#####################################################

obs = Observable(
    np.array(
        [
            [0.63, 0.5, 1, 1],
            [0.5, 0.82, 1, 1],
            [1, 1, 1, 0.33],
            [1, 1, 0.33, 0.3],
        ],
        dtype=float,
    )
)

# Declaration of the circuit with the right size
circuit = QCircuit(2, label="Observable test")
# Constructing the circuit by adding gates and measurements
circuit.add(H(0))
circuit.add(Rx(1.76, 1))
circuit.add(ExpectationMeasure(obs))

# Running the ideal computation on Braket and myQLM simulators, then retrieving the results
result = run(circuit, [AWSDevice.BRAKET_LOCAL_SIMULATOR, ATOSDevice.MYQLM_PYLINALG])
print(result)

#####################################################

# Declaration of the circuit with the right size
circuit = QCircuit(2, label="Observable test")
# Constructing the circuit by adding gates and measurements
circuit.add(H(0))
circuit.add(Rx(1.76, 1))
circuit.add(Ry(1.76, 1))
circuit.add(Rz(1.987, 0))

# Running the computation on myQLM and on Aer simulator, then retrieving the results
result = run(circuit, [AWSDevice.BRAKET_LOCAL_SIMULATOR, ATOSDevice.MYQLM_PYLINALG])
print(result)
