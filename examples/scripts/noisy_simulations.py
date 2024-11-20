from mpqp import QCircuit
from mpqp.measures import BasisMeasure
from mpqp.gates import *
from mpqp.noise import *
from mpqp.execution import *


circuit = QCircuit(
    [Rx(0.3, 2), H(0), CNOT(1, 0), SWAP(2, 1), U(0.9, 0.2, 1, 1), BasisMeasure()]
)

circuit.add(Depolarizing(0.2))  # Depolarizing on all qubits
circuit.add(
    BitFlip(0.12, targets=[0, 1], gates=[H, Rz, U])
)  # BitFlip on qubit 0 and 1, only on gates H, Z, X,U (only H and U will be affected in the circuit)
circuit.add(AmplitudeDamping(0.2, 0.9, targets=[2]))
circuit.add(PhaseDamping(0.1, gates=[SWAP]))

print(run(circuit, [IBMDevice.AER_SIMULATOR, AWSDevice.BRAKET_LOCAL_SIMULATOR]))
