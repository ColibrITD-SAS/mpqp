from cirq import Circuit

from mpqp.core.circuit import BindingMode, CircuitBinding
from mpqp.core.instruction.gates.native_gates import X
from mpqp.execution.devices import (
    AvailableDevice,
    IBMDevice,
    AWSDevice,
)

from mpqp import (
    CNOT,
    BasisMeasure,
    ExpectationMeasure,
    H,
    IBMDevice,
    Language,
    Observable,
    QCircuit,
    U,
    pI,
    pZ,
    pX,
    run,
)
from sympy import Symbol
from mpqp.execution.runner import run
from mpqp.execution.devices import IBMDevice, AWSDevice

theta, phi, psi = Symbol('θ'), Symbol('phi'), Symbol('psi')
c1 = QCircuit([U(theta, phi, psi, 0)], label="c1")
c2 = QCircuit([H(0)], label="c2")
c2_bis = QCircuit([X(0)], label="c2_bis")
c3 = QCircuit([H(0), CNOT(0, 1)], label="c3")
c4 = QCircuit([H(0), H(1), CNOT(0, 1)], label="c4")

v1 = {'θ': 1.0, 'phi': 1.0, 'psi': 1.0}
v2 = {'θ': 2.0, 'phi': 2.0, 'psi': 2.0}
v3 = {'θ': 3.0, 'phi': 3.0, 'psi': 3.0}
v4 = {'θ': 4.0, 'phi': 4.0, 'psi': 4.0}

m1 = ExpectationMeasure(Observable(pI), label="Exp1", shots=2024)
m2 = ExpectationMeasure(Observable(pX @ pZ), label="Exp2", shots=2024)
m3 = BasisMeasure(label="b3", shots=2024)
m4 = None

m_I = ExpectationMeasure(Observable(pI), label="Exp_I", shots=2024)
m_Z = ExpectationMeasure(Observable(pZ), label="Exp_Z", shots=2024)

binding_with_basismeasure = CircuitBinding(
    [c3 + QCircuit([m3]), c4 + QCircuit([m3])], values=[v1, v2]
)

print(run(binding_with_basismeasure, AWSDevice.BRAKET_LOCAL_SIMULATOR))
