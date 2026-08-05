from mpqp import *
from mpqp.core.circuit import BindingMode, CircuitBinding
from sympy import Symbol

theta, phi, psi = Symbol('θ'), Symbol('phi'), Symbol('psi')
c1 = QCircuit([U(theta, phi, psi, 0)], label="c1")
c2 = QCircuit([X(0)], label="c2")
c3 = QCircuit([H(0), CNOT(0, 1)], label="c3")
c4 = QCircuit([H(0), H(1), CNOT(0, 1)], label="c4")

v1 = {'θ': 1.0, 'phi': 1.0, 'psi': 1.0}
v2 = {'θ': 2.0, 'phi': 2.0, 'psi': 2.0}
v3 = {'θ': 3.0, 'phi': 3.0, 'psi': 3.0}
v4 = {'θ': 4.0, 'phi': 4.0, 'psi': 4.0}

m1 = ExpectationMeasure(Observable(pI), label="Exp1")
m2 = ExpectationMeasure(Observable(pX @ pZ), label="Exp2")
m3 = BasisMeasure(label="b3", shots=1024)
m4 = None

m_I = ExpectationMeasure(Observable(pI), label="Exp_I")
m_Z = ExpectationMeasure(Observable(pZ), label="Exp_Z")

cb_zip = CircuitBinding(
        circuits=c1, values=[v1, v2, v3], measurements=m1, mode=BindingMode.ZIP
    )
print(cb_zip)
