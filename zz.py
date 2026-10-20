from typing import Any

from mpqp import *

c3 = QCircuit([H(0), CNOT(0, 1)], label="c3")
c4 = QCircuit([H(0), H(1), CNOT(0, 1)], label="c4")
v1 = {"θ": 1.0, "phi": 1.0, "psi": 1.0}
v2 = {"θ": 2.0, "phi": 2.0, "psi": 2.0}
cb = CircuitBinding(
    [CircuitBinding(c3), CircuitBinding(c4)],
    values=[v1, v2],
    mode=BindingMode.ZIP,
)

print(run(cb, AWSDevice.BRAKET_LOCAL_SIMULATOR))
