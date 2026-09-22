from mpqp import *
from mpqp.core.circuitbinding import CircuitBinding, BindingMode

cb = CircuitBinding(
    [QCircuit([X(0), BasisMeasure()]), QCircuit([Z(0), BasisMeasure()])], shots=1
)

print(run(cb, AWSDevice.IQM_EMERALD))
