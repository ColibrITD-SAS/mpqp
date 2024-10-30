from mpqp.all import *

c = QCircuit(H.range(2) + [BasisMeasure(basis=HadamardBasis())])
print(run(c, IBMDevice.AER_SIMULATOR))
