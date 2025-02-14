import sys

import numpy as np

from mpqp import QCircuit
from mpqp.execution import IBMDevice, run
from mpqp.gates import *

n = int(sys.argv[1])
k = int(np.rint(np.pi * np.sqrt(2**n) / 4)) - 1
oracle = UnitaryMatrix(np.diag([1] * (2**n - 1) + [-1]))
diffusion = CustomGate(UnitaryMatrix(np.diag([-1] + [1] * (2**n - 1))), list(range(n)))
qaa = QCircuit(
    H.range(n)
    + sum(
        [
            [CustomGate(oracle, list(range(n)))] + H.range(n) + [diffusion]
            for _ in range(k)
        ],
        start=[],
    )
    + H.range(n)
)

print(run(qaa, IBMDevice.AER_SIMULATOR))
