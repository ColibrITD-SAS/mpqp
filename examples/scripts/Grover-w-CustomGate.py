import math
import sys

import numpy as np

from mpqp import QCircuit
from mpqp.execution import IBMDevice, run
from mpqp.gates import *

n = int(sys.argv[1])
k = math.floor(math.pi / (4 * math.asin(math.sqrt(1 / 2**n))))
oracle = UnitaryMatrix(np.diag([1] * (2**n - 1) + [-1]))
diffusion = CustomGate(UnitaryMatrix(np.diag([-1] + [1] * (2**n - 1))), list(range(n)))
qaa = QCircuit(
    H.range(n)
    + sum(
        [
            [CustomGate(oracle, list(range(n)))] + H.range(n) + [diffusion] + H.range(n)
            for _ in range(k)
        ],
        start=[],
    )
)

print(run(qaa, IBMDevice.AER_SIMULATOR))
