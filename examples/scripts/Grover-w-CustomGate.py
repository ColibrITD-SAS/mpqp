"""Call this script with the desired number of qubits as input, for instance
``python Grover-w-CustomGate.py 4``. The tagged state is `11...1`, and you can
check that after the execution of Grover's algorithm, the last state is the most
likely to be measured (the probabilities are displayed as output of this
script).

This script come in complement with a notebook also tackling Grover's
algorithm, but it is done differently in order to highlight different ways to
use MPQP."""

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
