"""add -l or --long to the cli args to run this test (disabled by default 
because too slow)"""

# 3M-TODO test everything
import numpy as np
import pytest

from mpqp.core.instruction.measurement import Observable, ExpectationMeasure
from mpqp.gates import *
from mpqp import QCircuit
from mpqp.measures import BasisMeasure
from mpqp.execution import run
from mpqp.execution.devices import GOOGLEDevice

import sys


@pytest.mark.parametrize(
    "circuit",
    [
        # SAMPLE JOB
        QCircuit(
            [
                T(0),
                CNOT(0, 1),
                Ry(np.pi / 2, 2),
                S(1),
                CZ(2, 1),
                SWAP(2, 0),
                BasisMeasure(list(range(3)), shots=2000),
            ]
        ),
        # STATEVECTOR JOB
        QCircuit(
            [
                T(0),
                CNOT(0, 1),
                Ry(np.pi / 2, 2),
                S(1),
                CZ(2, 1),
                SWAP(2, 0),
                BasisMeasure(list(range(3)), shots=0),
            ]
        ),
    ],
)
def running_remote_local_cirq(circuit: QCircuit):
    result = run(circuit, GOOGLEDevice.CIRQ)


if "-l" in sys.argv or "--long" in sys.argv:
    test_running_local_cirq_without = running_remote_local_cirq
