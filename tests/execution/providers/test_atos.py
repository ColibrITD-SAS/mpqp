"""Add ``--long`` to the cli args to run this test (disabled by default because 
too slow)"""

import sys

# 3M-TODO test everything
import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import ExpectationMeasure, Observable
from mpqp.execution import run
from mpqp.execution.devices import ATOSDevice
from mpqp.gates import *
from mpqp.measures import BasisMeasure


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
        # OBSERVABLE JOB
        QCircuit(
            [
                H(0),
                Rx(1.76, 1),
                ExpectationMeasure(
                    observable=Observable(
                        np.array(
                            [
                                [0.63, 0.5, 1, 1],
                                [0.5, 0.82, 1, 1],
                                [1, 1, 1, 0.33],
                                [1, 1, 0.33, 0.3],
                            ],
                        )
                    ),
                    targets=[0, 1],
                    shots=1000,
                ),
            ]
        ),
    ],
)
def running_remote_QLM_without_error(circuit: QCircuit):
    run(circuit, ATOSDevice.QLM_LINALG)


if "--long" in sys.argv:
    test_running_remote_QLM_without_error = running_remote_QLM_without_error
