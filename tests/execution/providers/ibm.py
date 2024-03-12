"""Add ``-l`` or ``--long`` to the cli args to run this test (disable by default 
because too slow)"""

# 3M-TODO test everything
import numpy as np
import pytest
from mpqp.core.instruction.measurement import Observable, ExpectationMeasure
from mpqp.gates import *
from mpqp import QCircuit
from mpqp.measures import BasisMeasure
from mpqp.execution import run
from mpqp.execution.devices import IBMDevice

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
        # OBSERVABLE JOB
        QCircuit(
            [
                H(0),
                Rx(1.76, 1),
                ExpectationMeasure(
                    [0, 1],
                    observable=Observable(
                        np.array(
                            [
                                [0.63, 0.5, 1, 1],
                                [0.5, 0.82, 1, 1],
                                [1, 1, 1, 0.33],
                                [1, 1, 0.33, 0.3],
                            ],
                            dtype=float,
                        )
                    ),
                    shots=1000,
                ),
            ]
        ),
    ],
)
def running_remote_IBM_without_error(circuit: QCircuit):
    run(circuit, IBMDevice.IBMQ_QASM_SIMULATOR)


if "-l" in sys.argv or "--long" in sys.argv:
    test_running_remote_IBM_without_error = running_remote_IBM_without_error
