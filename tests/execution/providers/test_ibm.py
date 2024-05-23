"""Add ``--long`` to the cli args to run this test (disabled by default because 
too slow)"""

import sys

# 3M-TODO test everything
import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import ExpectationMeasure, Observable
from mpqp.execution import run
from mpqp.execution.devices import IBMDevice
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
    run(circuit, IBMDevice.AER_SIMULATOR)


if "--long" in sys.argv:
    # in fact this is not slow anymore, because IBM disabled their remote
    # simulator, so this is a local one. Because of this, this test is not super
    # useful anymore. TODO: can we do better ?
    test_running_remote_IBM_without_error = running_remote_IBM_without_error
