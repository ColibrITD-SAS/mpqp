import numpy as np
import scipy.linalg as la

from mpqp import Barrier, QCircuit
from mpqp.gates import *
from mpqp.measures import BasisMeasure

# U_op = rand_unitary_2x2_matrix()
U_op = Y(0).to_matrix()
i2 = np.eye(2, dtype=np.complex64)

n = 5

controlled_u = UnitaryMatrix(la.block_diag(i2, U_op))

qft = QCircuit(
    sum(
        [
            [H(i)] + [CRk(j + 1, i + j, i) for j in range(1, n - i)] + [Barrier()]
            for i in range(n)
        ],
        start=[],
    )
)

qpe = (
    QCircuit(
        H.range(n)
        + [CustomGate(controlled_u, [n - i - 1, n]).power(2**i) for i in range(n)]
    )
    + qft.inverse()
    + QCircuit([BasisMeasure()])
)
