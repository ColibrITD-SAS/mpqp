import sys

import pytest

from mpqp import QCircuit
from mpqp.execution import run
from mpqp.execution.simulated_devices import IBMSimulatedDevice
from mpqp.gates import *
from mpqp.measures import BasisMeasure, ExpectationMeasure, Observable
from mpqp.tools import DeviceJobIncompatibleError
from mpqp.tools.maths import rand_hermitian_matrix


def circuits():
    return [
        QCircuit(
            [
                H(0),
                X(1),
                Y(2),
                Z(0),
                S(1),
                T(0),
                Rx(1.2324, 2),
                Ry(-2.43, 0),
                Rz(1.04, 1),
                Rk(-1, 0),
                P(-323, 2),
                U(1.2, 2.3, 3.4, 2),
                SWAP(2, 1),
                CNOT(0, 1),
                CZ(1, 2),
                CRk(4, 2, 0),
            ]
        ),
        QCircuit([H(0)]),
    ]


def ibm_simulated_devices():
    return list(IBMSimulatedDevice)


def test_generation_enum():
    assert len(list(IBMSimulatedDevice)) > 0


@pytest.mark.parametrize(
    "circuit, device", [(i, j) for i in circuits() for j in ibm_simulated_devices()]
)
def running_sample_job_ibm_simulated_devices(
    circuit: QCircuit, device: IBMSimulatedDevice
):
    c = circuit + QCircuit([BasisMeasure()], nb_qubits=circuit.nb_qubits)
    if device.value().num_qubits < c.nb_qubits:
        with pytest.raises(DeviceJobIncompatibleError):
            run(c, device)
    else:
        run(c, device)
    assert True


@pytest.mark.parametrize(
    "circuit, device", [(i, j) for i in circuits() for j in ibm_simulated_devices()]
)
def running_observable_job_ibm_simulated_devices(
    circuit: QCircuit, device: IBMSimulatedDevice
):
    c = circuit + QCircuit(
        [ExpectationMeasure(Observable(rand_hermitian_matrix(2**circuit.nb_qubits)))],
        nb_qubits=circuit.nb_qubits,
    )
    if device.value().num_qubits < c.nb_qubits:
        with pytest.raises(DeviceJobIncompatibleError):
            run(c, device)
    else:
        run(c, device)
    assert True


if "--long-local" in sys.argv:
    test_running_sample_job_ibm_simulated_devices = (
        running_sample_job_ibm_simulated_devices
    )
    test_running_observable_job_ibm_simulated_devices = (
        running_observable_job_ibm_simulated_devices
    )
