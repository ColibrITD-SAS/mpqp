import pytest
from mpqp import QCircuit
from mpqp.core.instruction.measurement import Observable, ExpectationMeasure
from mpqp.execution.devices import AWSDevice, ATOSDevice, IBMDevice
from mpqp.gates import *
from mpqp.measures import BasisMeasure
from mpqp.execution import run
from braket.devices import LocalSimulator
import numpy as np
from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit


def test_sample_demo():
    # Declaration of the circuit with the right size
    circuit = QCircuit(4)

    # Constructing the circuit by adding gates
    circuit.add(T(0))
    circuit.add(CNOT(0, 1))
    circuit.add(X(0))
    circuit.add(H(1))
    circuit.add(Z(2))
    circuit.add(CZ(2, 1))
    circuit.add([SWAP(2, 0), CNOT(0, 2)])
    circuit.add(Ry(3.14 / 2, 2))
    circuit.add(S(1))
    circuit.add(H(3))
    circuit.add(CNOT(1, 2))
    circuit.add(Rx(3.14, 1))
    circuit.add(CNOT(3, 0))
    circuit.add(Rz(3.14, 0))

    # Add measurement
    circuit.add(BasisMeasure([0, 1, 2, 3], shots=2000))

    # Run the circuit on a selected device
    run(
        circuit,
        [
            IBMDevice.AER_SIMULATOR,
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ],
    )

    assert True


def test_statevector_demo():
    # Declaration of the circuit with the right size
    circuit = QCircuit(4)

    # Constructing the circuit by adding gates
    circuit.add(T(0))
    circuit.add(CNOT(0, 1))
    circuit.add(X(0))
    circuit.add(H(1))
    circuit.add(Z(2))
    circuit.add(CZ(2, 1))
    circuit.add([SWAP(2, 0), CNOT(0, 2)])
    circuit.add(Ry(3.14 / 2, 2))
    circuit.add(S(1))
    circuit.add(H(3))
    circuit.add(CNOT(1, 2))
    circuit.add(Rx(3.14, 1))
    circuit.add(CNOT(3, 0))
    circuit.add(Rz(3.14, 0))

    # when no measure in the circuit, must run in statevector mode
    run(
        circuit,
        [
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ],
    )

    # same when we add a BasisMeasure with 0 shots
    circuit.add(BasisMeasure([0, 1, 2, 3], shots=0))

    # Run the circuit on a selected device
    run(
        circuit,
        [
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ],
    )

    assert True


def test_observable_demo():
    obs = Observable(
        np.array(
            [
                [0.63, 0.5, 1, 1],
                [0.5, 0.82, 1, 1],
                [1, 1, 1, 0.33],
                [1, 1, 0.33, 0.3],
            ],
            dtype=float,
        )
    )

    # Declaration of the circuit with the right size
    circuit = QCircuit(2, label="Observable test")
    # Constructing the circuit by adding gates and measurements
    circuit.add(H(0))
    circuit.add(Rx(1.76, 1))
    circuit.add(ExpectationMeasure([0, 1], observable=obs, shots=0))

    # Running the computation on myQLM and on Aer simulator, then retrieving the results
    run(circuit, [ATOSDevice.MYQLM_PYLINALG, IBMDevice.AER_SIMULATOR])

    assert True


def test_aws_executions():
    device = LocalSimulator()

    qasm_str = """OPENQASM 3.0;
    include 'stdgates.inc';
    qubit[2] q;
    bit c;
    h q[0];
    cx q[0],q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];"""

    circuit = qasm3_to_braket_Circuit(qasm_str)

    device.run(circuit, shots=100).result()

    #####################################################

    # Declaration of the circuit with the right size
    circuit = QCircuit(4)

    # Constructing the circuit by adding gates
    circuit.add(T(0))
    circuit.add(CNOT(0, 1))
    circuit.add(X(0))
    circuit.add(H(1))
    circuit.add(Z(2))
    circuit.add(CZ(2, 1))
    circuit.add(SWAP(2, 0))
    circuit.add(CNOT(0, 2))
    circuit.add(Ry(3.14 / 2, 2))
    circuit.add(S(1))
    circuit.add(H(3))
    circuit.add(CNOT(1, 2))
    circuit.add(Rx(3.14, 1))
    circuit.add(CNOT(3, 0))
    circuit.add(Rz(3.14, 0))
    # Add measurement
    circuit.add(BasisMeasure([0, 1, 2, 3], shots=2000))

    run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    #####################################################

    obs = Observable(
        np.array(
            [
                [0.63, 0.5, 1, 1],
                [0.5, 0.82, 1, 1],
                [1, 1, 1, 0.33],
                [1, 1, 0.33, 0.3],
            ],
            dtype=float,
        )
    )

    # Declaration of the circuit with the right size
    circuit = QCircuit(2, label="Observable test")
    # Constructing the circuit by adding gates and measurements
    circuit.add(H(0))
    circuit.add(Rx(1.76, 1))
    circuit.add(ExpectationMeasure([0, 1], observable=obs, shots=0))

    # Running the computation on myQLM and on Braket simulator, then retrieving the results
    run(circuit, [AWSDevice.BRAKET_LOCAL_SIMULATOR, ATOSDevice.MYQLM_PYLINALG])

    #####################################################

    # Declaration of the circuit with the right size
    circuit = QCircuit(
        [H(0), Rx(1.76, 1), Ry(1.76, 1), Rz(1.987, 0)], label="StateVector test"
    )

    # Running the computation on myQLM and on Aer simulator, then retrieving the results
    run(circuit, [AWSDevice.BRAKET_LOCAL_SIMULATOR, ATOSDevice.MYQLM_PYLINALG])


def test_all_native_gates():
    # Declaration of the circuit with the right size
    circuit = QCircuit(3, label="Test native gates")
    # Constructing the circuit by adding gates and measurements
    circuit.add([H(0), X(1), Y(2), Z(0), S(1), T(0)])
    circuit.add([Rx(1.2324, 2), Ry(-2.43, 0), Rz(1.04, 1), Rk(-1, 1), P(-323, 2)])
    circuit.add(U(1.2, 2.3, 3.4, 2))
    circuit.add(SWAP(2, 0))
    circuit.add([CNOT(0, 1), CRk(4, 2, 1), CZ(1, 2)])
    circuit.add(TOF([0, 1], 2))

    circuit.to_qasm2()
    with pytest.warns(
        UserWarning,
        match=r"There is a phase e\^\(i\(a\+c\)/2\) difference between U\(a,b,c\) gate in 2.0 and 3.0.",
    ):
        circuit.to_qasm3()
        run(
            circuit,
            [
                ATOSDevice.MYQLM_PYLINALG,
                ATOSDevice.MYQLM_CLINALG,
                IBMDevice.AER_SIMULATOR_STATEVECTOR,
                AWSDevice.BRAKET_LOCAL_SIMULATOR,
            ],
        )
