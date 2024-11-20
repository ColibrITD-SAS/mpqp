from typing import Any, Callable

import numpy as np
import pytest
from braket.devices import LocalSimulator

from mpqp import QCircuit
from mpqp.core.instruction.measurement import ExpectationMeasure, Observable
from mpqp.core.languages import Language
from mpqp.execution import run
from mpqp.execution.devices import ATOSDevice, AvailableDevice, AWSDevice, IBMDevice
from mpqp.gates import *
from mpqp.measures import BasisMeasure
from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning

# TODO: add CIRQ local simulator devices to this file


def warn_guard(device: AvailableDevice, run: Callable[[], Any]):
    if isinstance(device, AWSDevice):
        with pytest.warns(UnsupportedBraketFeaturesWarning):
            return run()
    else:
        return run()


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
    runner = lambda: run(
        circuit,
        [
            IBMDevice.AER_SIMULATOR,
            IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
            # IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER,
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            # IBMDevice.AER_SIMULATOR_STABILIZER,
            IBMDevice.AER_SIMULATOR_DENSITY_MATRIX,
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ],
    )

    warn_guard(AWSDevice.BRAKET_LOCAL_SIMULATOR, runner)

    assert True


def test_sample_demo_aer_stabilizers():
    # Declaration of the circuit with the right size
    circuit = QCircuit(4)

    # Constructing the circuit by adding gates
    circuit.add(CNOT(0, 1))
    circuit.add(X(0))
    circuit.add(H(1))
    circuit.add(Z(2))
    circuit.add(CZ(2, 1))
    circuit.add([SWAP(2, 0), CNOT(0, 2)])
    circuit.add(S(1))
    circuit.add(H(3))
    circuit.add(CNOT(1, 2))
    circuit.add(CNOT(3, 0))

    # Add measurement
    circuit.add(BasisMeasure([0, 1, 2, 3], shots=2000))

    # Run the circuit on a selected device
    run(
        circuit,
        [
            IBMDevice.AER_SIMULATOR,
            IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER,
            IBMDevice.AER_SIMULATOR_STABILIZER,
        ],
    )
    assert True


def test_statevector_demo():
    circuit = QCircuit(
        [
            T(0),
            CNOT(0, 1),
            X(0),
            H(1),
            Z(2),
            CZ(2, 1),
            SWAP(2, 0),
            CNOT(0, 2),
            Ry(1.7, 2),
            S(1),
            H(3),
            CNOT(1, 2),
            Rx(3.14, 1),
            CNOT(3, 0),
            Rz(3.14, 0),
        ]
    )

    # when no measure in the circuit, must run in statevector mode
    runner = lambda: run(
        circuit,
        [
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            IBMDevice.AER_SIMULATOR,
            IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ],
    )

    warn_guard(AWSDevice.BRAKET_LOCAL_SIMULATOR, runner)

    # same when we add a BasisMeasure with 0 shots
    circuit.add(BasisMeasure([0, 1, 2, 3], shots=0))

    # Run the circuit on a selected device
    runner = lambda: run(
        circuit,
        [
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            IBMDevice.AER_SIMULATOR,
            IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ],
    )

    warn_guard(AWSDevice.BRAKET_LOCAL_SIMULATOR, runner)

    assert True


def test_statevector_demo_stab():
    # Declaration of the circuit with the right size
    circuit = QCircuit(4)

    # Constructing the circuit by adding gates
    circuit.add(S(0))
    circuit.add(CNOT(0, 1))
    circuit.add(X(0))
    circuit.add(H(1))
    circuit.add(Z(2))
    circuit.add(CZ(2, 1))
    circuit.add([SWAP(2, 0), CNOT(0, 2)])
    circuit.add(S(1))
    circuit.add(H(3))
    circuit.add(CNOT(3, 0))

    # when no measure in the circuit, must run in statevector mode
    run(circuit, IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE)

    assert True


@pytest.mark.parametrize("shots", [0, 1000])
def test_observable_demo(shots: int):
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
    circuit.add([H(1), CNOT(1, 0)])
    circuit.add(ExpectationMeasure(obs, shots=shots))

    # Running the computation on myQLM and on Aer simulator, then retrieving the results
    runner = lambda: run(
        circuit,
        [
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            IBMDevice.AER_SIMULATOR,
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            # IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER,
            # IBMDevice.AER_SIMULATOR_STABILIZER,
            IBMDevice.AER_SIMULATOR_DENSITY_MATRIX,
            IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
        ],
    )

    warn_guard(AWSDevice.BRAKET_LOCAL_SIMULATOR, runner)

    assert True


def test_aws_qasm_executions():
    device = LocalSimulator()

    qasm_str = """OPENQASM 3.0;
    include 'stdgates.inc';
    qubit[2] q;
    bit c;
    h q[0];
    cx q[0],q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];"""

    runner = lambda: qasm3_to_braket_Circuit(qasm_str)
    circuit = warn_guard(AWSDevice.BRAKET_LOCAL_SIMULATOR, runner)
    device.run(circuit, shots=100).result()


def test_aws_mpqp_executions():
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

    runner = lambda: run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    warn_guard(AWSDevice.BRAKET_LOCAL_SIMULATOR, runner)

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
    circuit.add(ExpectationMeasure(obs, shots=0))

    # Running the computation on myQLM and on Braket simulator, then retrieving the results
    runner = lambda: run(
        circuit, [AWSDevice.BRAKET_LOCAL_SIMULATOR, ATOSDevice.MYQLM_PYLINALG]
    )
    warn_guard(AWSDevice.BRAKET_LOCAL_SIMULATOR, runner)

    #####################################################

    # Declaration of the circuit with the right size
    circuit = QCircuit(
        [H(0), Rx(1.76, 1), Ry(1.76, 1), Rz(1.987, 0)], label="StateVector test"
    )

    # Running the computation on myQLM and on Aer simulator, then retrieving the results
    runner = lambda: run(
        circuit, [AWSDevice.BRAKET_LOCAL_SIMULATOR, ATOSDevice.MYQLM_PYLINALG]
    )
    warn_guard(AWSDevice.BRAKET_LOCAL_SIMULATOR, runner)


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

    circuit.to_other_language(Language.QASM2)
    circuit.to_other_language(Language.QASM3, translation_warning=False)
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        run(
            circuit,
            [
                ATOSDevice.MYQLM_PYLINALG,
                ATOSDevice.MYQLM_CLINALG,
                IBMDevice.AER_SIMULATOR_STATEVECTOR,
                AWSDevice.BRAKET_LOCAL_SIMULATOR,
            ],
        )
