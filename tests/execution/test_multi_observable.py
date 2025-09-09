import numpy as np
import pytest

from mpqp.core import QCircuit
from mpqp.core.instruction import ExpectationMeasure, Observable
from mpqp.execution import AvailableDevice, IBMDevice
from mpqp.execution.devices import ATOSDevice, AWSDevice, GOOGLEDevice
from mpqp.execution.runner import run

from mpqp.gates import *


def list_circuits():
    return [
        QCircuit([H(0), CNOT(0, 1)]),
        QCircuit([H(0), X(1)]),
        QCircuit([Rx(np.pi / 2, 0), Ry(np.pi / 5, 1), CNOT(0, 1)]),
        # TODO add random circuit
    ]


def list_observables():
    return [
        [
            Observable(np.ones((4, 4), dtype=np.complex128)),
            Observable(np.diag([1, 2, -3, 4])),
            # TODO add random observable ?
        ]
    ]


def list_devices():
    return [IBMDevice.AER_SIMULATOR]


@pytest.mark.parametrize(
    "circuit, observables, device",
    [
        (i, j, k)
        for i in list_circuits()
        for j in list_observables()
        for k in list_devices()
    ],
)
def test_sequential_versus_multi(
    circuit: QCircuit, observables: list[Observable], device: AvailableDevice
):
    seq_results = [
        run(
            circuit
            + QCircuit([ExpectationMeasure(obs, shots=0)], nb_qubits=circuit.nb_qubits),
            device,
        )
        for obs in observables
    ]

    multi_result = run(
        circuit
        + QCircuit(
            [ExpectationMeasure(observables, shots=0)], nb_qubits=circuit.nb_qubits
        ),
        device,
    )
    expectation_values = multi_result.expectation_values
    assert isinstance(expectation_values, dict)
    print(multi_result)
    print(observables)
    assert len(seq_results) == len(expectation_values)

    # TODO modify here to match the logic of dict and observable.label etc
    for r1, e2 in zip(seq_results, expectation_values.values()):
        assert r1.expectation_values == e2


from mpqp.measures import PX, PY, PZ, PI


def pauliObservables():
    return [
        [Observable(PX @ PX @ PX + PI @ PX @ PI + PX @ PI @ PX - 2 * PZ @ PZ @ PZ)],
        [
            Observable(PX @ PX @ PX + PI @ PX @ PI + PX @ PI @ PX - 2 * PZ @ PZ @ PZ),
            Observable(PY @ PY @ PY + PX @ PX @ PX),
        ],
        [
            Observable(PX @ PX @ PX + PI @ PX @ PI + PX @ PI @ PX - 2 * PZ @ PZ @ PZ),
            Observable(PY @ PY @ PY + PX @ PX @ PX),
            Observable(PZ @ PI @ PZ - 5 * PX @ PX @ PX),
        ],
    ]


def optimized_devices():
    return [
        IBMDevice.AER_SIMULATOR,
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
    ]


@pytest.mark.parametrize(
    "observable, device",
    [(i, j) for i in pauliObservables() for j in optimized_devices()],
)
def test_pauli_grouping_optimization(
    observable: list[Observable], device: AvailableDevice
):
    from mpqp.execution import run, Result

    circuit = QCircuit(
        [
            Rx(np.pi / 3, 0),
            Ry(np.pi / 12, 1),
            CNOT(0, 1),
            Rz(np.pi / 6, 1),
            Ry(1.5 * np.pi, 2),
        ]
    )
    if device.supports_state_vector():
        non_optimized = run(
            circuit
            + QCircuit([ExpectationMeasure(observable, optimize_measurement=False)]),
            device,
        )
        optimized = run(
            circuit
            + QCircuit([ExpectationMeasure(observable, optimize_measurement=True)]),
            device,
        )
        assert isinstance(non_optimized, Result)
        assert isinstance(optimized, Result)
        optimized_expectation_values = optimized.expectation_values
        if isinstance(non_optimized.expectation_values, float) and isinstance(
            optimized_expectation_values, float
        ):
            assert round(non_optimized.expectation_values, 10) == round(
                optimized_expectation_values, 10
            )
        else:
            for i in range(len(non_optimized.expectation_values)):
                assert round(
                    non_optimized.expectation_values[f"observable_{i}"], 5
                ) == round(optimized_expectation_values[f"observable_{i}"], 5)
    else:
        assert True


@pytest.mark.parametrize(
    "expectation_value,circuit,observable",
    [
        [
            -0.2279775,
            QCircuit([Rx(0.23, 0), Rz(24.23, 1), CNOT(0, 1)]),
            Observable(PX @ PY + PZ @ PX),
        ],
        [
            0,
            QCircuit([Rx(0.23, 0), Rz(24.23, 1), CNOT(0, 1)]),
            Observable(PI @ PX + PY @ PZ),
        ],
    ],
)
def test_expectation_value_all_devices(
    expectation_value: float, circuit: QCircuit, observable: Observable
):
    devices = [
        IBMDevice.AER_SIMULATOR,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
        ATOSDevice.MYQLM_PYLINALG,
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
    ]
    circuit.add(ExpectationMeasure(observable, shots=0, optimize_measurement=True))

    assert all(
        round(  # pyright: ignore[reportCallIssue]
            run(
                circuit, device
            ).expectation_values,  # pyright: ignore[reportArgumentType]
            7,
        )
        == expectation_value
        for device in devices
    )
