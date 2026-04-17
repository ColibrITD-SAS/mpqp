import pytest

from mpqp import (
    BasisMeasure,
    ComputationalBasis,
    run,
    QCircuit,
    IBMDevice,
    ATOSDevice,
    AWSDevice,
    GOOGLEDevice,
)
from mpqp.core.instruction.gates.native_gates import X
from mpqp.execution.devices import AvailableDevice


def test_basis_measure_init():
    measure = BasisMeasure([0, 1], shots=1025, basis=ComputationalBasis())
    assert measure.targets == [0, 1]
    assert measure.shots == 1025
    assert isinstance(measure.basis, ComputationalBasis)


def test_basis_measure_init_fails_duplicate_c_targets():
    with pytest.raises(ValueError, match="Duplicate registers in targets"):
        BasisMeasure(targets=[0, 1], c_targets=[2, 2, 3], shots=1024)


def test_basis_measure_repr():
    measure = BasisMeasure([0, 1], shots=1025)
    representation = repr(measure)
    assert representation == "BasisMeasure([0, 1], shots=1025)"


def qcircuit_basis_measure() -> list[tuple[QCircuit, list[int]]]:
    return [
        (
            QCircuit(
                [X(0), X(1), X(2), X(3), BasisMeasure([3], shots=1024)], nb_qubits=4
            ),
            [0, 1024],
        ),
        (
            QCircuit(
                [X(0), X(1), X(2), X(3), BasisMeasure([1], shots=1024)], nb_qubits=4
            ),
            [0, 1024],
        ),
        (
            QCircuit(
                [X(0), X(1), X(2), X(3), BasisMeasure([0, 1], shots=1024)], nb_qubits=4
            ),
            [0, 0, 0, 1024],
        ),
        (
            QCircuit(
                [X(0), X(1), X(2), X(3), BasisMeasure([2, 3], shots=1024)], nb_qubits=4
            ),
            [0, 0, 0, 1024],
        ),
        (
            QCircuit(
                [X(0), X(1), X(2), X(3), BasisMeasure([1, 2, 3], shots=1024)],
                nb_qubits=4,
            ),
            [0, 0, 0, 0, 0, 0, 0, 1024],
        ),
    ]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "qcircuit, result_count",
    qcircuit_basis_measure(),
)
def test_basis_measure_not_all_targets_qiskit(
    qcircuit: QCircuit, result_count: list[int]
):
    exec_basis_measure_not_all_targets(IBMDevice.AER_SIMULATOR, qcircuit, result_count)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize(
    "qcircuit, result_count",
    qcircuit_basis_measure(),
)
def test_basis_measure_not_all_targets_cirq(
    qcircuit: QCircuit, result_count: list[int]
):
    exec_basis_measure_not_all_targets(
        GOOGLEDevice.CIRQ_LOCAL_SIMULATOR, qcircuit, result_count
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "qcircuit, result_count",
    qcircuit_basis_measure(),
)
def test_basis_measure_not_all_targets_braket(
    qcircuit: QCircuit, result_count: list[int]
):
    exec_basis_measure_not_all_targets(
        AWSDevice.BRAKET_LOCAL_SIMULATOR, qcircuit, result_count
    )


@pytest.mark.provider("atos")
@pytest.mark.parametrize(
    "qcircuit, result_count",
    qcircuit_basis_measure(),
)
def test_basis_measure_not_all_targets_atos(
    qcircuit: QCircuit, result_count: list[int]
):
    exec_basis_measure_not_all_targets(ATOSDevice.MYQLM_CLINALG, qcircuit, result_count)


def exec_basis_measure_not_all_targets(
    provider: AvailableDevice, qcircuit: QCircuit, result_count: list[int]
):
    result = run(qcircuit, provider)
    assert result.counts == result_count
