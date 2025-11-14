import contextlib

import numpy as np
import numpy.typing as npt
import pytest

from mpqp import (
    ATOSDevice,
    AWSDevice,
    Basis,
    BasisMeasure,
    ComputationalBasis,
    GOOGLEDevice,
    HadamardBasis,
    IBMDevice,
    QCircuit,
    Result,
    VariableSizeBasis,
    run,
)
from mpqp.execution import AvailableDevice
from mpqp.gates import *
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning
from mpqp.tools.maths import matrix_eq


@pytest.mark.parametrize(
    "init_vectors, size",
    [
        ([np.array([1, 0]), np.array([0, -1])], 1),
    ],
)
def test_right_init_basis(init_vectors: list[npt.NDArray[np.complex128]], size: int):
    b = Basis(init_vectors)
    assert b.nb_qubits == size
    assert (b.basis_vectors[i] == init_vectors[i] for i in range(len(init_vectors)))


@pytest.mark.parametrize(
    "init_vectors, part_of_error",
    [
        ([np.array([1, 0]), np.array([0, -1]), np.array([0, -1])], "number of vector"),
        ([np.array([1, 0])], "number of vector"),
        ([np.array([1, 0, 0]), np.array([0, -1])], "same size"),
        ([np.array([1, 1]), np.array([1, -1])], "orthogonal"),
        ([np.array([1, 1]) / np.sqrt(2), np.array([0, -1])], "orthogonal"),
    ],
)
def test_wrong_init_basis(
    init_vectors: list[npt.NDArray[np.complex128]], part_of_error: str
):
    with pytest.raises(ValueError) as error:
        Basis(init_vectors)
    assert part_of_error in error.exconly()


@pytest.mark.parametrize(
    "basis_vectors, size, result_pp",
    [
        (
            [
                np.array([1, 0, 0, 0]),
                np.array([0, -1, 0, 0]),
                np.array([0, 0, -1, 0]),
                np.array([0, 0, 0, 1]),
            ],
            4,
            (
                "Basis: [\n"
                "    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]\n"
                "]\n"
            ),
        ),
        (
            [np.array([-1, 0]), np.array([0, -1])],
            2,
            (
                "Basis: [\n"
                "    [1, 0, 0, 0],\n"
                "    [0, 1, 0, 0],\n"
                "    [0, 0, 1, 0],\n"
                "    [0, 0, 0, 1]\n"
                "]\n"
            ),
        ),
        (
            [np.array([1, 0]), np.array([0, -1])],
            3,
            (
                "Basis: [\n"
                "    [1, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, -1, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, -1, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 1, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, -1, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 1, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 1, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, -1]\n"
                "]\n"
            ),
        ),
    ],
)
def test_variable_size_basis(
    basis_vectors: list[npt.NDArray[np.complex128]],
    size: int,
    result_pp: str,
    capsys: pytest.CaptureFixture[str],
):
    b = VariableSizeBasis(basis_vectors)
    b.set_size(size)
    b.pretty_print()
    captured = capsys.readouterr()
    assert captured.out == result_pp


@pytest.mark.parametrize(
    "basis_vectors, size",
    [
        (
            [
                np.array([1, 0, 0, 0]),
                np.array([0, -1, 0, 0]),
                np.array([0, 0, -1, 0]),
                np.array([0, 0, 0, 1]),
            ],
            3,
        ),
        (
            [
                np.array([1, 0, 0, 0]),
                np.array([0, -1, 0, 0]),
                np.array([0, 0, -1, 0]),
                np.array([0, 0, 0, 1]),
            ],
            5,
        ),
        (
            [
                np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, -1, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, -1, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 1, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, -1, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 1, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 1, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, -1]),
            ],
            4,
        ),
    ],
)
def test_value_error_variable_size_basis(
    basis_vectors: list[npt.NDArray[np.complex128]],
    size: int,
):
    b = VariableSizeBasis(basis_vectors)
    with pytest.raises(ValueError):
        b.set_size(size)


@pytest.mark.parametrize(
    "basis, size, result_pp",
    [
        (
            ComputationalBasis,
            3,
            (
                "Basis: [\n"
                "    [1, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 1, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 1, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 1, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 1, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 1, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 1, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 1]\n"
                "]\n"
            ),
        ),
        (
            ComputationalBasis,
            2,
            (
                "Basis: [\n"
                "    [1, 0, 0, 0],\n"
                "    [0, 1, 0, 0],\n"
                "    [0, 0, 1, 0],\n"
                "    [0, 0, 0, 1]\n"
                "]\n"
            ),
        ),
        (
            HadamardBasis,
            2,
            (
                "Basis: [\n"
                "    [0.5, 0.5, 0.5, 0.5],\n"
                "    [0.5, -0.5, 0.5, -0.5],\n"
                "    [0.5, 0.5, -0.5, -0.5],\n"
                "    [0.5, -0.5, -0.5, 0.5]\n"
                "]\n"
            ),
        ),
    ],
)
def test_basis_implementations(
    basis: type[VariableSizeBasis],
    size: int,
    result_pp: str,
    capsys: pytest.CaptureFixture[str],
):
    b = basis(nb_qubits=size)  # pyright: ignore[reportCallIssue]
    b.pretty_print()
    captured = capsys.readouterr()
    assert captured.out == result_pp


@pytest.mark.parametrize(
    "circuit, expected_probabilities",
    [
        (
            QCircuit(
                [
                    H(0),
                    CNOT(0, 1),
                    SWAP(1, 2),
                    Z(2),
                    BasisMeasure(
                        [0, 1, 2],
                        basis=Basis([np.eye(8)[i] for i in range(8)]),
                        shots=0,
                    ),
                ]
            ),
            np.array([0.5, 0, 0, 0, 0, 0.5, 0, 0]),
        ),
        (
            QCircuit(
                [
                    X(0),
                    H(1),
                    CNOT(1, 2),
                    Y(2),
                    BasisMeasure(
                        [0, 1, 2],
                        basis=Basis(
                            [
                                np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                                np.array([0, 1, 0, 0, 0, 0, 0, 0]),
                                np.array([0, 0, 1, 1, 0, 0, 0, 0]) / np.sqrt(2),
                                np.array([0, 0, 1, -1, 0, 0, 0, 0]) / np.sqrt(2),
                                np.array([0, 0, 0, 0, 1, 0, 1, 0]) / np.sqrt(2),
                                np.array([0, 0, 0, 0, 1, 0, -1, 0]) / np.sqrt(2),
                                np.array([0, 0, 0, 0, 0, 1, 0, 1]) / np.sqrt(2),
                                np.array([0, 0, 0, 0, 0, 1, 0, -1]) / np.sqrt(2),
                            ]
                        ),
                        shots=0,
                    ),
                ]
            ),
            np.array([0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25]),
        ),
    ],
)
def test_run_with_custom_basis_probas(
    circuit: QCircuit, expected_probabilities: npt.NDArray[np.complex128]
):
    res = run(circuit, IBMDevice.AER_SIMULATOR)
    assert matrix_eq(expected_probabilities, res.probabilities)


list_circuit_expected_vector_index = [
    (QCircuit([X(0), X(0)]), 0),
    (QCircuit([X(0)]), 1),
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "circuit, expected_vector_index",
    list_circuit_expected_vector_index,
)
def test_valid_run_custom_basis_state_vector_one_qubit_qiskit(
    circuit: QCircuit,
    expected_vector_index: int,
):
    exec_valid_run_custom_basis_state_vector_one_qubit(
        circuit, expected_vector_index, IBMDevice.AER_SIMULATOR
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circuit, expected_vector_index",
    list_circuit_expected_vector_index,
)
def test_valid_run_custom_basis_state_vector_one_qubit_braket(
    circuit: QCircuit,
    expected_vector_index: int,
):
    exec_valid_run_custom_basis_state_vector_one_qubit(
        circuit, expected_vector_index, AWSDevice.BRAKET_LOCAL_SIMULATOR
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize(
    "circuit, expected_vector_index",
    list_circuit_expected_vector_index,
)
def test_valid_run_custom_basis_state_vector_one_qubit_cirq(
    circuit: QCircuit,
    expected_vector_index: int,
):
    exec_valid_run_custom_basis_state_vector_one_qubit(
        circuit, expected_vector_index, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize(
    "circuit, expected_vector_index",
    list_circuit_expected_vector_index,
)
def test_valid_run_custom_basis_state_vector_one_qubit_myqlm(
    circuit: QCircuit,
    expected_vector_index: int,
):
    exec_valid_run_custom_basis_state_vector_one_qubit(
        circuit, expected_vector_index, ATOSDevice.MYQLM_PYLINALG
    )


def exec_valid_run_custom_basis_state_vector_one_qubit(
    circuit: QCircuit, expected_vector_index: int, device: AvailableDevice
):
    vectors = [np.array([np.sqrt(3) / 2, 1 / 2]), np.array([-1 / 2, np.sqrt(3) / 2])]

    with (
        pytest.warns(UnsupportedBraketFeaturesWarning)
        if isinstance(device, AWSDevice)
        else contextlib.suppress()
    ):
        result = run(
            circuit + QCircuit([BasisMeasure(basis=Basis(vectors), shots=0)]), device
        )
    assert isinstance(result, Result)
    assert matrix_eq(vectors[expected_vector_index], result.amplitudes)


@pytest.mark.provider("qiskit")
def test_run_custom_basis_sampling_one_qubit_qiskit():
    exec_run_custom_basis_sampling_one_qubit(IBMDevice.AER_SIMULATOR)


@pytest.mark.provider("braket")
def test_run_custom_basis_sampling_one_qubit_braket():
    exec_run_custom_basis_sampling_one_qubit(AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("cirq")
def test_run_custom_basis_sampling_one_qubit_cirq():
    exec_run_custom_basis_sampling_one_qubit(GOOGLEDevice.CIRQ_LOCAL_SIMULATOR)


@pytest.mark.provider("myqlm")
def test_run_custom_basis_sampling_one_qubit_myqlm():
    exec_run_custom_basis_sampling_one_qubit(ATOSDevice.MYQLM_PYLINALG)


def exec_run_custom_basis_sampling_one_qubit(device: AvailableDevice):
    vectors = [np.array([np.sqrt(3) / 2, 1 / 2]), np.array([-1 / 2, np.sqrt(3) / 2])]
    basis = Basis(vectors)
    run(
        QCircuit([X(0), X(0), BasisMeasure(basis=basis)]),
        device,
    )
    assert True
