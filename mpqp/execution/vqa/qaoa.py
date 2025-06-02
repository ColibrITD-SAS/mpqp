from __future__ import annotations

from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import scipy
import scipy.linalg
import scipy.optimize

from mpqp import QCircuit
from mpqp.execution import AvailableDevice, Result, run
from mpqp.execution.vqa.qubo import Qubo
from mpqp.gates import CustomGate, H, UnitaryMatrix
from mpqp.measures import BasisMeasure, ExpectationMeasure, Observable
from mpqp.tools.maths import Matrix


class QAOAMixerType(Enum):
    MIXER_X = auto()
    MIXER_Y = auto()
    MIXER_Z = auto()


def qaoa_solver(
    problem: Qubo,
    depth: int,
    mixer: Union[QAOAMixerType, Matrix],
    device: AvailableDevice,
    optimizer: str,
) -> str:
    """This function solves decision problems using QAOA, the problem needs to
    be inputted as a QUBO expression.

    Args:
        problem: QUBO expression representing the problem.
        depth: The number of cost/mixer gates used in the circuit, the total depth of the ansatz being 2*depth.
        mixer: Type of the Mixer Hamiltonian to be used or directly the mixer Hamiltonian.
        device: The device that will be used to run the ansatz.
        optimizer: The optimizer used to minimize. 'Powell' is recommended for better results, but other can be more efficient on specific use cases.

    Returns:
        A string representing the resulting state of the QAOA optimization.

    Examples:
        >>> x0 = QuboAtom('x0')
        >>> x1 = QuboAtom('x1')
        >>> expr = -3*x0 - 5*x1 + 3*(x0 & x1)
        >>> qaoa_solver(expr, 4, QAOAMixerType.MIXER_X, IBMDevice.AER_SIMULATOR, 'Powell')
        '01'
    """
    observable = problem.to_cost_hamiltonian()
    problem_size = len(problem)
    if isinstance(mixer, QAOAMixerType):
        mixer = _generate_mixer_hamiltonian(problem_size, mixer)
    loss_optimize = partial(
        _loss, cost=observable, nqubit=problem_size, mixer=mixer, device=device
    )
    optimal_params = scipy.optimize.minimize(
        fun=loss_optimize, method=optimizer, x0=np.zeros(depth * 2)
    )

    circuit = _generate_ansatz(optimal_params.x, observable, problem_size, mixer)
    circuit.add(BasisMeasure(list(range(circuit.nb_qubits))))

    result = run(circuit, device)
    if TYPE_CHECKING:
        assert isinstance(result, Result)
    res = str(np.binary_repr(result.counts.index(max(result.counts))))
    return res.zfill(problem_size)


def _loss(
    parameters: list[float],
    cost: Observable,
    nqubit: int,
    mixer: Matrix,
    device: AvailableDevice,
) -> float:
    """Loss function calculating the expectation value of a QAOA ansatz
    initialized with the parameters.

    Args:
        parameters: List of floats representing the gamma and beta coefficient in the QAOA ansatz.
        cost: The cost Hamiltonian of the ansatz.
        nqubit: The size of the circuit.
        mixer: Mixer hamiltonian of the ansatz.
        device: The device that will be used to run the ansatz.

    Returns:
        A float containing the expectation value of the ansatz.
    """
    circuit = _generate_ansatz(parameters, cost, nqubit, mixer)
    circuit.add(ExpectationMeasure(cost, shots=0))
    result = run(circuit, device)
    if TYPE_CHECKING:
        assert isinstance(result, Result)
        assert isinstance(result.expectation_values, float)
    return result.expectation_values


def _generate_mixer_hamiltonian(
    qubits: int, type: QAOAMixerType
) -> npt.NDArray[np.complex128]:
    """Generates the mixer hamiltonian according to the mixer type.

    Args:
        qubits: Number of variables in the QUBO expression.
        type: The type of the mixer hamiltonian.

    Returns:
        The matrix of the Mixer Hamiltonian.
    """
    result = 0
    if type == QAOAMixerType.MIXER_X:
        mixer = np.array([[0, 1], [1, 0]])
    elif type == QAOAMixerType.MIXER_Y:
        mixer = np.array([[0, -1.0j], [1.0j, 0]])
    else:
        mixer = np.array([[1, 0], [0, -1]])

    result = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    for i in range(qubits):
        identity = np.eye(2**i)
        if i != 0:
            current = np.kron(identity, mixer)
        else:
            current = mixer
        if i != qubits - 1:
            identity = np.eye(2 ** (qubits - 1 - i))
            current = np.kron(current, identity)
        result += current
    return result


def _apply_unitary(circuit: QCircuit, operator: Matrix, parameter: float):
    """Apply the cost hamiltonian or the mixer hamiltonian to the generated
    ansatz.

    Args:
        circuit: Generated Ansatz on which the unitary matrix will me applied.
        operator: Either the cost hamiltonian or the mixer hamiltonian.
        parameter: The parameter used to create the unitary matrix.
    """
    unitary = scipy.linalg.expm(-1j * parameter * operator)
    unitary_gate = CustomGate(
        UnitaryMatrix(unitary.astype(np.complex128)), list(range(circuit.nb_qubits))
    )
    circuit.add(unitary_gate)


def _generate_ansatz(
    parameters: list[float],
    cost_hamiltonian: Observable,
    qubits: int,
    mixer: Matrix,
) -> QCircuit:
    """Generate the QAOA ansatz, which is composed of unitary operators acting
    on all of the circuit.

    Args:
        parameters: The parameters of the QAOA operators.
        cost_hamiltonian: The cost hamiltonian generated from que QUBO expression.
        qubits: Number of variables in the QUBO expression.
        mixer: The mixer hamiltonian chose for the ansatz.

    Returns:
        The generated ansatz.
    """
    ansatz = QCircuit(qubits)
    num_layers = len(parameters) // 2

    for i in range(qubits):
        ansatz.add(H(i))
    for i in range(num_layers):
        _apply_unitary(ansatz, cost_hamiltonian.matrix, parameters[2 * i])
        _apply_unitary(ansatz, mixer, parameters[2 * i + 1])
    return ansatz
