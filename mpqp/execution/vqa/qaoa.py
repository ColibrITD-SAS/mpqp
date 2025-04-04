import scipy.optimize
from mpqp.execution.vqa.qubo import *
from functools import partial

from enum import Enum
from mpqp.measures import ExpectationMeasure
from mpqp import QCircuit
from mpqp.tools.maths import *
from mpqp.execution import run, IBMDevice, Result
from mpqp.gates import *
from mpqp.measures import *

import numpy as np
import numpy.typing as npt
import scipy
import scipy.linalg
from typing import TYPE_CHECKING


class MixerType(Enum):
    MIXER_X = 1
    MIXER_Y = 2
    MIXER_Z = 3


def loss(
    parameters: list[float],
    cost: Observable,
    nqubit: int,
    mixer: npt.NDArray[np.complex128],
) -> float:
    """
    Loss function calculating the expectation value of a QAOA ansatz initialized with the parameters.

    Args :
        parameters : list of floats representing the gamma and beta coefficient in the QAOA ansatz.
        cost : the cost Hamiltonian of the ansatz
        nqubit : the size of the circuit
        mixer : mixer hamiltonian of the ansatz

    Returns:
        A float containing the expectation value of the ansatz.
    """
    circuit = generate_ansatz(parameters, cost, nqubit, mixer)
    circuit.add(ExpectationMeasure(cost, shots=0))
    result = run(circuit, IBMDevice.AER_SIMULATOR)
    if TYPE_CHECKING:
        assert isinstance(result, Result)
        assert isinstance(result.expectation_values, float)
    return result.expectation_values


def qaoa_solver(problem: Qubo, depth: int, type: MixerType):
    """
    This function solves decision problems using QAOA, the problem needs to be inputted as a QUBO expression.

    Args:
        problem : QUBO expression representing the problem
        depth : The number of cost/mixer gates used in the circuit, the total depth of the ansatz being 2*depth
        type : Type of the Mixer Hamiltonian to be used

    Returns:
        A string holding the result of the QAOA optimization

    Examples:
        >>> x0 = Qubo('x0')
        >>> x1 = Qubo('x1')
        >>> expr = -3*x0 - 5*x1 + 3*(x0 & x1)
        >>> qaoa_solver(expr, 2, MixerType.MIXER_X)
        "01"
    """
    # build the cost hamiltonian
    observable = problem.to_cost_hamiltonian()
    # 4. Optimization
    mixer = generate_mixer_hamiltonian(problem.get_size(), type)

    loss_optimize = partial(
        loss, cost=observable, nqubit=problem.get_size(), mixer=mixer
    )
    optimal_params = scipy.optimize.minimize(
        fun=loss_optimize, method='Powell', x0=np.zeros(depth * 2)
    )

    circuit = generate_ansatz(optimal_params.x, observable, problem.get_size(), mixer)
    circuit.add(BasisMeasure(list(range(circuit.nb_qubits))))

    # 6 interpret result
    result = run(circuit, IBMDevice.AER_SIMULATOR)
    if TYPE_CHECKING:
        assert isinstance(result, Result)
    res = str(np.binary_repr(result.counts.index(max(result.counts))))
    while len(res) < problem.get_size():
        res = '0' + res
    return res


def apply_unitary(
    circuit: QCircuit, operator: Matrix | npt.NDArray[np.complex128], parameter: float
):
    unitary: npt.NDArray[np.complex64] = scipy.linalg.expm(-1j * parameter * operator)  # type: ignore
    unitary_gate = CustomGate(UnitaryMatrix(unitary), list(range(circuit.nb_qubits)))
    circuit.add(unitary_gate)


def generate_mixer_hamiltonian(
    qubits: int, type: MixerType
) -> npt.NDArray[np.complex128]:
    """
    Generates the mixer Hamiltonian according to the mixer type.
    """
    result = 0
    if type == MixerType.MIXER_X:
        mixer = np.array([[0, 1], [1, 0]])
    elif type == MixerType.MIXER_Y:
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


def generate_ansatz(
    parameters: list[float],
    cost_hamiltonian: Observable,
    qubits: int,
    mixer: npt.NDArray[np.complex128],
) -> QCircuit:
    """
    Generate the QAOA ansatz, it is composed of unitary operators acting on all of the circuit.
    """
    ansatz = QCircuit(qubits)
    num_layers = len(parameters) // 2

    for i in range(qubits):
        ansatz.add(H(i))
    for i in range(num_layers):
        apply_unitary(ansatz, cost_hamiltonian.matrix, parameters[2 * i])
        apply_unitary(ansatz, mixer, parameters[2 * i + 1])
    return ansatz
