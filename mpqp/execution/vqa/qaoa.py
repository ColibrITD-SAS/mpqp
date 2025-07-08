"""This module is one implementation of a particular type of VQA: Qaoa.
This algorithm works by generating a circuit of alternating operators: cost
operators and mixer operators.
Cost operators are generated with the cost hamiltonian which represents the
problem we want to optimize.
Mixer operators are here to "search" for solutions, they can be custom to the
problem but a generic set does exist."""

from __future__ import annotations

from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt
import scipy
import scipy.linalg
import scipy.optimize
from networkx import Graph

from mpqp import QCircuit
from mpqp.execution import AvailableDevice, Result, run
from mpqp.execution.vqa.qubo import Qubo
from mpqp.gates import CustomGate, H, UnitaryMatrix
from mpqp.measures import BasisMeasure, ExpectationMeasure, Observable
from mpqp.tools.maths import Matrix


class QaoaMixer:
    """Class defining the Mixer hamiltonian used in the :func:`qaoa_solver` function.

    This class is used to help generate commonly used mixer hamiltonians,
    for available hamiltonian see :class:`~mpqp.execution.vqa.qaoa.QaoaMixerType`.

    Args:
        type: Type of the mixer hamiltonian to be generated.
        graph: Graph needed to generate certain types of hamiltonian.
        bitflip: Value needed to build the bitflip hamiltonian.
    """

    def __init__(
        self,
        type: QaoaMixerType,
        graph: Optional[Graph] = None,  # pyright: ignore[reportInvalidTypeArguments]
        bitflip: int = 0,
    ):
        self.type = type
        self.graph = graph
        self.bitflip = bitflip

    def generate_mixer_hamiltonian(self, qubits: int) -> npt.NDArray[np.complex128]:
        """Generates the mixer hamiltonian according to the mixer type.

        Args:
            qubits: Number of variables in the Qubo expression (also the number
                of qubits in the Qaoa ansatz).

        Returns:
            The matrix of the mixer hamiltonian.
        """
        if self.type == QaoaMixerType.MIXER_X:
            x_matrix = np.array([[0, 1], [1, 0]])
            result = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
            for i in range(qubits):
                result += _gen_ith_oper(qubits, x_matrix, i)
            return result
        if self.graph == None:
            raise ValueError(
                f"A graph is needed to generate the type {self.type} of hamiltonian."
            )
        if len(self.graph.nodes) > qubits:
            raise ValueError(
                f"Cannot have a graph with more nodes ({len(self.graph.nodes)})"
                f" than qubits ({qubits}) in the circuit."
            )
        if self.type == QaoaMixerType.MIXER_XY:
            result = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
            x_matrix = np.array([[0, 1], [1, 0]])
            y_matrix = np.array([[0, -1j], [1j, 0]])
            for i, j in self.graph.edges:
                result += _gen_ith_oper(qubits, x_matrix, i) @ _gen_ith_oper(
                    qubits, x_matrix, j
                ) + _gen_ith_oper(qubits, y_matrix, i) @ _gen_ith_oper(
                    qubits, y_matrix, j
                )
            result = result / 2
            return result
        else:
            result = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
            identity = np.eye(2**qubits)
            x_matrix = np.array([[0, 1], [1, 0]])
            z_matrix = np.array([[1, 0], [0, 1]])

            for vertex in self.graph.nodes:
                degree = len(self.graph[vertex])
                x_vertex = _gen_ith_oper(qubits, x_matrix, vertex)
                current = np.eye(2**qubits)
                for w in self.graph[vertex]:
                    current @= identity + (-1) ** self.bitflip * _gen_ith_oper(
                        qubits, z_matrix, w
                    )
                result += 0.5 ** (degree) * x_vertex @ current
            return result


class QaoaMixerType(Enum):
    r"""Enum class that provide a set of commonly used mixer hamiltonians:

    This mixer was introduced in A Quantum Approximate Optimization Algorithm by
    Edward Farhi, Jeffrey Goldstone, Sam Gutmann in https://arxiv.org/abs/1411.4028.

    MIXER_X = `\large \sum\limits_{i} X_i`

    Both mixers below were introduced in From the Quantum Approximate
    Optimization Algorithm to a Quantum Alternating Operator Ansatz by Stuart
    Hadfield, Zhihui Wang, Bryan O’Gorman, Eleanor G. Rieffel, Davide
    Venturelli, and Rupak Biswas in https://doi.org/10.3390/a12020034.

    MIXER_XY = `\large\frac{1}{2} \sum\limits_{(i,j)\in E(G)} X_i X_j + Y_i Y_j`

    MIXER_BITFLIP = `\large\sum \limits_{v\in V(G)} \frac{1}{2^{d(v)}}X_v \prod \limits_{w \in N(v)} (I + (-1)^b Z_w)`

    """

    MIXER_X = auto()
    MIXER_XY = auto()
    MIXER_BITFLIP = auto()


class QaoaResult:
    """This class is used to pack the different interpretations of the result of
    a Qaoa process.

    We put at disposition: the minimal cost found, the state associated with
    this cost and the interpretation of which variable acted on which qubits.

    Args:
        cost: The minimum cost that was found.
        final_state: The quantum state associated with this cost.
        values: The associated variables with the state.

    Notes: This class should only be instantiated by the program not by the user.
    """

    def __init__(self, cost: float, final_state: str, values: dict[str, int]):
        self.cost = cost
        self.values: dict[str, int] = values
        self.final_state: str = final_state

    def __str__(self) -> str:
        return (
            f"Minimum cost: {self.cost}\nAssociated state: {self.final_state}\n"
            f"Associated values: {self.values}"
        )


def qaoa_solver(
    problem: Qubo,
    depth: int,
    mixer: Union[QaoaMixer, Matrix],
    device: AvailableDevice,
    optimizer: str,
) -> QaoaResult:
    """This function solves decision problems using Qaoa, the problem needs to
    be inputted as a Qubo expression.

    Args:
        problem: Qubo expression representing the problem.
        depth: Number of cost/mixer gates used in the circuit, the total depth
            of the ansatz being 2*depth.
        mixer: Type of the Mixer hamiltonian to be used or directly the mixer
            hamiltonian.
        device: The device that will be used to run the ansatz.
        optimizer: The optimizer used to minimize. 'Powell' is recommended for
            better results, but other can be more efficient on specific use cases.

    Returns:
        A QaoaResult object holding the minimal cost found and the associated state.

    Examples:
        >>> x0 = QuboAtom('x0')
        >>> x1 = QuboAtom('x1')
        >>> expr = -3*x0 - 5*x1 + 3*(x0 & x1)
        >>> mixer = QaoaMixer(QaoaMixerType.MIXER_X)
        >>> qaoa_solver(expr, 4, mixer, IBMDevice.AER_SIMULATOR, 'Powell').final_state
        '01'
    """
    observable = problem.to_cost_hamiltonian()
    problem_size = problem.size()
    if isinstance(mixer, QaoaMixer):
        mixer_matrix = mixer.generate_mixer_hamiltonian(problem_size)
    else:
        mixer_matrix = mixer
    loss_optimize = partial(
        _loss, cost=observable, nb_qubit=problem_size, mixer=mixer_matrix, device=device
    )
    optimal_params = scipy.optimize.minimize(
        fun=loss_optimize, method=optimizer, x0=np.zeros(depth * 2)
    )

    circuit = _generate_ansatz(optimal_params.x, observable, problem_size, mixer_matrix)
    circuit.add(BasisMeasure(list(range(circuit.nb_qubits))))

    result = run(circuit, device)
    if TYPE_CHECKING:
        assert isinstance(result, Result)
    res = str(np.binary_repr(result.counts.index(max(result.counts))))

    res = res.zfill(problem_size)

    values = {}
    variables = problem.get_variables()
    for i in range(len(variables)):
        values.update({variables[i]: int(res[i])})
    cost = problem.evaluate(values)
    return QaoaResult(cost, res, values)


def _loss(
    parameters: list[float],
    cost: Observable,
    nb_qubit: int,
    mixer: Matrix,
    device: AvailableDevice,
) -> float:
    """Loss function calculating the expectation value of a Qaoa ansatz
    initialized with the parameters.

    Args:
        parameters: List of floats representing the gamma and beta coefficient
            in the Qaoa ansatz.
        cost: The cost hamiltonian of the ansatz.
        nb_qubit: Size of the circuit.
        mixer: Mixer hamiltonian of the ansatz.
        device: The device that will be used to run the ansatz.

    Returns:
        A float containing the expectation value of the ansatz.
    """
    circuit = _generate_ansatz(parameters, cost, nb_qubit, mixer)
    circuit.add(ExpectationMeasure(cost, shots=0))
    result = run(circuit, device)
    if TYPE_CHECKING:
        assert isinstance(result, Result)
        assert isinstance(result.expectation_values, float)
    return result.expectation_values


def _gen_ith_oper(
    qubits: int, matrix: npt.NDArray[np.complex128], i: int
) -> npt.NDArray[np.complex128]:
    """Returns the matrix equivalent to having the operator on the ith qubit on
    a circuit of size qubits. This function is used to generate some types of
    mixer hamiltonian."""
    from copy import deepcopy

    result = deepcopy(matrix)
    if i != 0:
        result = np.kron(np.eye(2**i), result)

    if qubits - i - 1 != 0:
        result = np.kron(result, np.eye(2 ** (qubits - i - 1)))
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
    """Generate the Qaoa ansatz, which is composed of unitary operators acting
    on all of the circuit.

    Args:
        parameters: The parameters of the Qaoa operators.
        cost_hamiltonian: The cost hamiltonian generated from que Qubo expression.
        qubits: Number of variables in the Qubo expression.
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
