"""This module is an implementation of one particular type of Variational
Quantum Algorithms: the Qaoa (Quantum Approximate Optimization Algorithm).
Mainly used for combinatorial optimization problems, and following the
trotterization principle, this algorithm works by generating a circuit of
alternated parametrized operators: the cost operator and the mixer operator.

Cost operators are generated based on the cost Hamiltonian, which encodes the
problem we want to optimize (usually expressed initially in Qubo formulation).

Mixer operators are here to escape from the natural convergence to the "closest"
eigenstate of the cost Hamiltonian, allowing the algorithm to explore more
widely the space of solutions. They can be customized for a specific problem,
but we provide a generic set of Mixer operators."""

from __future__ import annotations

from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt

from mpqp import QCircuit
from mpqp.execution import AvailableDevice, Result, run
from mpqp.execution.vqa import Optimizer, minimize
from mpqp.execution.vqa.qubo import Qubo
from mpqp.execution.vqa.vqa import OptimizerInput
from mpqp.gates import CustomGate, H
from mpqp.measures import BasisMeasure, ExpectationMeasure, Observable

if TYPE_CHECKING:
    from networkx import Graph
    from mpqp.tools.maths import Matrix


class QaoaMixer:
    """Class defining the Mixer hamiltonian used in the :func:`qaoa_solver` function.

    This class is used to help generate commonly used mixer Hamiltonians.
    The available Hamiltonians are regrouped in :class:`~mpqp.execution.vqa.qaoa.QaoaMixerType`.

    Args:
        type: Type of the mixer hamiltonian to be generated.
        graph: Graph needed to generate certain types of hamiltonian.
        bitflip: Value needed to build the bitflip hamiltonian.
    """

    def __init__(
        self,
        type: QaoaMixerType,
        graph: Optional["Graph"] = None,  # pyright: ignore[reportMissingTypeArgument]
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
        if self.graph is None:
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
        elif self.type == QaoaMixerType.MIXER_BITFLIP:
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
        else:
            raise NotImplementedError


class QaoaMixerType(Enum):
    r"""Enum class that provides a set of commonly used mixer Hamiltonians.

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
    this cost and the interpretation in terms of boolean variables of the Qubo problem.

    Args:
        cost: The minimum cost that was found.
        final_state: The quantum state associated with this cost.
        values: The associated variables with the state.
        final_parameters: Values of the parameters of the ansatz for the best found cost.

    Notes: This class is not meant to be instantiated directly by the user.
    """

    def __init__(
        self,
        cost: float,
        final_state: str,
        values: dict[str, int],
        final_params: OptimizerInput,
    ):
        self.cost = cost
        self.values: dict[str, int] = values
        self.final_state: str = final_state
        self.final_parameters: OptimizerInput = final_params

    def __str__(self) -> str:
        return (
            f"Minimum cost: {self.cost}\nAssociated state: {self.final_state}\n"
            f"Associated values: {self.values}"
        )


def qaoa_solver(
    problem: Qubo,
    depth: int,
    mixer: Union[QaoaMixer, Observable],
    device: AvailableDevice,
    optimizer: Optimizer = Optimizer.POWELL,
    init_params: Optional[list[float]] = None,
) -> QaoaResult:
    """This function solves decision problems using Qaoa, the problem needs to
    be inputted as a Qubo expression.

    Args:
        problem: Qubo expression representing the problem.
        depth: Number of layers in the ansatz, one layer being the application of
            one cost operator and one mixer operator.
        mixer: Type of the Mixer hamiltonian to be used or directly the mixer
            hamiltonian.
        device: The device that will be used to run the ansatz.
        optimizer: The optimizer used. Note: from our experience, not all of the
            available optimizers work well to solve Qaoa problems.
        init_params: List of parameters (float) used as the starting point for
            the optimization process, if empty initialize all parameters at 0.

    Returns:
        A QaoaResult containing the minimal cost found and the associated state.

    Examples:
        >>> x0 = QuboAtom('x0')
        >>> x1 = QuboAtom('x1')
        >>> expr = -3*x0 - 5*x1 + 4*(x0 & x1)
        >>> mixer = QaoaMixer(QaoaMixerType.MIXER_X)
        >>> qaoa_solver(expr, 4, mixer, IBMDevice.AER_SIMULATOR, Optimizer.POWELL).final_state # doctest: +SKIP
        '01'
    """
    observable = problem.to_cost_hamiltonian()
    problem_size = problem.size()
    if isinstance(mixer, QaoaMixer):
        mixer_matrix = mixer.generate_mixer_hamiltonian(problem_size)
    else:
        mixer_matrix = mixer.matrix
    loss_optimize = partial(
        _loss, cost=observable, nb_qubit=problem_size, mixer=mixer_matrix, device=device
    )
    if init_params is None:
        init_params = [0.0] * (depth * 2)
    elif len(init_params) != depth * 2:
        raise ValueError(
            f"Length of initial parameters must be the same as the number of gates in the ansatz, expected: {depth * 2} but got {len(init_params)}"
        )
    _, optimal_params = minimize(
        loss_optimize,
        method=optimizer,
        init_params=init_params,
    )

    # TODO: use .pretranspiled_circuit to avoid transpilation every time
    circuit = _generate_ansatz(optimal_params, observable, problem_size, mixer_matrix)
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
    return QaoaResult(problem.evaluate(values), res, values, optimal_params)


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
            in the Qaoa ansatz in this order: [gamma0, beta0, gamma1, beta1, ...].
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
        circuit: Generated Ansatz on which the unitary matrix will be applied.
        operator: Matrix representing either the cost Hamiltonian or the mixer Hamiltonian.
        parameter: The parameter controlling the application of the (cost/mixer) Hamiltonian, used to create the unitary matrix.
    """
    import scipy.linalg

    unitary = scipy.linalg.expm(-1j * parameter * operator)
    unitary_gate = CustomGate(
        unitary.astype(np.complex128), list(range(circuit.nb_qubits))
    )
    circuit.add(unitary_gate)


def _generate_ansatz(
    parameters: OptimizerInput,
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
