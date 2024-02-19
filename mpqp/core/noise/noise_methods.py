from __future__ import annotations
from typing import Optional

from mpqp.core.instruction.gates import (
    KrausRepresentation,
    PauliDecomposition,
)
from mpqp.core.languages import Language

import numpy as np
import numpy.typing as npt

# MYQLM
# from qat.lang.AQASM import Program, H, PH, CNOT, SWAP, RX
from qat.quops.quantum_channels import QuantumChannelKraus

# QISKIT

# Import from Qiskit Aer noise module
from qiskit_aer.noise import depolarizing_error


class GateNoise:
    """
    A class used to apply Noise on a specific Gate

    Noise can be defined in several ways, and this class allows us apply a
    specific qubit quantum error to a selected gate

    Args:

    Attributes:


    Inspiration from MyQLM to be removed later
    # for each gate, we specify the noise
    # note that the values in this dictionary are lambda functions
    # with as many arguments as the gate's number of arguments
    gates_noise = {"H": lambda: noise,
               "CNOT": lambda: noise2,
               "RX": lambda _: noise}

    #This will go in the MPQPNoiseModel class
    hw_model = HardwareModel(gates_spec, gates_noise, idle_noise=None)

    """

    def __init__(
        self,
    ):
        # 3M-TODO : implement and comment
        pass


class GateNoiseCombination:
    """
    A class that allows to combine different types of Noise on a selected Gate(s)
    by using composition, tensor product, and tensor expansion and thus produces a
    new quantum error.


    Args:?
        matrix : unitary matrix representing the gate
        gate_combination : combination of gates (sum, product, ...) defining the gate
        kraus_operators : generalized Kraus representation of the gate
        pauli_decomposition : when it is possible, decomposition of the gate in the Pauli basis (I, X, Y, Z)
        nb_qubits : number of qubits of the gate defined

        more has to be added here, once it is thought through

    Attributes:
        _current_type (str): string describing which definition is currently used to define the gate


    """

    def __init__(self):
        # 3M-TODO : implement and comment
        self.pp = 1


class NoiseModules:
    """
    A class that contains function and methods for noise modeling.


    Args:
        matrix : unitary matrix representing the gate
        gate_combination : combination of gates (sum, product, ...) defining the gate
        kraus_operators : generalized Kraus representation of the gate
        pauli_decomposition : when it is possible, decomposition of the gate in the Pauli basis (I, X, Y, Z)
        nb_qubits : number of qubits of the gate defined

        more has to be added here, once it is thought through

    Attributes:
        _current_type (str): string describing which definition is currently used to define the gate


    """

    def __init__(self, nb_qubits: int, param: float, language: str):
        self._nb_qubits = nb_qubits
        self._language = language
        # put more

    def depolarizing_error(
        self, param: float, num_qubits: int, language: Language = Language.QISKIT
    ):
        """
        Args:
            param: depolarizing error parameter.
            num_qubits: the number of qubits for the error channel.
            language: ...

        Instantiating a noise model representing a depolarizing noise with probability 5%

        Example:
            For Qiskit:
            >>> error = depolarizing_error(0.05, 1)

        Example:
            For MyQLM:
            >>> prob = 0.05
            >>> X = np.array([[0, 1], [1, 0]])
            >>> Y = np.array([[0, -1j], [1j, 0]])
            >>> Z = np.array([[1, 0], [0, -1]])
            >>> kraus_ops = [np.sqrt(1-prob)*np.identity(2),
            >>> np.sqrt(prob/3)*X, np.sqrt(prob/3)*Y, np.sqrt(prob/3)*Z]
            >>> error = QuantumChannelKraus(kraus_ops)

        #How to connect?

        """

        # 3M-TODO : finish and comment

        if language == Language.QISKIT:
            error = depolarizing_error(param, num_qubits)
        elif language == Language.MY_QLM:
            X = np.array([[0, 1], [1, 0]])
            Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            Z = np.array([[1, 0], [0, -1]])
            kraus_ops = [
                np.sqrt(1 - param) * np.identity(2),
                np.sqrt(param / 3) * X,
                np.sqrt(param / 3) * Y,
                np.sqrt(param / 3) * Z,
            ]
            error = QuantumChannelKraus(kraus_ops)
        else:
            raise NotImplementedError(f"{language} not supported")

        return error

    def compose_noise(
        self, language: Language = Language.QISKIT
    ):  # can be also in MYQLM
        """
        Composes two noise gates

        Args:
            error1: ?
            error2:?
            noise_combi_type: bool
            etc?


        Returns: new_error
            ?

        """

        # here use mpqp tensor product function?
        # if noise_combi_type == noise_composition:
        #     error = error1.compose(error2)
        # elif noise_combi_type == noise_tensor_product:
        #     error = error1.tensor(error2)
        # elif noise_combi_type == noise_expand_product:
        #     error = error1.expand(error2)

        pass

    def krauss_error(
        self, language: Language = Language.QISKIT
    ):  # can be also in MYQLM
        """
        a general n-qubit CPTP error channel given as a list of Kraus matrices

        Args: mpqp_noise_operators (list[matrix]): Kraus matrices.
            ?

        Returns:
            The quantum error object - a Kraus quantum error channel.

        """
        pass
