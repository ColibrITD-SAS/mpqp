"""Native gates is the set of all gates natively supported in OpenQASM. Since we
rely on this standard, all of them are indeed implemented. In addition, this
module contains a few abstract classes used to factorize the behaviors common to
a lot of gates.

You will find bellow the list of available native gates:

.. container::
    :name: native-gates-list
    
    ``to-be-generated``
"""

from __future__ import annotations

import inspect
import sys
from abc import abstractmethod
from numbers import Integral
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sympy import Expr
    from qiskit.circuit import Parameter

import numpy as np
import numpy.typing as npt

# pylance doesn't handle well Expr, so a lot of "type:ignore" will happen in
# this file :/
from typeguard import typechecked

from mpqp.core.instruction.gates.controlled_gate import ControlledGate
from mpqp.core.instruction.gates.gate import Gate, InvolutionGate, SingleQubitGate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
from mpqp.core.languages import Language
from mpqp.tools.generics import Matrix, SimpleClassReprABC, classproperty
from mpqp.tools.maths import cos, exp, sin

# from sympy import Expr, pi


@typechecked
def _qiskit_parameter_adder(
    param: Expr | float, qiskit_parameters: set["Parameter"]
) -> "Parameter | float | int":
    """To avoid having several parameters in qiskit for the same value we keep
    track of them in a set. This function takes care of this, this way you can
    directly call `QiskitGate(_qiskit_parameter_adder(<param>, <q_params_set>))`
    without having to manually take care of the de-duping.

    This process is a form of memoization.

    Args:
        param: The parameter you need for your qiskit gate.
        qiskit_parameters: The set of previously set qiskit parameters. This set
        is updated inplace.

    Returns:
        The memoized parameter
    """
    from sympy import Expr

    if isinstance(param, Expr):
        name = str(param)
        previously_set_param = list(
            filter(lambda elt: elt.name == name, qiskit_parameters)
        )
        if len(previously_set_param) > 1:
            raise ReferenceError(
                "Somehow two parameter got the same name, this shouldn't be "
                "possible. For help on this error please contact the authors of"
                " this library"
            )
        elif len(previously_set_param) == 1:
            qiskit_param = previously_set_param[0]
        else:
            from qiskit.circuit import Parameter

            qiskit_param = Parameter(name)
            qiskit_parameters.add(qiskit_param)
    else:
        qiskit_param = param
    return qiskit_param


@typechecked
class NativeGate(Gate, SimpleClassReprABC):
    """The standard on which we rely, OpenQASM, comes with a set of gates
    supported by default. More complicated gates can be defined by the user.
    This abstract class represent all those gates supported by default.

    Args:
        targets: List of indices referring to the qubits on which the gate will
            be applied.
        label: Label used to identify the gate.
    """

    qlm_aqasm_keyword: str
    """Keyword(s) corresponding to the gate in ``myQLM``. This needs to be
    available at the class level and is not enforced by the type checker so be
    careful about it!"""
    qiskit_string: str
    """Keyword corresponding to the gate in ``qiskit``. This needs to be
    available at the class level and is not enforced by the type checker so be
    careful about it!"""

    @classproperty
    def qasm2_gate(cls) -> str:
        """Keyword(s) corresponding to the gate in ``QASM2``."""
        return cls.qiskit_string

    native_gate_options = {"disable_symbol_warn": True}

    if TYPE_CHECKING:
        from braket.circuits import gates
        from qiskit.circuit.library import (
            CCXGate,
            CPhaseGate,
            CXGate,
            CZGate,
            HGate,
            IGate,
            PhaseGate,
            RXGate,
            RYGate,
            RZGate,
            SGate,
            SwapGate,
            TGate,
            XGate,
            YGate,
            ZGate,
        )

    @classproperty
    @abstractmethod
    def qiskit_gate(
        cls,
    ) -> type[
        XGate
        | YGate
        | ZGate
        | HGate
        | TGate
        | SGate
        | SwapGate
        | CXGate
        | CZGate
        | CCXGate
        | IGate
        | RXGate
        | RYGate
        | RZGate
        | PhaseGate
        | CPhaseGate
    ]:
        pass

    @classproperty
    @abstractmethod
    def braket_gate(
        cls,
    ) -> type[
        gates.X
        | gates.Y
        | gates.Z
        | gates.H
        | gates.T
        | gates.S
        | gates.Swap
        | gates.CNot
        | gates.CZ
        | gates.CCNot
        | gates.I
        | gates.Rx
        | gates.Ry
        | gates.Rz
        | gates.PhaseShift
        | gates.CPhaseShift
    ]:
        pass


@typechecked
class RotationGate(NativeGate, ParametrizedGate, SimpleClassReprABC):
    """Many gates can be classified as a simple rotation gate, around a specific
    axis (and potentially with a control qubit). All those gates have in common
    a single parameter: ``theta``. This abstract class helps up factorize this
    behavior, and simply having to tweak the matrix semantics and qasm
    translation of the specific gate.

    Args:
        theta: Angle of the rotation.
        target: Index referring to the qubits on which the gate will be applied.
    """

    def __init__(self, theta: Expr | float, target: int):
        self.parameters = [theta]
        definition = UnitaryMatrix(
            self.to_canonical_matrix(), **self.native_gate_options
        )
        ParametrizedGate.__init__(
            self, definition, [target], [self.theta], type(self).__name__.capitalize()
        )

    @property
    def theta(self):
        """Rotation angle (in radians)."""
        return self.parameters[0]

    def __repr__(self):
        return f"{type(self).__name__}({self.theta}, {self.targets[0]})"

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if qiskit_parameters is None:
            qiskit_parameters = set()
        try:
            theta = float(self.theta)
        except:
            theta = self.theta
        if language == Language.QISKIT:
            return self.qiskit_gate(_qiskit_parameter_adder(theta, qiskit_parameters))
        elif language == Language.BRAKET:
            from sympy import Expr

            # TODO: handle symbolic parameters for Braket
            if isinstance(theta, Expr):
                raise NotImplementedError(
                    "Symbolic expressions are not yet supported for braket "
                    "export, this feature is coming very soon!"
                )
            return self.braket_gate(theta)
        if language == Language.QASM2:
            from mpqp.qasm.mpqp_to_qasm import float_to_qasm_str

            instruction_str = self.qasm2_gate
            instruction_str += (
                "("
                + ",".join(float_to_qasm_str(float(param)) for param in self.parameters)
                + ")"
            )

            qubits = ""
            if isinstance(self, ControlledGate):
                qubits = ",".join([f"q[{j}]" for j in self.controls]) + ","
            qubits += ",".join([f"q[{j}]" for j in self.targets])

            return instruction_str + " " + qubits + ";"
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def inverse(self) -> Gate:
        return self.__class__(-self.parameters[0], self.targets[0])


@typechecked
class NoParameterGate(NativeGate, SimpleClassReprABC):
    """Abstract class describing native gates that do not depend on parameters.

    Args:
        targets: List of indices referring to the qubits on which the gate will
            be applied.
        label: Label used to identify the gate.
    """

    qlm_aqasm_keyword: str

    if TYPE_CHECKING:
        from braket.circuits import gates
        from qiskit.circuit.library import (
            CCXGate,
            CXGate,
            CZGate,
            HGate,
            IGate,
            SGate,
            SwapGate,
            TGate,
            XGate,
            YGate,
            ZGate,
        )

    @classproperty
    @abstractmethod
    def qiskit_gate(
        cls,
    ) -> type[
        XGate
        | YGate
        | ZGate
        | HGate
        | TGate
        | SGate
        | SwapGate
        | CXGate
        | CZGate
        | CCXGate
        | IGate
    ]:
        """Returns the corresponding ``qiskit`` class for this gate."""
        pass

    @classproperty
    @abstractmethod
    def braket_gate(
        cls,
    ) -> type[
        gates.X
        | gates.Y
        | gates.Z
        | gates.H
        | gates.T
        | gates.S
        | gates.Swap
        | gates.CNot
        | gates.CZ
        | gates.CCNot
        | gates.I
    ]:
        """Returns the corresponding ``braket`` class for this gate."""
        pass

    """Corresponding ``qiskit``'s gate class."""
    matrix: npt.NDArray[np.complex64]
    """Matricial semantics of the gate."""

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QISKIT:
            return self.qiskit_gate()
        elif language == Language.BRAKET:
            return self.braket_gate()
        elif language == Language.QASM2:
            instruction_str = self.qasm2_gate

            qubits = ""
            if isinstance(self, ControlledGate):
                qubits = ",".join([f"q[{j}]" for j in self.controls]) + ","
            qubits += ",".join([f"q[{j}]" for j in self.targets])

            return instruction_str + " " + qubits + ";"
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def to_canonical_matrix(self) -> Matrix:
        return self.matrix


@typechecked
class OneQubitNoParamGate(SingleQubitGate, NoParameterGate, SimpleClassReprABC):
    """Abstract Class describing one-qubit native gates that do not depend on
    parameters.

    Args:
        target: Index referring to the qubits on which the gate will be applied.
    """

    def __init__(self, target: int):
        SingleQubitGate.__init__(self, target, type(self).__name__)


class Id(OneQubitNoParamGate, InvolutionGate):
    r"""One qubit identity gate.

    `\begin{pmatrix}1&0\\0&1\end{pmatrix}`

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(Id(0).to_matrix())
        [[1, 0],
         [0, 1]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.I

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import IGate

        return IGate

    qlm_aqasm_keyword = "I"
    qiskit_string = "id"

    def __init__(self, target: int, label: Optional[str] = None):
        super().__init__(target)
        self.label = label
        self.matrix = np.eye(2, dtype=np.complex64)

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QISKIT:
            if self.label:
                return self.qiskit_gate(label=self.label)
            return self.qiskit_gate()
        elif language == Language.BRAKET:
            return self.braket_gate()
        elif language == Language.QASM2:

            instruction_str = self.qasm2_gate
            qubits = ",".join([f"q[{j}]" for j in self.targets])

            return instruction_str + " " + qubits + ";"
        else:
            raise NotImplementedError(f"Error: {language} is not supported")


class X(OneQubitNoParamGate, InvolutionGate):
    r"""One qubit X (NOT) Pauli gate.

    `\begin{pmatrix}0&1\\1&0\end{pmatrix}`

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(X(0).to_matrix())
        [[0, 1],
         [1, 0]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.X

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import XGate

        return XGate

    qlm_aqasm_keyword = "X"
    qiskit_string = "x"

    def __init__(self, target: int):
        super().__init__(target)
        self.matrix = np.array([[0, 1], [1, 0]])


class Y(OneQubitNoParamGate, InvolutionGate):
    r"""One qubit Y Pauli gate.

    `\begin{pmatrix}0&-i\\i&0\end{pmatrix}`

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(Y(0).to_matrix())
        [[0 , -1j],
         [1j, 0  ]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Y

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import YGate

        return YGate

    qlm_aqasm_keyword = "Y"
    qiskit_string = "y"

    def __init__(self, target: int):
        super().__init__(target)
        self.matrix = np.array([[0, -1j], [1j, 0]])


class Z(OneQubitNoParamGate, InvolutionGate):
    r"""One qubit Z Pauli gate.

    `\begin{pmatrix}1&0\\0&-1\end{pmatrix}`

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(Z(0).to_matrix())
        [[1, 0 ],
         [0, -1]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Z

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import ZGate

        return ZGate

    qlm_aqasm_keyword = "Z"
    qiskit_string = "z"

    def __init__(self, target: int):
        super().__init__(target)
        self.matrix = np.array([[1, 0], [0, -1]])


class H(OneQubitNoParamGate, InvolutionGate):
    r"""One qubit Hadamard gate. `\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}`

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(H(0).to_matrix())
        [[0.70711, 0.70711 ],
         [0.70711, -0.70711]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.H

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import HGate

        return HGate

    qlm_aqasm_keyword = "H"
    qiskit_string = "h"

    def __init__(self, target: int):
        super().__init__(target)
        self.matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


class P(RotationGate, SingleQubitGate):
    r"""One qubit parametrized Phase gate. Consist in a rotation around Z axis.

    `\begin{pmatrix}1&0\\0&e^{i\theta}\end{pmatrix}`

    Args:
        theta: Parameter representing the phase to apply.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(P(np.pi/3, 1).to_matrix())
        [[1, 0           ],
         [0, 0.5+0.86603j]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.PhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import PhaseGate

        return PhaseGate

    qlm_aqasm_keyword = "PH"
    qiskit_string = "p"

    def __init__(self, theta: Expr | float, target: int):
        super().__init__(theta, target)

    def to_canonical_matrix(self) -> Matrix:
        return np.array(  # pyright: ignore[reportCallIssue]
            [
                [1, 0],
                [
                    0,
                    exp(
                        self.parameters[0] * 1j  # pyright: ignore[reportOperatorIssue]
                    ),
                ],
            ]
        )


class CP(RotationGate, ControlledGate):
    """Two-qubit Controlled-P gate.

    Args:
        theta: Parameter representing the phase to apply.
        control: Index referring to the qubit used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(CP(0.5, 0, 1).to_matrix())
        [[1, 0, 0, 0               ],
         [0, 1, 0, 0               ],
         [0, 0, 1, 0               ],
         [0, 0, 0, 0.87758+0.47943j]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.CPhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import CPhaseGate

        return CPhaseGate

    # TODO: this is a special case, see if it needs to be generalized
    qlm_aqasm_keyword = "CNOT;PH"
    qiskit_string = "cp"

    def __init__(self, theta: Expr | float, control: int, target: int):
        self.parameters = [theta]
        ControlledGate.__init__(self, [control], [target], P(theta, target), "CP")
        definition = UnitaryMatrix(
            self.to_canonical_matrix(), **self.native_gate_options
        )
        ParametrizedGate.__init__(self, definition, [target], [theta], "CP")

    def to_canonical_matrix(self):
        e = exp(self.theta * 1j)  # pyright: ignore[reportOperatorIssue]
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, e]])

    def __repr__(self) -> str:
        theta = int(self.theta) if self.theta == int(self.theta) else self.theta
        return f"{type(self).__name__}({theta}, {self.controls[0]}, {self.targets[0]})"

    def inverse(self) -> Gate:
        return self.__class__(-self.parameters[0], self.controls[0], self.targets[0])

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )


class S(OneQubitNoParamGate):
    r"""One qubit S gate. It's equivalent to ``P(pi/2)``.
    It can also be defined as the square-root of the Z (Pauli) gate.

    `\begin{pmatrix}1&0\\0&i\end{pmatrix}`

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(S(0).to_matrix())
        [[1, 0 ],
         [0, 1j]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.S

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import SGate

        return SGate

    qlm_aqasm_keyword = "S"
    qiskit_string = "s"

    def __init__(self, target: int):
        super().__init__(target)
        self.matrix = np.array([[1, 0], [0, 1j]])


class T(OneQubitNoParamGate):
    r"""One qubit T gate. It is also referred to as the `\pi/4` gate because it
    consists in applying the phase gate with a phase of `\pi/4`.

    `\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}`

    The T gate can also be defined as the fourth-root of the Z (Pauli) gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(T(0).to_matrix())
        [[1, 0                 ],
         [0, 1.0*exp(0.25*I*pi)]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.T

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import TGate

        return TGate

    qlm_aqasm_keyword = "T"
    qiskit_string = "t"

    def __init__(self, target: int):
        super().__init__(target)

    def to_canonical_matrix(self):
        from sympy import pi

        return np.array([[1, 0], [0, exp((pi / 4) * 1j)]])


class SWAP(InvolutionGate, NoParameterGate):
    r"""Two-qubit SWAP gate.

    `\begin{pmatrix}1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&1\end{pmatrix}`

    Args:
        a: First target of the swapping operation.
        b: Second target of the swapping operation.

    Example:
        >>> pprint(SWAP(0, 1).to_matrix())
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Swap

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import SwapGate

        return SwapGate

    qlm_aqasm_keyword = "SWAP"
    qiskit_string = "swap"

    def __init__(self, a: int, b: int):
        super().__init__([a, b], "SWAP")
        self.matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.targets[0]}, {self.targets[1]})"

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )
    """Size of the gate."""

    def to_matrix(self, desired_gate_size: int = 0) -> npt.NDArray[np.complex64]:
        """Constructs the matrix representation of a SWAP gate for two qubits.

        Args:
            nb_qubits: The total number for qubits gate representation. If not
                provided, the minimum number of qubits required to generate the
                matrix will be used.

        Returns:
            The matrix representation of the SWAP gate.
        """
        control, target = self.targets[0], self.targets[1]

        max_qubits = max(control, target) + 1
        if desired_gate_size != 0 and desired_gate_size < max_qubits:
            raise ValueError(
                f"The number of qubits in the system must be at least {max_qubits}."
            )

        nb_qubits_swap = abs(control - target) + 1
        min_nb_qubits = min(control, target)
        swap_matrix = np.eye(2**nb_qubits_swap, dtype=np.complex64)

        for i in range(2**nb_qubits_swap):
            binary_state = list(format(i, f"0{nb_qubits_swap}b"))

            (
                binary_state[nb_qubits_swap - control + min_nb_qubits - 1],
                binary_state[nb_qubits_swap - target + min_nb_qubits - 1],
            ) = (
                binary_state[nb_qubits_swap - target + min_nb_qubits - 1],
                binary_state[nb_qubits_swap - control + min_nb_qubits - 1],
            )

            swapped_index = int("".join(binary_state), 2)

            swap_matrix[i, i] = 0
            swap_matrix[swapped_index, i] = 1

        if desired_gate_size != 0:
            swap_matrix = np.kron(np.eye(2**min_nb_qubits), swap_matrix)
            swap_matrix = np.kron(
                swap_matrix, np.eye(2 ** (desired_gate_size - max_qubits))
            )
        return swap_matrix


class U(NativeGate, ParametrizedGate, SingleQubitGate):
    r"""Generic one qubit unitary gate. It is parametrized by 3 Euler angles.

    `\begin{pmatrix}\cos(\theta/2)&-e^{i\gamma}\sin(\theta/2)\\e^{i\phi}\sin(\theta/2)&e^{i(\gamma+\phi)}\cos(\theta/2)\end{pmatrix}`

    Args:
        theta: Parameter representing the first angle of the gate U.
        phi: Parameter representing the second angle of the gate U.
        gamma: Parameter representing the third angle of the gate U.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(U(np.pi/3, 0, np.pi/4, 0).to_matrix())
        [[0.86603, -0.35355-0.35355j],
         [0.5    , 0.61237+0.61237j ]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits.gates import U as braket_U

        return braket_U

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import UGate

        return UGate

    qlm_aqasm_keyword = "U"
    qiskit_string = "u"

    def __init__(
        self,
        theta: Expr | float,
        phi: Expr | float,
        gamma: Expr | float,
        target: int,
    ):
        self.parameters = [theta, phi, gamma]
        definition = UnitaryMatrix(
            self.to_canonical_matrix(), **self.native_gate_options
        )
        ParametrizedGate.__init__(self, definition, [target], [theta, phi, gamma], "U")

    @property
    def theta(self):
        """See corresponding argument."""
        return self.parameters[0]

    @property
    def phi(self):
        """See corresponding argument."""
        return self.parameters[1]

    @property
    def gamma(self):
        """See corresponding argument."""
        return self.parameters[2]

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QISKIT:

            if qiskit_parameters is None:
                qiskit_parameters = set()

            return self.qiskit_gate(
                theta=_qiskit_parameter_adder(self.theta, qiskit_parameters),
                phi=_qiskit_parameter_adder(self.phi, qiskit_parameters),
                lam=_qiskit_parameter_adder(self.gamma, qiskit_parameters),
            )
        elif language == Language.BRAKET:
            from sympy import Expr

            # TODO handle symbolic parameters
            if (
                isinstance(self.theta, Expr)
                or isinstance(self.phi, Expr)
                or isinstance(self.gamma, Expr)
            ):
                raise NotImplementedError(
                    "Symbolic expressions are not yet supported for braket "
                    "export, this feature is coming very soon!"
                )

            return self.braket_gate(self.theta, self.phi, self.gamma)
        if language == Language.QASM2:
            from mpqp.qasm.mpqp_to_qasm import float_to_qasm_str

            instruction_str = self.qasm2_gate
            instruction_str += (
                "("
                + ",".join(float_to_qasm_str(float(param)) for param in self.parameters)
                + ")"
            )
            qubits = ",".join([f"q[{j}]" for j in self.targets])

            return instruction_str + " " + qubits + ";"
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def to_canonical_matrix(self):
        c, s, eg, ep = (
            cos(self.theta / 2),  # pyright: ignore[reportOperatorIssue]
            sin(self.theta / 2),  # pyright: ignore[reportOperatorIssue]
            exp(self.gamma * 1j),  # pyright: ignore[reportOperatorIssue]
            exp(self.phi * 1j),  # pyright: ignore[reportOperatorIssue]
        )
        return np.array(  # pyright: ignore[reportCallIssue]
            [
                [c, -eg * s],  # pyright: ignore[reportOperatorIssue]
                [ep * s, eg * ep * c],  # pyright: ignore[reportOperatorIssue]
            ]
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.theta}, {self.phi}, {self.gamma}, {self.targets[0]})"


class Rx(RotationGate, SingleQubitGate):
    r"""One qubit rotation around the X axis.

    `\begin{pmatrix}\cos(\theta/2)&-i\sin(\theta/2)\\-i\sin(\theta/2)&\cos(\theta/2)\end{pmatrix}`

    Args:
        theta: Parameter representing the angle of the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(Rx(np.pi/5, 1).to_matrix())
        [[0.95106  , -0.30902j],
         [-0.30902j, 0.95106  ]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Rx

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import RXGate

        return RXGate

    qlm_aqasm_keyword = "RX"
    qiskit_string = "rx"

    def __init__(self, theta: Expr | float, target: int):
        super().__init__(theta, target)

    def to_canonical_matrix(self):
        c = cos(self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        s = sin(self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        return np.array(  # pyright: ignore[reportCallIssue]
            [[c, -1j * s], [-1j * s, c]]  # pyright: ignore[reportOperatorIssue]
        )


class Ry(RotationGate, SingleQubitGate):
    r"""One qubit rotation around the Y axis.

    `\begin{pmatrix}\cos(\theta/2)&-\sin(\theta/2)\\\sin(\theta/2)&\cos(\theta/2)\end{pmatrix}`

    Args:
        theta: Parameter representing the angle of the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(Ry(np.pi/5, 1).to_matrix())
        [[0.95106, -0.30902],
         [0.30902, 0.95106 ]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Ry

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import RYGate

        return RYGate

    qlm_aqasm_keyword = "RY"
    qiskit_string = "ry"

    def __init__(self, theta: Expr | float, target: int):
        super().__init__(theta, target)

    def to_canonical_matrix(self):
        c = cos(self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        s = sin(self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        return np.array([[c, -s], [s, c]])


class Rz(RotationGate, SingleQubitGate):
    r"""One qubit rotation around the Z axis.

    `\begin{pmatrix}e^{i\theta/2}&0\\0&e^{-i\theta/2}\end{pmatrix}`

    Args:
        theta: Parameter representing the angle of the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(Rz(np.pi/5, 1).to_matrix())
        [[0.95106-0.30902j, 0               ],
         [0               , 0.95106+0.30902j]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Rz

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import RZGate

        return RZGate

    qlm_aqasm_keyword = "RZ"
    qiskit_string = "rz"

    def __init__(self, theta: Expr | float, target: int):
        super().__init__(theta, target)

    def to_canonical_matrix(self):
        e = exp(-1j * self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        return np.array(  # pyright: ignore[reportCallIssue]
            [[e, 0], [0, 1 / e]]  # pyright: ignore[reportOperatorIssue]
        )


class Rk(RotationGate, SingleQubitGate):
    r"""One qubit Phase gate of angle `\frac{2i\pi}{2^k}`.

    `\begin{pmatrix}1&0\\0&e^{i\pi/2^{k-1}}\end{pmatrix}`

    Args:
        k: Parameter used in the definition of the phase to apply.
        target: Index referring to the qubit on which the gate will be applied.

    Examples:
        >>> pprint(Rk(5, 0).to_matrix())
        [[1, 0               ],
         [0, 0.98079+0.19509j]]

        >>> pprint(Rk(k, 0).to_matrix())
        [[1, 0                     ],
         [0, 1.0*exp(2.0*I*pi/2**k)]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.PhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import PhaseGate

        return PhaseGate

    qlm_aqasm_keyword = "PH"
    qiskit_string = "p"

    def __init__(self, k: Expr | int, target: int):
        self.parameters = [k]
        definition = UnitaryMatrix(
            self.to_canonical_matrix(), **self.native_gate_options
        )
        ParametrizedGate.__init__(self, definition, [target], [self.k], "Rk")

    @property
    def theta(self) -> Expr | float:
        r"""Value of the rotation angle, parametrized by ``k`` with the relation
        `\theta = \frac{\pi}{2^{k-1}}`."""
        from sympy import pi

        p = np.pi if isinstance(self.k, Integral) else pi
        return p / 2 ** (self.k - 1)  # pyright: ignore[reportOperatorIssue]

    @property
    def k(self) -> Expr | int:
        """See corresponding argument."""
        return self.parameters[0]

    def to_canonical_matrix(self):
        e = exp(self.theta * 1j)  # pyright: ignore[reportOperatorIssue]
        return np.array([[1, 0], [0, e]])

    def __repr__(self):
        return f"{type(self).__name__}({self.k}, {self.targets[0]})"

    def inverse(self) -> Gate:
        return Rk_dagger(self.k, self.targets[0])

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QASM2:
            from mpqp.qasm.mpqp_to_qasm import float_to_qasm_str

            instruction_str = self.qasm2_gate
            instruction_str += (
                f"({float_to_qasm_str(2 * np.pi / (2 ** float(self.k)))})"
            )

            qubits = ",".join([f"q[{j}]" for j in self.targets])

            return instruction_str + " " + qubits + ";"
        else:
            return super().to_other_language(language, qiskit_parameters)


class Rk_dagger(RotationGate, SingleQubitGate):
    r"""One qubit Phase gate of angle `-\frac{2i\pi}{2^k}`.

    `\begin{pmatrix}1&0\\0&e^{-i\pi/2^{k-1}}\end{pmatrix}`

    Args:
        k: Parameter used in the definition of the phase to apply.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(Rk_dagger(5, 0).to_matrix())
        [[1, 0               ],
         [0, 0.98079-0.19509j]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.PhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import PhaseGate

        return PhaseGate

    qlm_aqasm_keyword = "PH"
    qiskit_string = "p"

    def __init__(self, k: Expr | int, target: int):
        self.parameters = [k]
        definition = UnitaryMatrix(
            self.to_canonical_matrix(), **self.native_gate_options
        )
        ParametrizedGate.__init__(self, definition, [target], [self.k], "Rk†")

    @property
    def theta(self) -> Expr | float:
        r"""Value of the rotation angle, parametrized by ``k`` with the relation
        `\theta = -\frac{\pi}{2^{k-1}}`."""
        from sympy import pi

        # TODO study the relevance of having pi from sympy
        p = np.pi if isinstance(self.k, Integral) else pi
        return -(p / 2 ** (self.k - 1))  # pyright: ignore[reportOperatorIssue]

    @property
    def k(self) -> Expr | float:
        """See corresponding argument."""
        return self.parameters[0]

    def to_canonical_matrix(self):
        e = exp(self.theta * 1j)  # pyright: ignore[reportOperatorIssue]
        return np.array([[1, 0], [0, e]])

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QASM2:
            from mpqp.qasm.mpqp_to_qasm import float_to_qasm_str

            instruction_str = self.qasm2_gate
            instruction_str += (
                f"({float_to_qasm_str(-2 * np.pi / (2 ** float(self.k)))})"
            )

            qubits = ",".join([f"q[{j}]" for j in self.targets])

            return instruction_str + " " + qubits + ";"
        else:
            return super().to_other_language(language, qiskit_parameters)

    def __repr__(self):
        return f"{type(self).__name__}({self.k}, {self.targets[0]})"

    def inverse(self) -> Gate:
        return Rk(self.parameters[0], self.targets[0])


class CNOT(InvolutionGate, ControlledGate, NoParameterGate):
    r"""Two-qubit Controlled-NOT gate.

    `\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{pmatrix}`

    Args:
        control: index referring to the qubit used to control the gate
        target: index referring to the qubit on which the gate will be applied

    Example:
        >>> pprint(CNOT(0, 1).to_matrix())
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.CNot

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import CXGate

        return CXGate

    qlm_aqasm_keyword = "CNOT"
    qiskit_string = "cx"

    def __init__(self, control: int, target: int):
        ControlledGate.__init__(self, [control], [target], X(target), "CNOT")

    def to_canonical_matrix(self):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )
    """Size of the gate."""


class CZ(InvolutionGate, ControlledGate, NoParameterGate):
    r"""Two-qubit Controlled-Z gate.

    `\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&-1\end{pmatrix}`

    Args:
        k: Parameter used in the definition of the phase to apply.
        control: Index referring to the qubit used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Examples:
        >>> pprint(CZ(0, 1).to_matrix())
        [[1, 0, 0, 0 ],
         [0, 1, 0, 0 ],
         [0, 0, 1, 0 ],
         [0, 0, 0, -1]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.CZ

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import CZGate

        return CZGate

    qiskit_string = "cz"
    qlm_aqasm_keyword = "CSIGN"

    def __init__(self, control: int, target: int):
        ControlledGate.__init__(self, [control], [target], Z(target), "CZ")

    def to_canonical_matrix(self):
        m = np.eye(4, dtype=complex)
        m[-1, -1] = -1
        return m

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )
    """Size of the gate."""


class CRk(RotationGate, ControlledGate):
    r"""Two-qubit Controlled-Rk gate.

    `\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&e^{i\pi/2^{k-1}}\end{pmatrix}`

    Args:
        k: Parameter used in the definition of the phase to apply.
        control: Index referring to the qubit used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Examples:
        >>> pprint(CRk(4, 0, 1).to_matrix())
        [[1, 0, 0, 0               ],
         [0, 1, 0, 0               ],
         [0, 0, 1, 0               ],
         [0, 0, 0, 0.92388+0.38268j]]

        >>> k = symbols("k")
        >>> pprint(CRk(k, 0, 1).to_matrix())
        [[1, 0, 0, 0                     ],
         [0, 1, 0, 0                     ],
         [0, 0, 1, 0                     ],
         [0, 0, 0, 1.0*exp(2.0*I*pi/2**k)]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.CPhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import CPhaseGate

        return CPhaseGate

    # TODO: this is a special case, see if it needs to be generalized
    qlm_aqasm_keyword = "CNOT;PH"
    qiskit_string = "cp"

    def __init__(self, k: Expr | int, control: int, target: int):
        self.parameters = [k]
        ControlledGate.__init__(self, [control], [target], Rk(k, target), "CRk")
        definition = UnitaryMatrix(
            self.to_canonical_matrix(), **self.native_gate_options
        )
        ParametrizedGate.__init__(self, definition, [target], [k], "CRk")

    @property
    def theta(self) -> Expr | float:
        r"""Value of the rotation angle, parametrized by ``k`` with the relation
        `\theta = \frac{\pi}{2^{k-1}}`."""
        from sympy import pi

        p = np.pi if isinstance(self.k, Integral) else pi
        return p / 2 ** (self.k - 1)  # pyright: ignore[reportOperatorIssue]

    @property
    def k(self) -> Expr | float:
        """See corresponding argument."""
        return self.parameters[0]

    def to_canonical_matrix(self):
        e = exp(self.theta * 1j)  # pyright: ignore[reportOperatorIssue]
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, e]])

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QASM2:
            from mpqp.qasm.mpqp_to_qasm import float_to_qasm_str

            instruction_str = self.qasm2_gate
            instruction_str += (
                f"({float_to_qasm_str(2 * np.pi / (2 ** float(self.k)))})"
            )

            qubits = ",".join([f"q[{j}]" for j in self.controls]) + ","
            qubits += ",".join([f"q[{j}]" for j in self.targets])

            return instruction_str + " " + qubits + ";"
        else:
            return super().to_other_language(language, qiskit_parameters)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.k}, {self.controls[0]}, {self.targets[0]})"

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )
    """Size of the gate."""

    def inverse(self) -> Gate:
        return CRk_dagger(self.parameters[0], self.controls[0], self.targets[0])


class CRk_dagger(RotationGate, ControlledGate):
    r"""Two-qubit Controlled-Rk-dagger gate.

    `\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&e^{-i\pi/2^{k-1}}\end{pmatrix}`

    Args:
        k: Parameter used in the definition of the phase to apply.
        control: Index referring to the qubit used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> pprint(CRk_dagger(4, 0, 1).to_matrix())
        [[1, 0, 0, 0               ],
         [0, 1, 0, 0               ],
         [0, 0, 1, 0               ],
         [0, 0, 0, 0.92388-0.38268j]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.CPhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import CPhaseGate

        return CPhaseGate

    # TODO: this is a special case, see if it needs to be generalized
    qlm_aqasm_keyword = "CNOT;PH"
    qiskit_string = "cp"

    def __init__(self, k: Expr | int, control: int, target: int):
        self.parameters = [k]
        ControlledGate.__init__(self, [control], [target], Rk_dagger(k, target), "CRk†")
        definition = UnitaryMatrix(
            self.to_canonical_matrix(), **self.native_gate_options
        )
        ParametrizedGate.__init__(self, definition, [target], [k], "CRk†")

    @property
    def theta(self) -> Expr | float:
        r"""Value of the rotation angle, parametrized by ``k`` with the relation
        `\theta = -\frac{\pi}{2^{k-1}}`."""
        from sympy import pi

        p = np.pi if isinstance(self.k, Integral) else pi
        return -(p / 2 ** (self.k - 1))  # pyright: ignore[reportOperatorIssue]

    @property
    def k(self) -> Expr | int:
        """See corresponding argument."""
        return self.parameters[0]

    def to_canonical_matrix(self):
        e = exp(self.theta * 1j)  # pyright: ignore[reportOperatorIssue]
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, e]])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.k}, {self.controls[0]}, {self.targets[0]})"

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )
    """Size of the gate."""

    def inverse(self) -> Gate:
        return CRk(self.k, self.controls[0], self.targets[0])

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QASM2:
            from mpqp.qasm.mpqp_to_qasm import float_to_qasm_str

            instruction_str = self.qasm2_gate
            instruction_str += (
                f"({float_to_qasm_str(-2 * np.pi / (2 ** float(self.k)))})"
            )

            qubits = ",".join([f"q[{j}]" for j in self.controls]) + ","
            qubits += ",".join([f"q[{j}]" for j in self.targets])

            return instruction_str + " " + qubits + ";"
        else:
            return super().to_other_language(language, qiskit_parameters)


class TOF(InvolutionGate, ControlledGate, NoParameterGate):
    r"""Three-qubit Controlled-Controlled-NOT gate, also known as Toffoli Gate.

    `\begin{pmatrix}1&0&0&0&0&0&0&0\\0&1&0&0&0&0&0&0\\0&0&1&0&0&0&0&0\\0&0&0&1&0&0&0&0\\0&0&0&0&1&0&0&0\\0&0&0&0&0&1&0&0\\0&0&0&0&0&0&0&1\\0&0&0&0&0&0&1&0\end{pmatrix}`

    Args:
        control: List of indices referring to the qubits used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Examples:
        >>> pprint(TOF([0, 1], 2).to_matrix())
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 0]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.CCNot

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import CCXGate

        return CCXGate

    qlm_aqasm_keyword = "CCNOT"
    qiskit_string = "ccx"

    def __init__(self, control: list[int], target: int):
        if len(control) != 2:
            raise ValueError("A Toffoli gate must have exactly 2 control qubits.")
        ControlledGate.__init__(self, control, [target], X(target), "TOF")

    def to_canonical_matrix(self):
        m = np.identity(8, dtype=complex)
        m[-2:, -2:] = np.ones(2) - np.identity(2)
        return m

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        3
    )
    """Size of the gate."""


NATIVE_GATES = [
    cls
    for _, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if issubclass(cls, NativeGate)
    and not any("ABC" in base.__name__ for base in cls.__bases__)
]
"""All concrete native gates."""
