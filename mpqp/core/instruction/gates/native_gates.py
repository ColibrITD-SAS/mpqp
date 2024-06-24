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
    This class represent all those gates supported by default.

    Args:
        targets: List of indices referring to the qubits on which the gate will
            be applied.
        label: Label used to identify the gate.
    """

    native_gate_options = {"disable_symbol_warn": True}


@typechecked
class RotationGate(NativeGate, ParametrizedGate, SimpleClassReprABC):
    """Many gates can be classified as a simple rotation gate, around a specific
    axis (and potentially with a control qubit). All those gates have in common
    a single parameter: ``theta``. This class help up factorize this behavior,
    and simply having to tweak the matrix semantics and qasm translation of
    the specific gate.

    Args:
        theta: Angle of the rotation.
        target: Index referring to the qubits on which the gate will be applied.
    """

    if TYPE_CHECKING:
        from braket.circuits import gates
        from qiskit.circuit.library import CPhaseGate, PhaseGate, RXGate, RYGate, RZGate

    @classproperty
    @abstractmethod
    def qiskit_gate(cls) -> type[RXGate | RYGate | RZGate | PhaseGate | CPhaseGate]:
        pass

    @classproperty
    @abstractmethod
    def braket_gate(
        cls,
    ) -> type[gates.Rx | gates.Ry | gates.Rz | gates.PhaseShift | gates.CPhaseShift]:
        pass

    def __init__(self, theta: Expr | float, target: int):
        self.parameters = [theta]
        definition = UnitaryMatrix(self.to_matrix(), **self.native_gate_options)
        ParametrizedGate.__init__(
            self, definition, [target], [self.theta], type(self).__name__.capitalize()
        )

    @property
    def theta(self):
        return self.parameters[0]

    def __repr__(self):
        return (
            f"{type(self).__name__}({f'{self.theta}'.rstrip('0')}, {self.targets[0]})"
        )

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if qiskit_parameters is None:
            qiskit_parameters = set()
        theta = float(self.theta) if self._numeric_parameters else self.theta
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
        else:
            raise NotImplementedError(f"Error: {language} is not supported")


@typechecked
class NoParameterGate(NativeGate, SimpleClassReprABC):
    """Class describing native gates that do not depend on parameters.

    Args:
        targets: List of indices referring to the qubits on which the gate will
            be applied.
        label: Label used to identify the gate.
    """

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
        pass

    qlm_aqasm_keyword: str

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
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def to_matrix(self) -> Matrix:
        return self.matrix


@typechecked
class OneQubitNoParamGate(SingleQubitGate, NoParameterGate, SimpleClassReprABC):
    """Class describing one-qubit native gates that do not depend on parameters.

    Args:
        target: Index referring to the qubits on which the gate will be applied.
    """

    def __init__(self, target: int):
        SingleQubitGate.__init__(self, target, type(self).__name__)


class Id(OneQubitNoParamGate, InvolutionGate):
    """One qubit identity gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> Id(0).to_matrix()
        array([[1.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j]], dtype=complex64)

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.I

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import IGate

        return IGate

    def __init__(self, target: int):

        super().__init__(target)

        self.qlm_aqasm_keyword = "I"
        self.matrix = np.eye(2, dtype=np.complex64)


class X(OneQubitNoParamGate, InvolutionGate):
    """One qubit X (NOT) Pauli gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> X(0).to_matrix()
        array([[0, 1],
               [1, 0]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.X

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import XGate

        return XGate

    def __init__(self, target: int):

        super().__init__(target)

        self.qlm_aqasm_keyword = "X"
        self.matrix = np.array([[0, 1], [1, 0]])


class Y(OneQubitNoParamGate, InvolutionGate):
    """One qubit Y Pauli gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> Y(0).to_matrix()
        array([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Y

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import YGate

        return YGate

    def __init__(self, target: int):

        super().__init__(target)

        self.qlm_aqasm_keyword = "Y"

    matrix = np.array([[0, -1j], [1j, 0]])


class Z(OneQubitNoParamGate, InvolutionGate):
    """One qubit Z Pauli gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> Z(0).to_matrix()
        array([[ 1,  0],
               [ 0, -1]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Z

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import ZGate

        return ZGate

    def __init__(self, target: int):

        super().__init__(target)

        self.qlm_aqasm_keyword = "Z"

    matrix = np.array([[1, 0], [0, -1]])


class H(OneQubitNoParamGate, InvolutionGate):
    """One qubit Hadamard gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> H(0).to_matrix()
        array([[ 0.70710678,  0.70710678],
               [ 0.70710678, -0.70710678]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.H

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import HGate

        return HGate

    def __init__(self, target: int):

        super().__init__(target)

        self.qlm_aqasm_keyword = "H"

    matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


class P(RotationGate, SingleQubitGate):
    """One qubit parametrized Phase gate. Consist in a rotation around Z axis.

    Args:
        theta: Parameter representing the phase to apply.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> P(np.pi/3, 1).to_matrix()
        array([[1. +0.j       , 0. +0.j       ],
               [0. +0.j       , 0.5+0.8660254j]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.PhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import PhaseGate

        return PhaseGate

    def __init__(self, theta: Expr | float, target: int):

        super().__init__(theta, target)

        self.qlm_aqasm_keyword = "PH"

    def to_matrix(self) -> Matrix:
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


class S(OneQubitNoParamGate):
    """One qubit S gate. It's equivalent to ``P(pi/2)``.
    It can also be defined as the square-root of the Z (Pauli) gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> S(0).to_matrix()
        array([[1.+0.j, 0.+0.j],
               [0.+0.j, 0.+1.j]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.S

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import SGate

        return SGate

    def __init__(self, target: int):

        super().__init__(target)

        self.qlm_aqasm_keyword = "S"

    matrix = np.array([[1, 0], [0, 1j]])


class T(OneQubitNoParamGate):
    r"""One qubit T gate. It is also referred to as the `\pi/4` gate because it
    consists in applying the phase gate with a phase of `\pi/4`.

    The T gate can also be defined as the fourth-root of the Z (Pauli) gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> T(0).to_matrix()
        array([[1, 0],
               [0, exp(0.25*I*pi)]], dtype=object)

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.T

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import TGate

        return TGate

    def __init__(self, target: int):

        super().__init__(target)

        self.qlm_aqasm_keyword = "T"

    def to_matrix(self) -> Matrix:
        from sympy import pi

        return np.array([[1, 0], [0, exp((pi / 4) * 1j)]])


class SWAP(InvolutionGate, NoParameterGate):
    """Two-qubit SWAP gate.

    Args:
        a: First target of the swapping operation.
        b: Second target of the swapping operation.

    Example:
        >>> SWAP(0, 1).to_matrix()
        array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Swap

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import SwapGate

        return SwapGate

    def __init__(self, a: int, b: int):

        self.qlm_aqasm_keyword = "SWAP"

        self.matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        super().__init__([a, b], "SWAP")

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )


class U(NativeGate, ParametrizedGate, SingleQubitGate):
    """Generic one qubit unitary gate. It is parametrized by 3 Euler angles.

    Args:
        theta: Parameter representing the first angle of the gate U.
        phi: Parameter representing the second angle of the gate U.
        gamma: Parameter representing the third angle of the gate U.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> U(np.pi/3, 0, np.pi/4, 0).to_matrix()
        array([[ 0.8660254 +0.j        , -0.35355339-0.35355339j],
               [ 0.5       +0.j        ,  0.61237244+0.61237244j]])

    """

    def __init__(
        self,
        theta: Expr | float,
        phi: Expr | float,
        gamma: Expr | float,
        target: int,
    ):
        self.parameters = [theta, phi, gamma]
        definition = UnitaryMatrix(self.to_matrix(), **self.native_gate_options)
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
            from qiskit.circuit.library import UGate

            if qiskit_parameters is None:
                qiskit_parameters = set()

            return UGate(
                theta=_qiskit_parameter_adder(self.theta, qiskit_parameters),
                phi=_qiskit_parameter_adder(self.phi, qiskit_parameters),
                lam=_qiskit_parameter_adder(self.gamma, qiskit_parameters),
            )
        elif language == Language.BRAKET:
            from braket.circuits import gates
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

            return gates.U(self.theta, self.phi, self.gamma)
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def to_matrix(self) -> Matrix:
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

    qlm_aqasm_keyword = "U"


class Rx(RotationGate, SingleQubitGate):
    """One qubit rotation around the X axis

    Args:
        theta: Parameter representing the angle of the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> Rx(np.pi/5, 1).to_matrix()
        array([[0.95105652+0.j        , 0.        -0.30901699j],
               [0.        -0.30901699j, 0.95105652+0.j        ]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Rx

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import RXGate

        return RXGate

    def __init__(self, theta: Expr | float, target: int):

        super().__init__(theta, target)

        self.qlm_aqasm_keyword = "RX"

    def to_matrix(self) -> Matrix:
        c = cos(self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        s = sin(self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        return np.array(  # pyright: ignore[reportCallIssue]
            [[c, -1j * s], [-1j * s, c]]  # pyright: ignore[reportOperatorIssue]
        )


class Ry(RotationGate, SingleQubitGate):
    """One qubit rotation around the Y axis

    Args:
        theta: Parameter representing the angle of the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> Ry(np.pi/5, 1).to_matrix()
        array([[ 0.95105652, -0.30901699],
               [ 0.30901699,  0.95105652]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Ry

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import RYGate

        return RYGate

    def __init__(self, theta: Expr | float, target: int):

        super().__init__(theta, target)

        self.qlm_aqasm_keyword = "RY"

    def to_matrix(self) -> Matrix:
        c = cos(self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        s = sin(self.parameters[0] / 2)  # pyright: ignore[reportOperatorIssue]
        return np.array([[c, -s], [s, c]])


class Rz(RotationGate, SingleQubitGate):
    """One qubit rotation around the Z axis

    Args:
        theta: Parameter representing the angle of the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> print(clean_matrix(Rz(np.pi/5, 1).to_matrix()))
        [[0.9510565-0.309017j, 0],
         [0, 0.9510565+0.309017j]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.Rz

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import RZGate

        return RZGate

    def __init__(self, theta: Expr | float, target: int):

        super().__init__(theta, target)

        self.qlm_aqasm_keyword = "RZ"

    def to_matrix(self) -> Matrix:
        e = exp(
            -1j * self.parameters[0] / 2  # pyright: ignore[reportOperatorIssue]
        )  # pyright: ignore[reportArgumentType]
        return np.array(  # pyright: ignore[reportCallIssue]
            [[e, 0], [0, 1 / e]]  # pyright: ignore[reportOperatorIssue]
        )


class Rk(RotationGate, SingleQubitGate):
    r"""One qubit Phase gate of angle `\frac{2i\pi}{2^k}`.

    Args:
        k: Parameter used in the definition of the phase to apply.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> Rk(5, 0).to_matrix()
        array([[1.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.98078528+0.19509032j]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.PhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import PhaseGate

        return PhaseGate

    def __init__(self, k: Expr | int, target: int):

        self.qlm_aqasm_keyword = "PH"

        self.parameters = [k]
        definition = UnitaryMatrix(self.to_matrix(), **self.native_gate_options)
        ParametrizedGate.__init__(self, definition, [target], [self.k], "Rk")

    @property
    def theta(self) -> Expr | float:
        r"""Value of the rotation angle, parametrized by ``k`` with the relation
        `\theta = \frac{\pi}{2^{k-1}}`."""
        from sympy import pi

        return pi / 2 ** (self.k - 1)  # pyright: ignore[reportOperatorIssue]

    @property
    def k(self) -> Expr | float:
        """See corresponding argument."""
        return self.parameters[0]

    def to_matrix(self) -> Matrix:
        from sympy import pi

        p = np.pi if isinstance(self.k, Integral) else pi
        e = exp(p * 1j / 2 ** (self.k - 1))  # pyright: ignore[reportOperatorIssue]
        return np.array([[1, 0], [0, e]])

    def __repr__(self):
        return f"{type(self).__name__}({self.k}, {self.targets[0]})"


class CNOT(InvolutionGate, NoParameterGate, ControlledGate):
    """Two-qubit Controlled-NOT gate.

    Args:
        control: index referring to the qubit used to control the gate
        target: index referring to the qubit on which the gate will be applied

    Example:
        >>> CNOT(0, 1).to_matrix()
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]])

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.CNot

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import CXGate

        return CXGate

    def __init__(self, control: int, target: int):

        self.qlm_aqasm_keyword = "CNOT"
        ControlledGate.__init__(self, [control], [target], X(target), "CNOT")

    def to_matrix(self) -> Matrix:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )


class CZ(InvolutionGate, NoParameterGate, ControlledGate):
    """Two-qubit Controlled-Z gate.

    Args:
        control: Index referring to the qubit used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> print(clean_matrix(CZ(0, 1).to_matrix()))
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
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

    def __init__(self, control: int, target: int):

        self.qlm_aqasm_keyword = "CSIGN"
        ControlledGate.__init__(self, [control], [target], Z(target), "CZ")

    def to_matrix(self) -> Matrix:
        m = np.eye(4, dtype=complex)
        m[-1, -1] = -1
        return m

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )


class CRk(RotationGate, ControlledGate):
    """Two-qubit Controlled-Rk gate.

    Args:
        k: Parameter used in the definition of the phase to apply.
        control: Index referring to the qubit used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> print(clean_matrix(CRk(4, 0, 1).to_matrix()))
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0.9238795+0.3826834j]]

    """

    @classproperty
    def braket_gate(cls):
        from braket.circuits import gates

        return gates.CPhaseShift

    @classproperty
    def qiskit_gate(cls):
        from qiskit.circuit.library import CPhaseGate

        return CPhaseGate

    def __init__(self, k: Expr | int, control: int, target: int):

        self.qlm_aqasm_keyword = ["CNOT", "PH"]
        self.parameters = [k]
        ControlledGate.__init__(self, [control], [target], Rk(k, target), "CRk")
        definition = UnitaryMatrix(self.to_matrix(), **self.native_gate_options)
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

    def to_matrix(self) -> Matrix:
        e = exp(self.theta * 1j)  # pyright: ignore[reportOperatorIssue]
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, e]])

    def __repr__(self):
        return f"{type(self).__name__}({self.k}, {self.controls[0]}, {self.targets[0]})"

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        2
    )


class TOF(InvolutionGate, NoParameterGate, ControlledGate):
    """Three-qubit Controlled-Controlled-NOT gate, also known as Toffoli Gate

    Args:
        control: List of indices referring to the qubits used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> print(clean_matrix(TOF([0, 1], 2).to_matrix()))
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

    def __init__(self, control: list[int], target: int):

        self.qlm_aqasm_keyword = "CCNOT"
        if len(control) != 2:
            raise ValueError("A Toffoli gate must have exactly 2 control qubits.")
        ControlledGate.__init__(self, control, [target], X(target), "TOF")

    def to_matrix(self) -> Matrix:
        m = np.identity(8, dtype=complex)
        m[-2:, -2:] = np.ones(2) - np.identity(2)
        return m

    nb_qubits = (  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
        3
    )


NATIVE_GATES = [CNOT, CRk, CZ, H, Id, P, Rk, Rx, Ry, Rz, S, SWAP, T, TOF, U, X, Y, Z]
# 3M-TODO : check the possibility to detect when a custom gate can be defined as a native gate, problem with
#  parametrized gates maybe
