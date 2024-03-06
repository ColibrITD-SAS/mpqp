"""Native gates is the set of all gates natively supported in OpenQASM. Since we
rely on this standard, all of them are indeed implemented. In addition, this
module contains a few abstract classes used to factorize the behaviors common to
a lot of gates. Feel free to use them for your own custom gates!"""

from __future__ import annotations

from abc import ABC
from numbers import Integral
from typing import Optional

import numpy as np
import numpy.typing as npt
from qiskit.circuit import Parameter
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
    UGate,
    XGate,
    YGate,
    ZGate,
)
from sympy import Expr, pi

# pylance doesn't handle well Expr, so a lot of "type:ignore" will happen in
# this file :/
from typeguard import typechecked

from mpqp.core.instruction.gates.controlled_gate import ControlledGate
from mpqp.core.instruction.gates.gate import Gate, InvolutionGate, SingleQubitGate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
from mpqp.core.languages import Language
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import cos, exp, sin


@typechecked
def _qiskit_parameter_adder(param: Expr | float, qiskit_parameters: set[Parameter]):
    """To avoid having several parameters in qiskit for the same value we keep
    track of them in a set. This function takes care of this, this way you can
    directly call `QiskitGate(_qiskit_parameter_adder(<param>, <q_params_set>))`
    without having to manually take care of the de-duping.

    Args:
        param: The parameter you need for your qiskit gate.
        qiskit_parameters: The set of previously set qiskit parameters. This set
        is updated inplace.

    Returns:

    """
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
            qiskit_param = Parameter(name)
            qiskit_parameters.add(qiskit_param)
    else:
        qiskit_param = param
    return qiskit_param


@typechecked
class NativeGate(Gate, ABC):
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
class RotationGate(NativeGate, ParametrizedGate, ABC):
    """Many gates can be classified as a simple rotation gate, around a specific
    axis (and potentially with a control qubit). All those gates have in common
    a single parameter: ``theta``. This class help up factorize this behavior,
    and simply having to tweak the matricial semantics and qasm translation of
    the specific gate.

    Args:
        theta: Angle of the rotation.
        target: Index referring to the qubits on which the gate will be applied.
    """

    qiskit_gate: type[RXGate | RYGate | RZGate | PhaseGate | CPhaseGate]

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
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        if qiskit_parameters is None:
            qiskit_parameters = set()
        theta = float(self.theta) if self._numeric_parameters else self.theta
        if language == Language.QISKIT:
            return self.qiskit_gate(_qiskit_parameter_adder(theta, qiskit_parameters))
        else:
            raise NotImplementedError(f"Error: {language} is not supported")


@typechecked
class NoParameterGate(NativeGate, ABC):
    """Class describing native gates that do not depend on parameters.

    Args:
        targets: List of indices referring to the qubits on which the gate will
            be applied.
        label: Label used to identify the gate.
    """

    qiskit_gate: type[
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
    ]
    """Corresponding ``qiskit``'s gate class."""
    matrix: npt.NDArray[np.complex64]
    """Matricial semantics of the gate."""

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        if language == Language.QISKIT:
            return self.qiskit_gate()
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def to_matrix(self) -> Matrix:
        return self.matrix


@typechecked
class OneQubitNoParamGate(SingleQubitGate, NoParameterGate, ABC):
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
        array([[1, 0],
               [0, 1]])
    """

    qiskit_gate = IGate
    matrix = np.eye(2)


class X(OneQubitNoParamGate, InvolutionGate):
    """One qubit X (NOT) Pauli gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> X(0).to_matrix()
        array([[0, 1],
               [1, 0]])
    """

    qiskit_gate = XGate
    matrix = np.array([[0, 1], [1, 0]])


class Y(OneQubitNoParamGate, InvolutionGate):
    """One qubit Y Pauli gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> Y(0).to_matrix()
        array([[ 0.+0.j, -0.-1.j],
               [ 0.+1.j,  0.+0.j]])
    """

    qiskit_gate = YGate
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

    qiskit_gate = ZGate
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

    qiskit_gate = HGate
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

    qiskit_gate = PhaseGate

    def to_matrix(self) -> Matrix:
        return np.array([[1, 0], [0, exp(self.parameters[0] * 1j)]])  # type:ignore


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

    qiskit_gate = SGate
    matrix = np.array([[1, 0], [0, 1j]])


class T(OneQubitNoParamGate):
    """One qubit T gate. It consists in applying a phase of Pi/4.
    The T gate can be defined as the fourth-root of the Z (Pauli) gate.

    Args:
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> T(0).to_matrix()
        array([[1.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.70710678+0.70710678j]])
    """

    qiskit_gate = TGate
    matrix = np.array([[1, 0], [0, exp((pi / 4) * 1j)]])


class SWAP(InvolutionGate, NoParameterGate):
    """Two-qubit SWAP gate.

    Args:
        a: First target of the swapping operation.
        b: Second target of the swapping operation.

    Example:
        >>> SWAP(0, 1)
        array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])
    """

    qiskit_gate = SwapGate
    matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def __init__(self, a: int, b: int):
        super().__init__([a, b], "SWAP")


class U(NativeGate, ParametrizedGate):
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
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        if qiskit_parameters is None:
            qiskit_parameters = set()
        if language == Language.QISKIT:
            return UGate(
                theta=_qiskit_parameter_adder(self.theta, qiskit_parameters),
                phi=_qiskit_parameter_adder(self.phi, qiskit_parameters),
                lam=_qiskit_parameter_adder(self.gamma, qiskit_parameters),
            )
        else:
            raise NotImplementedError(f"Error: {language} is not supported")

    def to_matrix(self) -> Matrix:
        c, s, eg, ep = (
            cos(self.theta / 2),  # type:ignore
            sin(self.theta / 2),  # type:ignore
            exp(self.gamma * 1j),  # type:ignore
            exp(self.phi * 1j),  # type:ignore
        )
        return np.array([[c, -eg * s], [ep * s, eg * ep * c]])  # type:ignore


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

    qiskit_gate = RXGate

    def to_matrix(self) -> Matrix:
        c, s = cos(self.parameters[0] / 2), sin(self.parameters[0] / 2)  # type:ignore
        return np.array([[c, -1j * s], [-1j * s, c]])  # type:ignore


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

    qiskit_gate = RYGate

    def to_matrix(self) -> Matrix:
        c, s = cos(self.parameters[0] / 2), sin(self.parameters[0] / 2)  # type:ignore
        return np.array([[c, -s], [s, c]])


class Rz(RotationGate, SingleQubitGate):
    """One qubit rotation around the Z axis

    Args:
        theta: Parameter representing the angle of the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> Rz(np.pi/5, 1).to_matrix()
        array([[0.95105652-0.30901699j, 0.        +0.j        ],
               [0.        +0.j        , 0.95105652-0.30901699j]])
    """

    qiskit_gate = RZGate

    def to_matrix(self) -> Matrix:
        e = exp(-1j * self.parameters[0] / 2)  # type:ignore
        return np.array([[e, 0], [0, e]])


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

    qiskit_gate = PhaseGate

    def __init__(self, k: Expr | int, target: int):
        self.parameters = [k]
        definition = UnitaryMatrix(self.to_matrix(), **self.native_gate_options)
        ParametrizedGate.__init__(self, definition, [target], [self.k], "Rk")

    @property
    def theta(self) -> Expr | float:
        r"""Value of the rotation angle, parametrized by ``k`` with the relation
        `\theta = \frac{\pi}{2^{k-1}}`."""
        return pi / 2 ** (self.k - 1)  # type:ignore

    @property
    def k(self) -> Expr | float:
        """See corresponding argument."""
        return self.parameters[0]

    def to_matrix(self) -> Matrix:
        p = np.pi if isinstance(self.k, Integral) else pi
        e = exp(p * 1j / 2 ** (self.k - 1))  # type:ignore
        return np.array([[1, 0], [0, e]])

    def __repr__(self):
        return f"{type(self).__name__}({self.k}, {self.targets[0]})"


class CNOT(InvolutionGate, NoParameterGate, ControlledGate):
    """
    Two-qubit Controlled-NOT gate.

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

    qiskit_gate = CXGate

    def __init__(self, control: int, target: int):
        ControlledGate.__init__(self, [control], [target], X(target), "CNOT")

    def to_matrix(self) -> Matrix:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class CZ(InvolutionGate, NoParameterGate, ControlledGate):
    """Two-qubit Controlled-Z gate.

    Args:
        control: Index referring to the qubit used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> CZ(0, 1).to_matrix()
        array([[ 1,  0,  0,  0],
               [ 0,  1,  0,  0],
               [ 0,  0,  1,  0],
               [ 0,  0,  0, -1]])
    """

    qiskit_gate = CZGate

    def __init__(self, control: int, target: int):
        ControlledGate.__init__(self, [control], [target], Z(target), "CZ")

    def to_matrix(self) -> Matrix:
        m = np.eye(8, dtype=complex)
        m[-1, -1] = -1
        return m


class CRk(RotationGate, ControlledGate):
    """Two-qubit Controlled-Rk gate.

    Args:
        k: Parameter used in the definition of the phase to apply.
        control: Index referring to the qubit used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> CRk(4, 0, 1)
        array([[1.        +0.j        , 0.        +0.j        , 0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 1.        +0.j        , 0.        +0.j        , 0.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        , 1.        +0.j        , 1.        +0.j        ],
               [0.        +0.j        , 0.        +0.j        , 0.        +0.j        , 0.92387953+0.38268343j]])
    """

    qiskit_gate = CPhaseGate

    def __init__(self, k: Expr | int, control: int, target: int):
        self.parameters = [k]
        ControlledGate.__init__(self, [control], [target], Rk(k, target), "CRk")
        definition = UnitaryMatrix(self.to_matrix(), **self.native_gate_options)
        ParametrizedGate.__init__(self, definition, [target], [k], "CRk")

    @property
    def theta(self) -> Expr | float:
        r"""Value of the rotation angle, parametrized by ``k`` with the relation
        `\theta = \frac{\pi}{2^{k-1}}`."""
        p = np.pi if isinstance(self.k, Integral) else pi
        return p / 2 ** (self.k - 1)  # type:ignore

    @property
    def k(self) -> Expr | float:
        """See corresponding argument."""
        return self.parameters[0]

    def to_matrix(self) -> Matrix:
        e = exp(self.theta * 1j)  # type:ignore
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, e]])

    def __repr__(self):
        return f"{type(self).__name__}({self.k}, {self.controls[0]}, {self.targets[0]})"


class TOF(InvolutionGate, NoParameterGate, ControlledGate):
    """Three-qubit Controlled-Controlled-NOT gate, also known as Toffoli Gate

    Args:
        control: List of indices referring to the qubits used to control the gate.
        target: Index referring to the qubit on which the gate will be applied.

    Example:
        >>> TOF([0, 1], 2).to_matrix()
        array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 1, 0]])
    """

    qiskit_gate = CCXGate

    def __init__(self, control: list[int], target: int):
        if len(control) != 2:
            raise ValueError("A Toffoli gate must have exactly 2 control qubits.")
        ControlledGate.__init__(self, control, [target], X(target), "TOF")

    def to_matrix(self) -> Matrix:
        m = np.identity(8, dtype=complex)
        m[-2:, -2:] = np.ones(2) - np.identity(2)
        return m


NATIVE_GATES = [CNOT, CRk, CZ, H, Id, P, Rk, Rx, Ry, Rz, S, SWAP, T, TOF, U, X, Y, Z]
# TODO : check the possibility to detect when a custom gate can be defined as a native gate, problem with
#  parametrized gates maybe
