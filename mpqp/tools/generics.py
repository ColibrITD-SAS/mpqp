"""This module contains a collection of generic types and functions needed 
across the library.

Type aliases :obj:`Matrix`, :obj:`OneOrMany` and :obj:`ArbitraryNestedSequence`
are used across the library. In particular, :obj:`ArbitraryNestedSequence` is
used in cases the nesting of the sequence is unknown; and in these cases you
might want to flatten the list using :func:`flatten`.

On occasion, there is also a need to "flatten" the string representation of an 
object *i.e.* to display it on one line. In this case :func:`one_line_repr` is
your friend.

Lastly, we find the default list search mechanism in python a bit too
restrictive. :func:`find` allow us a much more versatile search using an 
``oracle``.
"""

from __future__ import annotations

from abc import ABCMeta
from inspect import getsource
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from aenum import Enum


# This is needed because for some reason pyright does not understand that Enum
# is a class (probably because Enum does weird things to the Enum class)
if TYPE_CHECKING:
    from enum import Enum
    from mpqp.core.circuit import QCircuit

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

T = TypeVar("T")
"""A generic type."""
OneOrMany = Union[T, Sequence[T]]
"""Type alias for single elements of type :obj:`T`, or sequences of such 
elements."""
ArbitraryNestedSequence = Union[Sequence["ArbitraryNestedSequence[T]"], T]
"""This type alias allows us to define heterogeneously nested Sequences of 
:obj:`T`.

Examples:
    >>> l: ArbitraryNestedSequence[int]
    >>> l = [[0,1],0,[[0,2],3]]
    >>> l = 1
    >>> l = [2,1,3]

"""
Matrix = Union[npt.NDArray[np.complex64], npt.NDArray[np.object_]]
"""Type alias denoting all the matrices we consider (either matrices of complex 
or of ``sympy`` expressions, given to ``numpy`` as objects)"""


@typechecked
def flatten_generator(lst: ArbitraryNestedSequence[T]) -> Iterator[T]:
    """Helper generator function for flattening an arbitrarily nested list.

    Args:
        lst: The list, or nested list, to be flattened.

    Yields:
        Elements from the input list in a flattened order.
    """
    if isinstance(lst, Sequence):
        for el in lst:
            yield from flatten_generator(el)
    else:
        yield lst


@typechecked
def flatten(lst: ArbitraryNestedSequence[T]) -> list[T]:
    """Flattens an arbitrarily nested Sequence.

    This is a wrapper around :func:`flatten_generator`.

    Args:
        lst: The nested sequence, to be flattened.

    Returns:
        A flattened list containing all elements from the input list.

    Example:
        >>> nested_list = [[1, 2, [3, 4]], [5, [6, 7]], 8]
        >>> flatten(nested_list)
        [1, 2, 3, 4, 5, 6, 7, 8]

    """
    return list(flatten_generator(lst))


@typechecked
def find(sequence: Sequence[T], oracle: Callable[[T], bool]) -> T:
    """Finds the first element in the sequence that satisfies the given oracle.

    Args:
        sequence: The sequence to search for the element.
        oracle: A callable function that takes an element and returns ``True``
            if the element satisfies the condition.

    Returns:
        The first element in the sequence that satisfies the oracle.

    Raises:
        ValueError: If no element in the sequence satisfies the given oracle.

    Example:
        >>> numbers = [1, 2, 3, 4, 5]
        >>> is_even = lambda x: x % 2 == 0
        >>> find(numbers, is_even)
        2

    """
    return sequence[find_index(sequence, oracle)]


def find_index(iterable: Iterable[T], oracle: Callable[[T], bool]) -> int:
    """Finds the index of the first element in the iterable that satisfies the
    given oracle.

    Args:
        iterable: The iterable to search for the element.
        oracle: A callable function that takes an element and returns ``True``
            if the element satisfies the condition.

    Returns:
        The index of the first element in the iterable that satisfies the oracle.

    Raises:
        ValueError: If no element in the iterable satisfies the given oracle.

    Example:
        >>> numbers = [1, 2, 3, 4, 5]
        >>> is_even = lambda x: x % 2 == 0
        >>> find(numbers, is_even)
        2

    """
    for index, item in enumerate(iterable):
        if oracle(item):
            return index
    raise ValueError("No objects satisfies the given oracle")


def random_circuit(
    gate_classes: Optional[list[type]] = None,
    nb_qubits: int = 5,
    nb_gates: int = np.random.randint(5, 10),
):
    """This function creates a QCircuit with a specified number of qubits and gates.
    The gates are chosen randomly from the provided list of native gate classes.

    args:
        nb_qubits : Number of qubits in the circuit.
        gate_classes : List of native gate classes to use in the circuit.
        nb_gates : Number of gates to add to the circuit. Defaults to a random
         integer between 5 and 10.

    Returns:
        A quantum circuit with the specified number of qubits and randomly chosen gates.

    Raises:
        ValueError: If the number of qubits is too low for the specified gates.

    Examples:
        >>> random_circuit([U, TOF], 3) # doctest: +SKIP
        Generates a random quantum circuit with 3 qubits using U and TOF gates.
        >>> from mpqp.core.instruction.gates import native_gates
        >>> random_circuit(native_gates.NATIVE_GATES, 4, 10) # doctest: +SKIP
        Generates a random quantum circuit with 4 qubits and 10 gates native gates.
    """
    from mpqp.core.instruction.gates.native_gates import NATIVE_GATES
    from mpqp.core.circuit import QCircuit
    from mpqp.core.instruction.gates.gate import SingleQubitGate
    from mpqp.core.instruction.gates.native_gates import U, TOF, OneQubitNoParamGate
    from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
    import random

    if gate_classes is None:
        gate_classes = NATIVE_GATES

    qubits = list(range(nb_qubits))
    qcircuit = QCircuit(nb_qubits)
    if any(
        not issubclass(gate, SingleQubitGate)
        and ((gate == TOF and nb_qubits <= 2) or nb_qubits <= 1)
        for gate in gate_classes
    ):
        raise ValueError("number of qubits to low for this gates")

    for _ in range(nb_gates):
        gate_class = random.choice(gate_classes)
        target = random.choice(qubits)
        if issubclass(gate_class, SingleQubitGate):
            if issubclass(gate_class, ParametrizedGate):
                if gate_class == U:  # type: ignore[reportUnnecessaryComparison]
                    theta = float(random.uniform(0, 2 * np.pi))
                    phi = float(random.uniform(0, 2 * np.pi))
                    gamma = float(random.uniform(0, 2 * np.pi))
                    qcircuit.add(U(theta, phi, gamma, target))
                else:
                    qcircuit.add(gate_class(float(random.uniform(0, 2 * np.pi)), target))  # type: ignore[reportCallIssue]
            elif issubclass(gate_class, OneQubitNoParamGate):
                qcircuit.add(gate_class(target))
        else:
            control = random.choice(qubits)
            while control == target:
                control = random.choice(qubits)
            if issubclass(gate_class, ParametrizedGate):
                qcircuit.add(gate_class(float(random.uniform(0, 2 * np.pi)), control, target))  # type: ignore[reportArgumentType]
            elif gate_class == TOF:
                control2 = random.choice(qubits)
                while control2 == target or control2 == control:
                    control = random.choice(qubits)
                qcircuit.add(gate_class([control, control2], target))
            else:
                qcircuit.add(gate_class(control, target))
    return qcircuit


def compute_expected_matrix(qcircuit: QCircuit):
    """
    Computes the expected matrix resulting from applying single-qubit gates
    in reverse order on a quantum circuit.

    args:
        qcircuit : The quantum circuit object containing instructions.

    returns:
        Expected matrix resulting from applying the gates.

    raises:
        ValueError: If any gate in the circuit is not a SingleQubitGate.
    """
    from mpqp.core.instruction.gates.gate import Gate, SingleQubitGate
    from sympy import N

    gates = [
        instruction
        for instruction in qcircuit.instructions
        if isinstance(instruction, Gate)
    ]
    nb_qubits = qcircuit.nb_qubits

    result_matrix = np.eye(2**nb_qubits, dtype=complex)

    for gate in reversed(gates):
        if not isinstance(gate, SingleQubitGate):
            raise ValueError(
                f"Unsupported gate: {type(gate)} only SingleQubitGate can be computed for now"
            )
        matrix = np.eye(2**nb_qubits, dtype=complex)
        gate_matrix = gate.to_matrix()
        index = gate.targets[0]
        matrix = np.kron(
            np.eye(2**index, dtype=complex),
            np.kron(gate_matrix, np.eye(2 ** (nb_qubits - index - 1), dtype=complex)),
        )

        result_matrix = np.dot(result_matrix, matrix)

    return np.vectorize(N)(result_matrix).astype(complex)


class SimpleClassReprMeta(type):
    """Metaclass used to change the repr of the class (not the instances) to
    display the name of the class only (instead of the usual
    <class mpqp.path.ClassName>)"""

    def __repr__(cls):
        return cls.__name__


class SimpleClassReprABCMeta(SimpleClassReprMeta, ABCMeta):
    pass


class SimpleClassReprABC(metaclass=SimpleClassReprABCMeta):
    """This class is the equivalent of ABC (it signifies that it's subclass
    isn't meant to be instantiated directly), but it adds the small feature of
    setting the ``repr`` to be the class name, which is for instance useful for
    gates."""

    pass


class classproperty:
    """Decorator yo unite the ``classmethod`` and ``property`` decorators."""

    def __init__(self, func: Callable[..., Any]):
        self.fget = func

    def __get__(self, instance: object, owner: object):
        return self.fget(owner)


def _get_doc(enum: type[Any], member: str):
    src = getsource(enum)
    member_pointer = src.find(member)
    docstr_start = member_pointer + src[member_pointer:].find('"""') + 3
    docstr_end = docstr_start + src[docstr_start:].find('"""')
    return src[docstr_start:docstr_end]


class MessageEnum(Enum):
    """Enum subclass allowing you to access the docstring of the members of your
    enum through the ``message`` property.

    Example:
        >>> class A(MessageEnum):  # doctest: +SKIP
        ...     '''an enum'''
        ...     VALUE1 = auto()
        ...     '''member VALUE1'''
        ...     VALUE2 = auto()
        ...     '''member VALUE2'''
        >>> A.VALUE1.message  # doctest: +SKIP
        'member VALUE1'

    Warning:
        This implementation is not very robust, in particular, in case some
        members are not documented, it will mess things up. In addition, this
        can only work for code in file, and will not work in the interpreter.
    """

    message: str
    """Each of the members of the eum will have the ``message`` attribute."""

    def __init__(self, *args: Any, **kwds: dict[str, Any]) -> None:
        super().__init__(*args, **kwds)
        for member in type(self).__members__:
            type(self).__members__[member].message = _get_doc(type(self), member)
