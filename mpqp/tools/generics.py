from __future__ import annotations
import re
from typing import Iterator, Sequence, TypeVar, Union, Callable, Iterable
import numpy as np
import numpy.typing as npt
from typeguard import typechecked

T = TypeVar("T")
ListOrSingle = Union[list[T], T]
"""Type alias for both elements of type ``T``, or list of elements of type ``T``."""
ArbitraryNestedSequence = Union[Sequence["ArbitraryNestedSequence"], T]
"""Since arbitrarily nested list are defined by recursion, this type allow us to 
define a base case.

Examples:
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
            yield from flatten_generator(el)  # type: ignore
    else:
        yield lst


@typechecked
def flatten(lst: ArbitraryNestedSequence[T]) -> list[T]:
    """Flattens an arbitrarily nested list.

    Example:
        >>> nested_list = [[1, 2, [3, 4]], [5, [6, 7]], 8]
        >>> flatten(nested_list)
        [1, 2, 3, 4, 5, 6, 7, 8]

    Args:
        lst: The list, or nested list, to be flattened.

    Returns:
        A flattened list containing all elements from the input list.
    """
    return list(flatten_generator(lst))


def one_lined_repr(obj: object):
    """One-liner returning a representation of the given object by removing
    extra whitespace.

    Args:
        obj: The object for which a representation is desired.
    """
    return re.sub(r"\s+", " ", repr(obj))


@typechecked
def find(iterable: Iterable[T], oracle: Callable[[T], bool]) -> T:
    """
    Finds the first element in the iterable that satisfies the given oracle.

    Example:
        >>> numbers = [1, 2, 3, 4, 5]
        >>> is_even = lambda x: x % 2 == 0
        >>> find(numbers, is_even)
        2

    Args:
        iterable: The iterable to search for the element.
        oracle: A callable function that takes an element and returns ``True``
            if the element satisfies the condition.

    Returns:
        The first element in the iterable that satisfies the oracle.

    Raises:
        ValueError: If no element in the iterable satisfies the given oracle.
    """
    for item in iterable:
        if oracle(item):
            return item
    raise ValueError("No objects satisfies the given oracle")
