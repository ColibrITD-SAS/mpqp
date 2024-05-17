from __future__ import annotations

import re
from typing import Callable, Iterable, Iterator, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

T = TypeVar("T")
OneOrMany = Union[T, Sequence[T]]
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

    Args:
        lst: The list, or nested list, to be flattened.

    Returns:
        A flattened list containing all elements from the input list.

    Example:
        >>> nested_list = [[1, 2, [3, 4]], [5, [6, 7]], 8]
        >>> flatten(nested_list)
        [1, 2, 3, 4, 5, 6, 7, 8]

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
    """Finds the first element in the iterable that satisfies the given oracle.

    Args:
        iterable: The iterable to search for the element.
        oracle: A callable function that takes an element and returns ``True``
            if the element satisfies the condition.

    Returns:
        The first element in the iterable that satisfies the oracle.

    Raises:
        ValueError: If no element in the iterable satisfies the given oracle.

    Example:
        >>> numbers = [1, 2, 3, 4, 5]
        >>> is_even = lambda x: x % 2 == 0
        >>> find(numbers, is_even)
        2

    """
    for item in iterable:
        if oracle(item):
            return item
    raise ValueError("No objects satisfies the given oracle")


def clean_array(array: list[complex] | npt.NDArray[np.complex64 | np.float32]):
    """Cleans and formats elements of an array.
    This function rounds the real parts of complex numbers in the array to 7 decimal places
    and formats them as integers if they are whole numbers. It returns a string representation
    of the cleaned array without parentheses.

    Args:
        array: An array containing numeric elements, possibly including complex numbers.

    Returns:
        str: A string representation of the cleaned array without parentheses.

    Example:
        >>> clean_array([1.234567895546, 2.3456789645645, 3.45678945645])
        '[1.2345679, 2.345679, 3.4567895]'
        >>> clean_array([1+2j, 3+4j, 5+6j])
        '[1+2j, 3+4j, 5+6j]'
        >>> clean_array([1+0j, 0+0j, 5.])
        '[1, 0, 5]'
        >>> clean_array([1.0, 2.0, 3.0])
        '[1, 2, 3]'

    """
    cleaned_array = [
        (
            int(element.real)
            if (np.iscomplexobj(element) or isinstance(element, float))
            and int(element.real) == element
            else (
                np.round(element.real, 7)
                if (np.iscomplexobj(element) and np.imag(element) == 0)
                or isinstance(element, float)
                else (
                    str(np.round(element, 7)).replace("(", "").replace(")", "")
                    if np.iscomplexobj(element)
                    else element
                )
            )
        )
        for element in array
    ]
    return "[" + ", ".join(map(str, cleaned_array)) + "]"


def clean_matrix(matrix: Matrix):
    """Cleans and formats elements of a matrix.
    This function cleans and formats the elements of a matrix. It rounds the real parts of complex numbers
    in the matrix to 7 decimal places and formats them as integers if they are whole numbers. It returns a
    string representation of the cleaned matrix without parentheses.

    Args:
        matrix: A matrix containing numeric elements, possibly including complex numbers.

    Returns:
        str: A string representation of the cleaned matrix without parentheses.

    Examples:
        >>> print(clean_matrix([[1.234567895546, 2.3456789645645, 3.45678945645],
        ...               [1+0j, 0+0j, 5.],
        ...               [1.0, 2.0, 3.0]]))
        [[1.2345679, 2.345679, 3.4567895],
         [1, 0, 5],
         [1, 2, 3]]

    """
    # TODO: add an option to align cols
    cleaned_matrix = [clean_array(row) for row in matrix]
    return "[" + ",\n ".join(cleaned_matrix) + "]"
