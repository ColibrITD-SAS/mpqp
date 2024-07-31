from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt

from .generics import Matrix


def state_vector_ket_shape(sv: npt.NDArray[np.complex64]) -> str:
    """Formats a state vector into its ket format."""
    if len(sv.shape) != 1:
        raise ValueError(f"Input state {sv} should be a vector (1 dimensional matrix).")
    nb_qubits = int(np.log2(len(sv)))
    if 2**nb_qubits != len(sv):
        raise ValueError(f"Input state {sv} should have a power of 2 size")
    return (
        " ".join(
            f"{with_sign(v)}|{np.binary_repr(i,nb_qubits)}⟩"
            for i, v in enumerate(sv)
            if v.round(3) != 0
        )
    )[2:]


def with_sign(val: np.complex64) -> str:
    """Sometimes, we want values under a specific format, in particular
    ``<sign> <value>``. Where value is as simple as possible (*e.g.* no period
    or no imaginary part if there is no need).

    Args:
        val: The value to be formatted.

    Returns:
        The formatted value
    """
    rounded = _remove_null_imag(val)
    if rounded == 1:
        return "+ "
    if rounded.real == 0:
        sign = "+ " if rounded.imag >= 0 else "- "
        rounded = _remove_unnecessary_decimals(abs(rounded))
        return sign + str(rounded) + "j"
    str_rounded = str(rounded)
    if str_rounded.startswith("-") or str_rounded.startswith("(-"):
        return "- " + str(-rounded)
    return "+ " + str_rounded


def _remove_null_imag(val: np.complex64) -> np.complex64 | np.float32 | int:
    val = np.round(val, 3)
    if val.imag != 0:
        return val
    return _remove_unnecessary_decimals(val.real)


def _remove_unnecessary_decimals(val: np.float32 | int) -> np.float32 | int:
    val = np.float32(val)
    if val.is_integer():
        return int(val)
    return val


def clean_1D_array(
    array: list[complex] | npt.NDArray[np.complex64 | np.float32],
) -> str:
    """Cleans and formats elements of an array.
    This function rounds the real parts of complex numbers in the array to 7 decimal places
    and formats them as integers if they are whole numbers. It returns a string representation
    of the cleaned array without parentheses.

    Args:
        array: An array containing numeric elements, possibly including complex numbers.

    Returns:
        A string representation of the cleaned array without parentheses.

    Example:
        >>> clean_1D_array([1.234567895546, 2.3456789645645, 3.45678945645])
        '[1.2345679, 2.345679, 3.4567895]'
        >>> clean_1D_array([1+2j, 3+4j, 5+6j])
        '[1+2j, 3+4j, 5+6j]'
        >>> clean_1D_array([1+0j, 0+0j, 5.])
        '[1, 0, 5]'
        >>> clean_1D_array([1.0, 2.0, 3.0])
        '[1, 2, 3]'

    """
    array = np.array(array, dtype=np.complex64)
    cleaned_array = [
        (
            int(element.real)
            if int(element.real) == element
            else (
                np.round(element.real, 7)
                if (np.imag(element) == 0)
                else (str(np.round(element, 7)).replace("(", "").replace(")", ""))
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
    cleaned_matrix = [clean_1D_array(row) for row in matrix]
    return "[" + ",\n ".join(cleaned_matrix) + "]"


def pretty_print_matrix(matrix: Matrix):
    print(clean_matrix(matrix))


def one_lined_repr(obj: object):
    """One-liner returning a representation of the given object by removing
    extra whitespace.

    Args:
        obj: The object for which a representation is desired.
    """
    return re.sub(r"\s+", " ", repr(obj))
