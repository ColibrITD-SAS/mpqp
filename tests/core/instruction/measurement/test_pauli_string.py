import numpy as np
import numpy.typing as npt
import pytest

from mpqp.core.instruction.measurement.pauli_string import I, PauliString
from mpqp.tools.maths import matrix_eq
from random import randint


def pauli_string_combinations():
    operation = ["@", "+", "-", "/", "*", "u-", "u+"]
    pauli = [
        ("I", "np.eye(2)"),
        ("(I@I)", "np.eye(4)"),
        ("(I+I)", "(2*np.eye(2))"),
    ]  # Assuming I is the identity matrix
    first_values = [True, False]
    result = []

    for ps_1 in pauli:
        for ps_2 in pauli:
            for op in operation:
                for first_val in first_values:
                    if first_val:
                        if op == "@":
                            result.append(
                                (
                                    eval(ps_2[0] + op + ps_1[0]),
                                    eval("np.kron(" + ps_2[1] + "," + ps_1[1] + ")"),
                                )
                            )
                        elif ps_1[0] == ps_2[0] and (op == "+" or op == "-"):
                            result.append(
                                (
                                    eval(ps_2[0] + op + ps_1[0]),
                                    eval(ps_2[1] + op + ps_1[1]),
                                )
                            )
                        elif op == "u-" or op == "u+":
                            result.append(
                                (eval(op[1] + ps_1[0]), eval(op[1] + ps_1[1]))
                            )
                        elif op != "/" and op != "+" and op != "-":
                            rd = randint(1, 5)
                            result.append(
                                (
                                    eval(str(rd) + op + ps_1[0]),
                                    eval(str(rd) + op + ps_1[1]),
                                )
                            )
                    else:
                        if op == "@":
                            result.append(
                                (
                                    eval(ps_1[0] + op + ps_2[0]),
                                    eval("np.kron(" + ps_1[1] + "," + ps_2[1] + ")"),
                                )
                            )
                        elif ps_1[0] == ps_2[0] and (op == "+" or op == "-"):
                            result.append(
                                (
                                    eval(ps_1[0] + op + ps_2[0]),
                                    eval(ps_1[1] + op + ps_2[1]),
                                )
                            )
                        elif op != "+" and op != "-" and op != "u+" and op != "u-":
                            rd = randint(1, 5)
                            result.append(
                                (
                                    eval(ps_1[0] + op + str(rd)),
                                    eval(ps_1[1] + op + str(rd)),
                                )
                            )
    return result


@pytest.mark.parametrize("ps, matrix", pauli_string_combinations())
def test_operations(ps: PauliString, matrix: npt.NDArray[np.complex64]):
    assert matrix_eq(ps.to_matrix(), matrix)
