import doctest
import os
import numpy as np
from mpqp.all import *
from mpqp.execution import BatchResult
from mpqp.execution import remote_result_from_id
from mpqp.core.instruction.measurement import pauli_string
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.tools.maths import is_unitary, is_hermitian, normalize
from mpqp.tools.generics import find, flatten
from mpqp.qasm import open_qasm_2_to_3

test_globals = globals().copy()
test_globals.update(locals())

def run_doctests_in_folder(folder_path):
    """
    Run doctests on all Python files in the specified folder and its subfolders.

    :param folder_path: Path to the folder containing Python files.
    """
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join("..\\"+root, filename)
                print(f"Running doctests in {os.path.join(os.getcwd(),root,filename)}")
                doctest.testfile(file_path, globs=test_globals)


# Specify the path to the main folder containing Python files and subfolders
main_folder_path = "mpqp"

# Run doctests on all Python files in the main folder and its subfolders
run_doctests_in_folder(main_folder_path)
