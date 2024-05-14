import doctest
import os

import numpy as np

import mpqp
from mpqp.all import *
from mpqp.core.instruction.measurement import pauli_string
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.execution import BatchResult, remote_result_from_id
from mpqp.qasm import open_qasm_2_to_3
from mpqp.tools.generics import find, flatten
from mpqp.tools.maths import is_hermitian, is_unitary, normalize

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
                file_path = os.path.join("..\\" + root, filename)
                print(f"Running doctests in {os.path.join(os.getcwd(),root,filename)}")
                doctest.testfile(file_path, globs=test_globals)


# Specify the path to the main folder containing Python files and subfolders
main_folder_path = "mpqp"

# Run doctests on all Python files in the main folder and its subfolders
run_doctests_in_folder(main_folder_path)

if False:
    from doctest import DocTestFinder, DocTestRunner

    finder = DocTestFinder()
    runner = DocTestRunner()

    class SafeRunner:
        def __enter__(self):
            # move mpqp to mpqp.bak
            pass

        def __exit__(self, exc_type, exc_value, exc_tb):
            # move mpqp.bak to mpqp
            pass

    for test in finder.find(mpqp, "mpqp", globs=test_globals):
        with SafeRunner():
            runner.run(test)
