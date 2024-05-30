# pyright: reportUnusedImport=false
import importlib
import os
import sys
from doctest import DocTestFinder, DocTestRunner, Example
from types import TracebackType
from typing import Optional, Type

import numpy as np
import pytest
from dotenv import dotenv_values, set_key, unset_key

from mpqp.tools.errors import UnsupportedBraketFeaturesWarning

sys.path.insert(0, os.path.abspath("."))

from mpqp.all import *
from mpqp.core.instruction.measurement import pauli_string
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.execution import BatchResult
from mpqp.execution.connection.env_manager import (
    MPQP_CONFIG_PATH,
    get_env_variable,
    get_existing_config_str,
    load_env_variables,
    save_env_variable,
)
from mpqp.qasm import open_qasm_2_to_3, remove_user_gates
from mpqp.qasm.open_qasm_2_and_3 import parse_user_gates
from mpqp.tools.generics import clean_array, clean_matrix, find, flatten
from mpqp.tools.maths import is_hermitian, is_unitary, normalize, rand_orthogonal_matrix


class SafeRunner:
    def __enter__(self):
        if not os.path.exists(MPQP_CONFIG_PATH):  # Ensure the config file exists
            open(MPQP_CONFIG_PATH, "a").close()
        env = get_existing_config_str()

        # Unset keys from the .env file
        val = dotenv_values(MPQP_CONFIG_PATH)
        for key in val.keys():
            set_key(MPQP_CONFIG_PATH, key, "")
            load_env_variables()
            if os.getenv(key) != None:
                del os.environ[key]

        # Write the content to the backup file
        open(MPQP_CONFIG_PATH + "_tmp", "w").write(env)
        open(MPQP_CONFIG_PATH, "w").close()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        backup_env = open(MPQP_CONFIG_PATH + "_tmp", "r").read()

        # Unset keys from the .env file
        val = dotenv_values(MPQP_CONFIG_PATH)
        for key in val.keys():
            set_key(MPQP_CONFIG_PATH, key, "")
            load_env_variables()
            if os.getenv(key) != None:
                del os.environ[key]

        # Reload configuration from backup file
        open(MPQP_CONFIG_PATH, "w").write(backup_env)
        load_env_variables()


def test_documentation():
    print(os.getcwd())

    test_globals = globals().copy()
    test_globals.update(locals())

    pass_file = ["connection", "noise_methods", "remote_handle"]
    saf_file = ["env"]

    finder = DocTestFinder()
    runner = DocTestRunner()

    with pytest.warns(UnsupportedBraketFeaturesWarning):
        folder_path = "mpqp"
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if any(str in filename for str in pass_file):
                    continue
                elif filename.endswith(".py"):
                    print(
                        f"Running doctests in {os.path.join(os.getcwd(),root,filename)}"
                    )
                    my_module = importlib.import_module(
                        os.path.join(root, filename)
                        .replace(".py", "")
                        .replace("\\", ".")
                        .replace("/", ".")
                    )
                    saf = any(str in filename for str in saf_file)
                    for test in finder.find(my_module, "mpqp", globs=test_globals):
                        if (
                            test.docstring
                            and "3M-TODO" not in test.docstring
                            and "6M-TODO" not in test.docstring
                        ):
                            if saf:
                                with SafeRunner():
                                    if "PYTEST_CURRENT_TEST" not in os.environ:
                                        assert runner.run(test).failed == 0
                            else:
                                assert runner.run(test).failed == 0
