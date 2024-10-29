# pyright: reportUnusedImport=false
import importlib
import os
import sys
import warnings
from doctest import SKIP, DocTest, DocTestFinder, DocTestRunner
from functools import partial
from types import TracebackType
from typing import Any, Optional, Type

import pytest
from dotenv import dotenv_values, set_key, unset_key

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
from mpqp.tools.circuit import random_circuit
from mpqp.tools.display import clean_1D_array, clean_matrix, pprint
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning
from mpqp.tools.generics import find, find_index, flatten
from mpqp.tools.maths import *


class SafeRunner:
    def __enter__(self):
        # Ensure the config file exists
        if not os.path.exists(MPQP_CONFIG_PATH):
            open(MPQP_CONFIG_PATH, "a").close()
        env = get_existing_config_str()

        # Unset keys from the .env file
        val = dotenv_values(MPQP_CONFIG_PATH)
        for key in val.keys():
            set_key(MPQP_CONFIG_PATH, key, "")
            load_env_variables()
            if os.getenv(key) is not None:
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
            if os.getenv(key) is not None:
                del os.environ[key]

        # Reload configuration from backup file
        open(MPQP_CONFIG_PATH, "w").write(backup_env)
        load_env_variables()


class RandomDoctestRunner(DocTestRunner):
    def run(self, test: DocTest, *args: Any, **kwargs: Any):
        test.filename
        if "rand" in test.name and "seed=" not in test.examples[0].source:
            for example in test.examples:
                example.options[SKIP] = True
        return super().run(test, *args, **kwargs)


test_globals = globals().copy()
test_globals.update(locals())

to_pass = ["connection", "noise_methods", "remote_handle"]
unsafe_files = ["env"]

finder = DocTestFinder()
runner = DocTestRunner()


def run_doctest(root: str, filename: str):
    warnings.filterwarnings("ignore", category=UnsupportedBraketFeaturesWarning)
    assert True
    my_module = importlib.import_module(
        os.path.join(root, filename)
        .replace(".py", "")
        .replace("\\", ".")
        .replace("/", ".")
    )
    safe_needed = any(str in filename for str in unsafe_files)
    for test in finder.find(my_module, "mpqp", globs=test_globals):
        if (
            test.docstring
            and "3M-TODO" not in test.docstring
            and "6M-TODO" not in test.docstring
        ):
            if safe_needed:
                with SafeRunner():
                    if "PYTEST_CURRENT_TEST" not in os.environ:
                        assert runner.run(test).failed == 0
            else:
                assert runner.run(test).failed == 0


folder_path = "mpqp"
for root, _, files in os.walk(folder_path):
    for filename in files:
        if all(str not in filename for str in to_pass) and filename.endswith(".py"):
            t_function_name = "test_" + "mpqp".join(
                (root + filename).split("mpqp")
            ).replace("\\", "_").replace("/", "_").replace(".", "_")
            print(root + "\\" + filename)
            locals()[t_function_name] = partial(
                run_doctest, root=root, filename=filename
            )
