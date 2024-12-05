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
from anytree import Node
from dotenv import dotenv_values, set_key, unset_key
from numpy.random import default_rng

from mpqp.all import *
from mpqp.core.instruction.measurement import pauli_string
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.db import *
from mpqp.execution import BatchResult
from mpqp.execution.connection.env_manager import (
    _create_config_if_needed,  # pyright: ignore[reportPrivateUsage]
)
from mpqp.execution.connection.env_manager import (
    MPQP_ENV,
    get_env_variable,
    get_existing_config_str,
    load_env_variables,
    save_env_variable,
)
from mpqp.execution.providers.aws import estimate_cost_single_job
from mpqp.execution.runner import generate_job
from mpqp.noise.noise_model import _plural_marker  # pyright: ignore[reportPrivateUsage]
from mpqp.qasm import (
    qasm2_to_cirq_Circuit,
    qasm2_to_myqlm_Circuit,
    qasm2_to_Qiskit_Circuit,
    qasm3_to_braket_Program,
)
from mpqp.qasm.mpqp_to_qasm import mpqp_to_qasm2
from mpqp.qasm.open_qasm_2_and_3 import (
    convert_instruction_3_to_2,
    open_qasm_2_to_3,
    open_qasm_3_to_2,
    open_qasm_file_conversion_3_to_2,
    parse_user_gates,
    remove_include_and_comment,
    remove_user_gates,
)
from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit
from mpqp.qasm.qasm_to_mpqp import qasm2_parse
from mpqp.tools.circuit import random_circuit, random_gate, random_noise
from mpqp.tools.display import *
from mpqp.tools.display import clean_1D_array, clean_matrix, format_element, pprint
from mpqp.tools.errors import (
    OpenQASMTranslationWarning,
    UnsupportedBraketFeaturesWarning,
)
from mpqp.tools.generics import find, find_index, flatten
from mpqp.tools.maths import *
from mpqp.tools.maths import (
    is_hermitian,
    is_power_of_two,
    is_unitary,
    normalize,
    rand_orthogonal_matrix,
)

sys.path.insert(0, os.path.abspath("."))


class SafeRunner:
    def __enter__(self):
        _create_config_if_needed()
        env = get_existing_config_str()

        # Unset keys from the .env file
        val = dotenv_values(MPQP_ENV)
        for key in val.keys():
            set_key(MPQP_ENV, key, "")
            load_env_variables()
            if os.getenv(key) is not None:
                del os.environ[key]

        # Write the content to the backup file
        MPQP_ENV.with_suffix(".env_tmp").open("w").write(env)
        MPQP_ENV.open("w").close()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        backup_env = MPQP_ENV.with_suffix(".env_tmp").open("r").read()

        # Unset keys from the .env file
        val = dotenv_values(MPQP_ENV)
        for key in val.keys():
            set_key(MPQP_ENV, key, "")
            load_env_variables()
            if os.getenv(key) is not None:
                del os.environ[key]

        # Reload configuration from backup file
        open(MPQP_ENV, "w").write(backup_env)
        load_env_variables()


class DBRunner:
    def __enter__(self):
        import shutil

        db_original = os.path.join(os.getcwd(), "tests/test_database.db")
        db_temp = os.path.join(os.getcwd(), "tests/test_database_tmp.db")

        shutil.copyfile(db_original, db_temp)
        setup_db("tests/test_database_tmp.db")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        os.remove(os.path.join(os.getcwd(), "tests/test_database_tmp.db"))


test_globals = globals().copy()
test_globals.update(locals())

to_pass = ["connection", "noise_methods", "remote_handle"]
unsafe_files = ["env", "db"]

finder = DocTestFinder()
runner = DocTestRunner()


def stable_random(*args: Any, **kwargs: Any):
    user_seed = args[0] if len(args) != 0 else None
    return default_rng(user_seed or 351)


def run_doctest(root: str, filename: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('numpy.random.default_rng', stable_random)
    warnings.filterwarnings("ignore", category=UnsupportedBraketFeaturesWarning)
    warnings.filterwarnings("ignore", category=OpenQASMTranslationWarning)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*Noise is not applied to any gate, as there is no eligible gate in the circuit.*",
    )
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
                    if "db" in filename:
                        with DBRunner():
                            assert runner.run(test).failed == 0
                    else:
                        assert runner.run(test).failed == 0
            else:
                assert runner.run(test).failed == 0


folder_path = "mpqp"
for root, _, files in os.walk(folder_path):
    for filename in files:
        if all(str not in filename for str in to_pass) and filename.endswith(".py"):
            t_function_name = "test_doc_" + "mpqp".join(
                (root + "_" + filename).split("mpqp")
            ).replace("\\", "_").replace("/", "_").replace(".py", "")
            print(root + "\\" + filename)
            locals()[t_function_name] = partial(
                run_doctest, root=root, filename=filename
            )
