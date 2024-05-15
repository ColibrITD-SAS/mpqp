import doctest
import importlib
import os

import numpy as np


from mpqp.all import *
from mpqp.execution.connection.env_manager import (
    get_existing_config_str,
    MPQP_CONFIG_PATH,
)
from mpqp.core.instruction.measurement import pauli_string
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.execution import BatchResult
from mpqp.qasm import open_qasm_2_to_3
from mpqp.tools.generics import clean_array, clean_matrix
from mpqp.qasm import replace_custom_gates, parse_custom_gates
from mpqp.execution.connection.env_manager import (
    get_env_variable,
    save_env_variable,
    load_env_variables,
)

from mpqp.tools.generics import find, flatten
from mpqp.tools.maths import is_hermitian, is_unitary, normalize, rand_orthogonal_matrix

test_globals = globals().copy()
test_globals.update(locals())

pass_file = ["connection", "noise_methods", "remote_handle"]
saf_file = ["env"]

from doctest import DocTestFinder, DocTestRunner
from dotenv import dotenv_values, unset_key, set_key

finder = DocTestFinder()
runner = DocTestRunner()


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

        # Write the content to the backup file
        open(MPQP_CONFIG_PATH + "_tmp", "w").write(env)
        open(MPQP_CONFIG_PATH, "w").close()

    def __exit__(self, exc_type, exc_value, exc_tb):
        backup_env = open(MPQP_CONFIG_PATH + "_tmp", "r").read()

        # Unset keys from the .env file
        val = dotenv_values(MPQP_CONFIG_PATH)
        for key in val.keys():
            set_key(MPQP_CONFIG_PATH, key, "")
        load_env_variables()

        # Reload configuration from backup file
        open(MPQP_CONFIG_PATH, "w").write(backup_env)
        load_env_variables()


folder_path = "mpqp"
for root, _, files in os.walk(folder_path):
    for filename in files:
        if any(str in filename for str in pass_file):
            continue
        elif filename.endswith(".py"):
            print(f"Running doctests in {os.path.join(os.getcwd(),root,filename)}")
            my_module = importlib.import_module(
                os.path.join(root, filename).replace(".py", "").replace("\\", ".")
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
                            runner.run(test)
                    else:
                        runner.run(test)
