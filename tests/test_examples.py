import os
import subprocess
import sys

NOTEBOOK_DIR = "examples/notebooks/"
PYTHON_FILES_DIR = "examples/scripts/"


def generate_tests_for_notebooks():
    # nbmake: https://pypi.org/project/nbmake/
    # Ignore a Code Cell: "metadata": { "tags": ["skip-execution"]}
    notebook_files = [
        file for file in os.listdir(NOTEBOOK_DIR) if file.endswith(".ipynb")
    ]

    def make_test_func(file: str):
        def test_func():
            env = os.environ.copy()
            venv_path = os.environ.get('VIRTUAL_ENV')
            if venv_path:
                env["PATH"] = f"{os.path.join(venv_path, 'bin')}"
                env["VIRTUAL_ENV"] = venv_path
            env["PYTHONIOENCODING"] = "UTF-8"
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--nbmake", file], env=env
            )
            if result.returncode != 0:
                raise Exception(f"failed with return code {result.returncode}")

        return test_func

    env = os.environ.copy()
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        env["PATH"] = f"{os.path.join(venv_path, 'bin')}"
        env["VIRTUAL_ENV"] = venv_path
    env["PYTHONIOENCODING"] = "UTF-8"

    for notebook in notebook_files:
        test_name = f"test_{os.path.splitext(notebook)[0]}"
        globals()[test_name] = make_test_func(os.path.join(NOTEBOOK_DIR, notebook))


def generate_tests_for_python_scripts():
    python_files = [
        file for file in os.listdir(PYTHON_FILES_DIR) if file.endswith(".py")
    ]

    def make_test_func(py_file: str):
        def test_func():
            py_file_path = os.path.join(PYTHON_FILES_DIR, py_file)
            result = subprocess.run([sys.executable, py_file_path], env=env)
            assert (
                result.returncode == 0
            ), f"failed with return code {result.returncode}"

        return test_func

    env = os.environ.copy()
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        env["PATH"] = f"{os.path.join(venv_path, 'bin')}"
        env["VIRTUAL_ENV"] = venv_path
    env["PYTHONIOENCODING"] = "UTF-8"
    env["MPLBACKEND"] = "Agg"  # don't show

    for py_file in python_files:
        test_name = f"test_{os.path.splitext(py_file)[0]}"
        globals()[test_name] = make_test_func(py_file)


if "--long-local" in sys.argv or "--long" in sys.argv:
    generate_tests_for_python_scripts()
    generate_tests_for_notebooks()
