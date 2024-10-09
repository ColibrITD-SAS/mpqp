import os
import subprocess
import sys

NOTEBOOK_DIR = "examples/notebooks/"
PYTHON_FILES_DIR = "examples/scripts/"


def generate_tests_for_notebooks():
    """Generates Pytest-based test functions for each Jupyter notebook in a
    specified directory.

    Each test will run the notebook through `nbmake`
    (https://pypi.org/project/nbmake), which ensures that the notebooks are
    executable and error-free.

    Skip execution of a specific notebook cell by adding the following tag to
    its metadata (open it as a text file and and the following in a cell block):
    ```json
    "metadata": { "tags": ["skip-execution"] }
    ```

    """
    notebook_files = [
        file for file in os.listdir(NOTEBOOK_DIR) if file.endswith(".ipynb")
    ]

    def make_test_func(file: str):
        def test_func():
            notebook_path = os.path.join(NOTEBOOK_DIR, file)
            env["PYTHONPATH"] = f"{project_root};{os.path.join(project_root, 'tests')}"
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--nbmake", notebook_path], env=env
            )
            if result.returncode != 0:
                raise Exception(f"failed with return code {result.returncode}")

        return test_func

    for notebook in notebook_files:
        test_name = f"test_{os.path.splitext(notebook)[0]}"
        globals()[test_name] = make_test_func(notebook)


def generate_tests_for_python_scripts():
    python_files = [
        file for file in os.listdir(PYTHON_FILES_DIR) if file.endswith(".py")
    ]

    def make_test_func(py_file: str):
        def test_func():
            py_file_path = os.path.join(PYTHON_FILES_DIR, py_file)
            command = [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, r'{project_root}'); exec(open(r'{py_file_path}').read())",
            ]
            result = subprocess.run(command, env=env)
            assert (
                result.returncode == 0
            ), f"failed with return code {result.returncode}"

        return test_func

    for py_file in python_files:
        test_name = f"test_{os.path.splitext(py_file)[0]}"
        globals()[test_name] = make_test_func(py_file)


if "--long-local" in sys.argv or "--long" in sys.argv:
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        env["PATH"] = f"{os.path.join(venv_path, 'bin')}"
        env["VIRTUAL_ENV"] = venv_path
    env["PYTHONIOENCODING"] = "UTF-8"
    env["MPLBACKEND"] = "Agg"

    generate_tests_for_python_scripts()
    generate_tests_for_notebooks()
