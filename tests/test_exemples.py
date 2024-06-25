import os
import sys
import subprocess

NOTEBOOK_DIR = "examples/notebooks/"
PYTHON_FILES_DIR = "examples/scripts/"
SKIP_NOTEBOOK = os.path.join(NOTEBOOK_DIR, "2_Execution_Bell_circuit.ipynb")

def run_notebooks():
    result = subprocess.run(["pytest", "--nbmake", NOTEBOOK_DIR, "--ignore", SKIP_NOTEBOOK])
    if result.returncode != 0:
        raise Exception(f"Some notebooks failed")

def generate_tests_for_python_scripts():
    python_files = [file for file in os.listdir(PYTHON_FILES_DIR) if file.endswith('.py')]

    def make_test_func(py_file: str):
        def test_func():
            py_file_path = os.path.join(PYTHON_FILES_DIR, py_file)
            print(f"Running Python script: {py_file}")
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "UTF-8"
            env["MPLBACKEND"] = "Agg" # don't show 
            result = subprocess.run([sys.executable, py_file_path], env=env)
            assert result.returncode == 0, f"{py_file} failed with return code {result.returncode}"
        return test_func

    for py_file in python_files:
        test_name = f"test_{os.path.splitext(py_file)[0]}"
        globals()[test_name] = make_test_func(py_file)


if "--long" in sys.argv:
    generate_tests_for_python_scripts()
    test_notebooks = run_notebooks