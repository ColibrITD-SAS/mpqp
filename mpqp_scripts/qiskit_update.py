import subprocess
import sys

qiskit_packages = [
    "qiskit",
    "qiskit-aer",
    "qiskit-algorithms",
    "qiskit_alice_bob_provider",
    "qiskit-aqua",
    "qiskit-finance",
    "qiskit-ibm-provider",
    "qiskit-ibm-runtime",
    "qiskit-ibmq-provider",
    "qiskit-ignis",
    "qiskit-optimization",
    "qiskit-qir-alice-bob-fork",
    "qiskit-terra",
    "ibm-cloud-sdk-core",
    "ibm-platform-services",
    "IBMQuantumExperience",
]


def run_command(command: str):
    """Run a shell command and handle errors."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")


def update_packages():
    """Uninstall all specified Qiskit packages."""
    print(f"Uninstalling qiskit package...")
    for pkg in qiskit_packages:
        run_command(f"{sys.executable} -m pip uninstall -y {pkg}")
    print("Installing mpqp...")
    run_command(f"{sys.executable} -m pip install --upgrade mpqp")


if __name__ == "__main__":
    print(f"Using Python executable: {sys.executable}")
    update_packages()
