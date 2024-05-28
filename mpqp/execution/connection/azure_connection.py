from getpass import getpass
from typing import Callable, Optional
from termcolor import colored

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable


def config_azure_account():
    """
    Configure azure account with resource_id and  Location.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    was_configured = get_env_variable(f"AZURE_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            f"AZURE account is already configured. Do you want to update it? [y/N]"
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    resource_id = getpass(f"Enter your AZURE resource_id: ")
    if resource_id == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    Location = getpass(f"Enter your AZURE Location: ")
    if Location == "":
        print(colored("Empty Location", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    if test_connection(resource_id, Location):
        save_env_variable(f"AZURE_RESOURCE_ID", resource_id)
        save_env_variable(f"AZURE_LOCATION", Location)
        save_env_variable(f"AZURe_CONFIGURED", "True")
        return f"AZURE account correctly configured", []
    else:
        if not was_configured:
            save_env_variable(f"AZURE_RESOURCE_ID", resource_id)
            save_env_variable(f"AZURE_LOCATION", Location)
            save_env_variable(f"AZURE_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []


def test_connection(resource_id: str, Location: str) -> bool:
    """
    Test the connection to the IonQ service.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    from azure.quantum import Workspace
    from azure.quantum.qiskit import AzureQuantumProvider

    try:
        workspace = Workspace(
            resource_id=resource_id,
            location=Location,
        )
        AzureQuantumProvider(workspace)
        return True
    except:
        print(colored("Wrong credentials", "red"))
        return False
