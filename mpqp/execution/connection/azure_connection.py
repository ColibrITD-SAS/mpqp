from getpass import getpass
from typing import TYPE_CHECKING
from termcolor import colored

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable

if TYPE_CHECKING:
    from azure.quantum.qiskit import AzureQuantumProvider
    from azure.quantum import Workspace


def config_azure_account():
    """
    Configure Azure account with resource_id and  Location.

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

    resource_id = getpass(f"Enter your AZURE resource_id (hidden): ")
    if resource_id == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    Location = input(f"Enter your AZURE Location: ")
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


def get_azure_account_info() -> str:
    """
    Get Azure resource Id and Location.

    Returns:
        str: A string containing the azure resource Id and Location.
    """
    azure_resource_id = get_env_variable("AZURE_RESOURCE_ID")
    if azure_resource_id == "":
        display = "Not configured"
    else:
        display = azure_resource_id[:5] + "*****"
    azure_location = get_env_variable("AZURE_LOCATION")

    return "   AZURE_RESOURCE_ID: " + display + "\n   AZURE_LOCATION: " + azure_location


def test_connection(resource_id: str, Location: str) -> bool:
    """
    Test the connection to Azure service.

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


def get_azure_workspace() -> "Workspace":
    from azure.quantum import Workspace

    return Workspace(
        resource_id=get_env_variable(f"AZURE_RESOURCE_ID"),
        location=get_env_variable(f"AZURE_LOCATION"),
    )


def get_azure_provider() -> "AzureQuantumProvider":
    from azure.quantum.qiskit import AzureQuantumProvider

    return AzureQuantumProvider(get_azure_workspace())


def get_all_job_ids():
    """Retrieves all the task ids of this account/group from Azure

    Example:
        >>> get_all_jobs_ids()
        ['6a46ae9a-d02f-4a23-b46f-eae43471bc22',
         '11db7e68-2b17-4b00-a4ec-20f662fd4876',
         '292d329f-727c-4b92-83e1-7d4bedd4b243',
         '4b94c703-2ce8-480b-b3f3-ecb2580dbb82',
         'edc094aa-23e8-4a8c-87be-f2e09281d79d',
         'af9e623a-dd1c-4ecb-9db6-dbbd1af08110']

    """
    workspace = get_azure_workspace()
    return [job.id for job in workspace.list_jobs()]


def get_jobs_by_id(job_id: str):
    """Retrieves an Azure job by its id"""
    workspace = get_azure_workspace()
    return workspace.get_job(job_id)
