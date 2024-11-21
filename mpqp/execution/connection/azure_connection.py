from getpass import getpass
from typing import TYPE_CHECKING

from termcolor import colored

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable

if TYPE_CHECKING:
    from azure.quantum import Workspace
    from azure.quantum.qiskit import AzureQuantumProvider


def config_azure_account():
    """Configure the Azure account by setting the resource ID and location.

    This function will prompt the user for their Azure resource ID and location.
    If the account is already configured, the user will be given the option to
    update the configuration. The function validates the connection to Azure
    before saving the credentials.

    Returns:
        A tuple containing a message indicating the result of the configuration
        and an empty list. If the configuration is successful, the message
        indicates the success, otherwise, it indicates a failure or
        cancellation.

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
    """Retrieve Azure resource ID and location information from the environment
    variables and format them in a way to be displayed to the user.

    Returns:
        The resource id and location in a displayable format.

    Example:
        >>> get_azure_account_info()
        AZURE_RESOURCE_ID: /subs*****
        AZURE_LOCATION: East US

    """
    azure_resource_id = get_env_variable("AZURE_RESOURCE_ID")
    if azure_resource_id == "":
        display = "Not configured"
    else:
        display = azure_resource_id[:5] + "*****"
    azure_location = get_env_variable("AZURE_LOCATION")

    return "   AZURE_RESOURCE_ID: " + display + "\n   AZURE_LOCATION: " + azure_location


def test_connection(resource_id: str, Location: str) -> bool:
    """Test the connection to Azure service.

    Args:
        resource_id: The Azure resource ID.
        location: The Azure resource location.

    Returns:
        ``True`` if the connection is successful.

    Example:
        >>> resource_id = "/subscriptions/ac1e2d6a-6adf-acad-b795-eaa8bfe45cbc/resourceGroups/MyGroup/providers/Microsoft.Quantum/Workspaces/myworkspace"
        >>> test_connection(resource_id, "East US")
        True

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
    """Retrieve the Azure Quantum Workspace instance.

    Returns:
        Workspace: An instance of the Azure Quantum Workspace configured with the
        resource ID and location from the environment variables.

    Example:
        >>> workspace = get_azure_workspace()
        <azure.quantum.workspace.Workspace object at 0x000000000000>

    """
    from azure.quantum import Workspace

    return Workspace(
        resource_id=get_env_variable(f"AZURE_RESOURCE_ID"),
        location=get_env_variable(f"AZURE_LOCATION"),
    )


def get_azure_provider() -> "AzureQuantumProvider":
    """Retrieve the Azure Quantum Provider.

    Returns:
        An instance of Azure Quantum Provider linked to the Azure Quantum
        Workspace.

    Example:
        >>> provider = get_azure_provider()
        <azure.quantum.qiskit.provider.AzureQuantumProvider object at 0x000000000000>

    """
    from azure.quantum.qiskit import AzureQuantumProvider

    return AzureQuantumProvider(get_azure_workspace())


def get_all_job_ids():
    """Retrieve the job IDs associated with the current Azure account/group.

    Returns:
        All job IDs of your tasks saved in Azure Quantum Workspace.

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
    """Retrieve a specific Azure Quantum job by its ID.

    Args:
        job_id: The ID of the job to retrieve.

    Returns:
        The Azure Quantum job object.

    Example:
        >>> job = get_jobs_by_id('6a46ae9a-d02f-4a23-b46f-eae43471bc22')
        <azure.quantum.job.job.Job object at 0x0000000000000>

    """
    workspace = get_azure_workspace()
    return workspace.get_job(job_id)
