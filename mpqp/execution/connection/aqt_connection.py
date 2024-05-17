from getpass import getpass
from termcolor import colored

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable


def test_aqt_connection() -> bool:
    """
    Test the connection to the aqt service.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    return False


def config_aqt_key():
    """
    Configure the AQT account by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    was_configured = get_env_variable("AQT_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "An AQT key is already configured. Do you want to update it? [y/N]"
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    token = getpass("Enter your aqt token (hidden): ")
    if token == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    old_token = get_env_variable("AQT_API_KEY")
    save_env_variable("AQT_API_KEY", token)
    if test_aqt_connection():
        save_env_variable("AQT_CONFIGURED", "True")
        return "AQT key correctly configured", []
    else:
        if was_configured:
            save_env_variable("AQT_API_KEY", old_token)
        else:
            save_env_variable("AQT_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []


def get_aqt_account_info() -> str:
    """
    Get the AQT API key from the environment variables.

    Returns:
        str: A string containing the AQT API key.
    """
    aqt_api_key = get_env_variable("AQT_API_KEY")

    return f"""   AQT_API_KEY : '{aqt_api_key}'"""


def get_all_job_ids() -> list[str]:
    """
    Retrieves all job IDs associated with AQT jobs.

    Returns:
        A list of job IDs.
    """
    from qiskit_aqt_provider import AQTProvider, aqt_job
    from qiskit_aqt_provider.primitives import AQTSampler
    
    AQTProvider.
    AQTSampler.
    aqt_job_ids = []
    if get_env_variable("AQT_API_KEY") == "True":
        service = aqt.Service()
        aqt_job_ids = [job.job_id() for job in service.list_jobs()]
    return aqt_job_ids
