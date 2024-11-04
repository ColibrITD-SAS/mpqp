from typing import Optional

from termcolor import colored

from mpqp.execution.connection.env_manager import config_key, get_env_variable


def config_ionq_key():
    """Configure the IonQ account by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty
        list (used to conform to the protocol needed by the functions calling
        this one).
    """
    return config_key("IONQ_API_KEY", "IONQ", test_ionq_connection)


def test_ionq_connection(key: Optional[str] = None) -> bool:
    """Test the connection to the IonQ service.

    Args:
        key: The API key for authenticating with the IonQ service.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    import cirq_ionq as ionq
    from cirq_ionq.ionq_exceptions import IonQException

    service = ionq.Service(api_key=key, default_target="simulator")
    try:
        service.list_jobs()
        return True
    except IonQException:
        print(colored("Wrong credentials", "red"))
        return False


def get_ionq_account_info() -> str:
    """Get the IonQ API key from the environment variables.

    Returns:
        str: A string containing the IonQ API key.
    """
    ionq_api_key = get_env_variable("IONQ_API_KEY")
    if ionq_api_key == "":
        display = "Not configured"
    else:
        display = ionq_api_key[:5] + "*****"

    return "   IONQ_API_KEY: " + display


def get_ionq_job_ids() -> list[str]:
    """Retrieves ionq job IDs associated with IonQ jobs.

    Returns:
        A list of job IDs.
    """
    ionq_job_ids = []
    if get_env_variable("IONQ_API_KEY_CONFIGURED") == "True":
        import cirq_ionq as ionq

        service = ionq.Service()
        ionq_job_ids = [job.job_id() for job in service.list_jobs()]
    return ionq_job_ids
