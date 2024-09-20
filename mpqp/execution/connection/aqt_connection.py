from typing import Optional
from termcolor import colored

from mpqp.execution.connection.env_manager import config_key, get_env_variable


def config_aqt_key():
    """
    # TODO add aqt in list of provider
    Configure the AQT account by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    return config_key("AQT_TOKEN", "AQT", test_aqt_connection)


def test_aqt_connection(key: Optional[str] = None) -> bool:
    """

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    from qiskit_aqt_provider import AQTProvider
    from qiskit_aqt_provider.aqt_provider import NoTokenWarning

    try:
        AQTProvider(access_token=key)
        return True
    except NoTokenWarning:
        print(colored("Wrong credentials", "red"))
        return False


def get_aqt_job_ids() -> list[str]:
    """
    # TODO
    Retrieves all job IDs associated with AQT jobs.

    Returns:
        A list of job IDs.
    """
    from qiskit_aqt_provider import AQTProvider, aqt_job
    from qiskit_aqt_provider.primitives import AQTSampler

    # provider = AQTProvider(access_token=get_env_variable("AQT_API_KEY"))

    # Get all jobs associated with the provider
    # Extract job IDs from the retrieved jobs

    return []
