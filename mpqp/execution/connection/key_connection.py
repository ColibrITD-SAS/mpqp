from getpass import getpass
from typing import Callable, Optional

from termcolor import colored

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable


def config_key(
    key_name: str, configuration_name: str, test_connection: Callable[[str], bool]
):
    """
    Configure a key by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    was_configured = get_env_variable(f"{configuration_name}_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            f"{configuration_name} key is already configured. Do you want to update it? [y/N]"
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    token = getpass(f"Enter your {configuration_name} token (hidden): ")
    if token == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    if test_connection(token):
        save_env_variable(f"{key_name}", token)
        save_env_variable(f"{configuration_name}_CONFIGURED", "True")
        return f"{configuration_name} key correctly configured", []
    else:
        if not was_configured:
            save_env_variable(f"{configuration_name}_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []


# TODO: move the providers specific keys to their own file ? and the
# `config_key` to the env_manager ?
def config_ionq_key():
    """
    Configure the IonQ account by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    configuration_name = "IONQ"
    key_name = "IONQ_API_KEY"
    return config_key(key_name, configuration_name, test_ionq_connection)


def test_ionq_connection(key: Optional[str] = None) -> bool:
    """
    Test the connection to the IonQ service.

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


def config_aqt_key():
    """
    # TODO add aqt in list of provider
    Configure the AQT account by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    configuration_name = "AQT"
    key_name = "AQT_TOKEN"
    return config_key(key_name, configuration_name, test_aqt_connection)


def test_aqt_connection(key: Optional[str] = None) -> bool:
    """
    # TODO install qiskit_aqt_provider by updating qiskit to 1.0.0
    Test the connection to the AQT service.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    raise NotImplementedError
    # from qiskit_aqt_provider import AQTProvider
    # from qiskit_aqt_provider.aqt_provider import NoTokenWarning

    # try:
    #     AQTProvider(access_token=key)
    #     return True
    # except NoTokenWarning:
    #     print(colored("Wrong credentials", "red"))
    #     return False


def get_ionq_job_ids() -> list[str]:
    """
    Retrieves ionq job IDs associated with IonQ jobs.

    Returns:
        A list of job IDs.
    """
    ionq_job_ids = []
    if get_env_variable("IONQ_API_KEY") == "True":
        import cirq_ionq as ionq

        service = ionq.Service()
        ionq_job_ids = [job.job_id() for job in service.list_jobs()]
    return ionq_job_ids


def get_aqt_job_ids() -> list[str]:
    """Retrieves all job IDs associated with AQT jobs.
    # TODO

    Returns:
        A list of job IDs.
    """
    raise NotImplementedError
    # from qiskit_aqt_provider import AQTProvider, aqt_job
    # from qiskit_aqt_provider.primitives import AQTSampler

    # provider = AQTProvider()
    # aqt_job_ids = []
    # if get_env_variable("AQT_API_KEY") == "True":
    #     return aqt_job_ids
    #     aqt_job_ids = [job.job_id() for job in service.list_jobs()]
    # return aqt_job_ids
