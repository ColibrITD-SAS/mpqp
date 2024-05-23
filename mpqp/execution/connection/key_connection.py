from getpass import getpass
from typing import Callable
from termcolor import colored

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable

def config_key(key_name :str, configuration_name: str, test_connection:  Callable[[], bool]):
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
    old_token = get_env_variable(f"{key_name}")
    save_env_variable(f"{key_name}", token)
    if test_connection():
        save_env_variable(f"{configuration_name}_CONFIGURED", "True")
        return f"{configuration_name} key correctly configured", []
    else:
        if was_configured:
            save_env_variable(f"{key_name}", old_token)
        else:
            save_env_variable(f"{configuration_name}_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []
    

def config_ionq_key():
    """
    Configure the IonQ account by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    configuration_name = 'IONQ'
    key_name = 'IONQ_API_KEY'
    return config_key(key_name, configuration_name, test_ionq_connection)


def test_ionq_connection() -> bool:
    """
    Test the connection to the IonQ service.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    from cirq_ionq.ionq_exceptions import IonQException
    import cirq_ionq as ionq

    service = ionq.Service(default_target="simulator")
    try:
        service.list_jobs()
        return True
    except IonQException:
        print(colored("Wrong credentials", "red"))
        return False


def config_aqt_key():
    """
    Configure the AQT account by setting the API token.

    Returns:
        tuple: A message indicating the result of the configuration and an empty list.
    """
    configuration_name = 'AQT'
    key_name = 'AQT_TOKEN'
    return config_key(key_name, configuration_name, test_ionq_connection)

