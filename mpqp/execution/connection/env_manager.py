"""This module takes care of saving and loading the configuration of supported
providers."""

import os
from getpass import getpass
from typing import Callable

from dotenv import load_dotenv, set_key
from termcolor import colored
from typeguard import typechecked

MPQP_CONFIG_PATH = os.path.expanduser("~") + "/.mpqp"


def _create_config_if_needed():
    """If there is not already a ``.mpqp`` file we create it."""
    if not os.path.exists(MPQP_CONFIG_PATH):
        open(MPQP_CONFIG_PATH, "a").close()


def get_existing_config_str() -> str:
    """Gets the content of the ``.mpqp`` config file.

    Returns:
        The string with .mpqp file content.

    Example:
        >>> save_env_variable('QLM_USER', 'hjaffali')
        True
        >>> save_env_variable('QLM_PASSWD', '****************')
        True
        >>> save_env_variable('QLM_CONFIGURED', 'True')
        True
        >>> save_env_variable('BRAKET_CONFIGURED', 'True')
        True
        >>> print(get_existing_config_str()) # doctest: +NORMALIZE_WHITESPACE
        QLM_USER='hjaffali'
        QLM_PASSWD='****************'
        QLM_CONFIGURED='True'
        BRAKET_CONFIGURED='True'

    """
    if not os.path.exists(MPQP_CONFIG_PATH):
        return ""
    with open(MPQP_CONFIG_PATH, "r") as mpqp:
        file_str = mpqp.read()
    return file_str


def load_env_variables() -> bool:
    """Loads the variables stored in the ``.mpqp`` file.

    Returns:
        ``True`` if the variables are loaded correctly.

    Example:
        >>> os.getenv("IBM_CONFIGURED")
        >>> open(os.path.expanduser("~") + "/.mpqp", "w").write("IBM_CONFIGURED='True'\\n")
        22
        >>> os.getenv("IBM_CONFIGURED")
        >>> load_env_variables()
        True
        >>> os.getenv("IBM_CONFIGURED")
        'True'

    """
    load_dotenv(MPQP_CONFIG_PATH, override=True)
    return True


@typechecked
def get_env_variable(key: str) -> str:
    """Loads the ``.mpqp`` env file and returns the value associated with the key
    in parameter. If the variable does not exist, an empty string is returned.

    Args:
        key: The key for which we want to get the value.

    Example:
        >>> save_env_variable("BRAKET_CONFIGURED", 'True')
        True
        >>> get_env_variable("BRAKET_CONFIGURED")
        'True'
        >>> get_env_variable("RaNdOM")
        ''

    """
    _create_config_if_needed()
    load_env_variables()
    val = os.getenv(key, "")

    return val


@typechecked
def save_env_variable(key: str, value: str) -> bool:
    """Adds or updates the ``key`` environment variable in ``.mpqp`` file.

    Args:
        key: Name of the environment variable.
        value: Value to be saved.

    Returns:
        ``True`` if the save was successful.

    Examples:
        >>> get_env_variable("RaNdOM")
        ''
        >>> save_env_variable("RaNdOM", "azertyuiop")
        True
        >>> get_env_variable("RaNdOM")
        'azertyuiop'

    """
    _create_config_if_needed()

    try:
        a, _, _ = set_key(MPQP_CONFIG_PATH, key, value)
        if a is None:
            raise SystemError(
                "Something went wrong when trying to modify the MPQP "
                "connections configuration."
            )
        load_env_variables()
    except ValueError as e:
        print(e)
        return False

    return a


def config_key(
    key_name: str, configuration_name: str, test_connection: Callable[[str], bool]
):
    """Configure a key by setting the API token.

    Args:
        key_name: The name of the key to be saved in the environment variables.
        configuration_name: The name of the service for which the API token is
            being configured.
        test_connection: A callable function taking as input token and returning
            a boolean indicating whether the connection setup was successful.

    Returns:
        tuple: A message indicating the result of the configuration and an empty
        list (used to conform to the protocol needed by the functions calling
        this one).
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
            save_env_variable(f"{key_name}", token)
            save_env_variable(f"{configuration_name}_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []
