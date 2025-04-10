"""This module takes care of saving and loading the configuration of supported
providers in our configuration file, located at ``~/.mpqp/.env``."""

import os
from getpass import getpass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv, set_key
from termcolor import colored
from typeguard import typechecked

MPQP_ENV = Path("~/.mpqp/.env").expanduser()


def _create_config_if_needed():
    """If configuration file does not exist we create it."""
    # convert from old format to new one
    if MPQP_ENV.parent.is_file():
        os.rename(MPQP_ENV.parent, MPQP_ENV.parent.with_suffix(".tmp"))
        MPQP_ENV.parent.mkdir(parents=True, exist_ok=True)
        os.rename(MPQP_ENV.parent.with_suffix(".tmp"), MPQP_ENV)
        return

    if not MPQP_ENV.exists():
        MPQP_ENV.parent.mkdir(parents=True, exist_ok=True)
        MPQP_ENV.touch()


def get_existing_config_str() -> str:
    """Gets the content of the configuration file.

    Returns:
        The content of the configuration file.

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
    if not MPQP_ENV.exists():
        return ""
    with MPQP_ENV.open("r") as env:
        file_str = env.read()
    return file_str


def load_env_variables() -> bool:
    """Loads the variables stored in the configuration file.

    Returns:
        ``True`` if the variables are loaded correctly.

    Example:
        >>> os.getenv("IBM_CONFIGURED")  # doctest: +SKIP
        >>> open(os.path.expanduser("~") + "/.mpqp/.env", "w").write("IBM_CONFIGURED='True'\\n")
        22
        >>> os.getenv("IBM_CONFIGURED")  # doctest: +SKIP
        >>> load_env_variables()
        True
        >>> os.getenv("IBM_CONFIGURED")
        'True'

    """
    return load_dotenv(MPQP_ENV, override=True)


@typechecked
def get_env_variable(key: str) -> str:
    """Loads the configuration file and returns the value associated with the key
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

    return os.getenv(key, "")


@typechecked
def save_env_variable(key: str, value: str) -> bool:
    """Adds or updates the ``key`` environment variable in the configuration file.

    Args:
        key: Name of the environment variable.
        value: Value of the environment variable.

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
        a, _, _ = set_key(MPQP_ENV, key, value)
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
