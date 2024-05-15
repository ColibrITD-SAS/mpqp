"""This module takes care of saving and loading the configuration of supported
providers."""

import os

from dotenv import load_dotenv, set_key
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
