from __future__ import annotations

import os
from getpass import getpass
from typing import Any

from termcolor import colored
from typeguard import typechecked

from mpqp.execution.connection.env_manager import (
    get_env_variable,
    load_env_variables,
    save_env_variable,
)
from mpqp.tools.errors import QLMRemoteExecutionError

QLM_connection = None


@typechecked
def config_qlm_account(username: str, password: str, global_config: bool) -> bool:
    """Configures and saves locally QLM account's information.

    Args:
        username: QLM username.
        password: QLM password.
        global_config: If True, this QLM account will be configured to work even
            outside MPQP.

    Raises:
        QLMRemoteExecutionError: If the account could not be saved.
    """
    # store the username and password in environment variables QLM_USER and QLM_PASSWD in .mpqp
    prev_user = get_env_variable("QLM_USER")
    prev_pass = get_env_variable("QLM_PASSWD")
    prev_configure = get_env_variable("QLM_CONFIGURED")
    if prev_configure == "":
        prev_configure = "False"
    file_content = None
    netrc_path = os.path.expanduser("~") + "/.netrc"
    if global_config:
        try:
            with open(netrc_path, 'r') as file:
                file_content = file.read()
        except:
            pass

    save_env_variable("QLM_USER", username)
    if not global_config:
        save_env_variable("QLM_PASSWD", password)

    try:
        if global_config:
            print("we are in the global part")
            # if file doesn't exist, create it, or overwrite the credentials in the ~/.netrc file
            with open(netrc_path, "w") as file:
                file.write(
                    f"""\
machine qlm35e.neasqc.eu
login {username}
password {password}"""
                )
            # Set the permissions to read and right for user only
            os.chmod(netrc_path, 0o600)
        else:
            if os.path.exists(netrc_path):
                rename_decision = input(
                    f"'~/.netrc' already exists and will override the configuration. Do you want to rename it in '~/.netrc_back'? [Y/n]"
                )
                if rename_decision.lower().strip() in ('', 'y', 'yes'):
                    os.rename(netrc_path, netrc_path + "_back")

        from qat.qlmaas import (
            QLMaaSConnection,  # pyright: ignore[reportAttributeAccessIssue]
        )

        c = QLMaaSConnection(
            hostname="qlm35e.neasqc.eu",
            port=443,
            authentication="password",
            timeout=2500,
        )
        c.create_config()

    except Exception as err:
        save_env_variable("QLM_USER", prev_user)
        save_env_variable("QLM_PASSWD", prev_pass)
        save_env_variable("QLM_CONFIGURED", prev_configure)
        if global_config:
            if file_content is None:
                try:
                    os.remove(netrc_path)
                except FileNotFoundError:
                    print(f"{netrc_path} does not exist.")
            else:
                with open(netrc_path, "w") as file:
                    file.write(file_content)
        if "Invalid credential" in str(err):
            return False
        raise QLMRemoteExecutionError(
            "Error when saving the account.\nTrace: " + str(err)
        )
    return True


def setup_qlm_account() -> tuple[str, list[Any]]:
    """Setups the QLM account, by looking at the existing configuration, asking
    for username/password and updating the current account."""
    already_configured = get_env_variable("QLM_CONFIGURED") == "True"

    if already_configured:
        decision = input(
            "An QLM account is already configured. Do you want to update it? [y/N] "
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    username = input("Enter your QLM username: ")
    password = getpass("Enter your QLM password (hidden): ")
    global_decision = input(
        "Do you want to configure this account only for MPQP? [y/N] "
    )
    global_config = True if global_decision.lower().strip() != "y" else False

    connection_success = config_qlm_account(username, password, global_config)
    if connection_success:
        save_env_variable("QLM_CONFIGURED", "True")

    if connection_success:
        return "QLM account correctly configured", []
    else:
        print(colored("Wrong credential", "red"))
        getpass("Press 'Enter' to continue")
        return "", []


def get_all_job_ids() -> list[str]:
    """Retrieves from the remote QLM all the job-ids associated with this account.

    Returns:
        List of all job-ids associated with this account.

    Example:
        >>> get_all_job_ids()
        ['Job144361', 'Job144360', 'Job144359', 'Job144358', 'Job144357', 'Job143334', 'Job143333', 'Job143332',
        'Job141862', 'Job141861', 'Job141722', 'Job141720', 'Job141716', 'Job141715', 'Job141712', 'Job19341']

    """

    if get_env_variable("QLM_CONFIGURED") == "True":
        connection = get_QLMaaSConnection()
        return [job_info.id for job_info in connection.get_jobs_info()]
    return []


def get_QLMaaSConnection():
    """Connects to the QLM and returns the QLMaaSConnection. If the connection
    was already established, we only return the one stored in global variable,
    otherwise we instantiate a new QLMaaSConnection."""

    global QLM_connection
    if QLM_connection is None:
        loaded = load_env_variables()
        if not loaded:
            raise IOError("Could not load environment variables from ~/.mpqp")

        if get_env_variable("QLM_CONFIGURED") == "False":
            raise QLMRemoteExecutionError(
                "Error when instantiating QLMaaSConnection. No QLM account configured."
            )

        from qat.qlmaas.connection import QLMaaSConnection

        QLM_connection = QLMaaSConnection(
            hostname="qlm35e.neasqc.eu",
            port=443,
            authentication="password",
            timeout=2500,
        )
    return QLM_connection
