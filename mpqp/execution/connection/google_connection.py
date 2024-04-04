from getpass import getpass
from termcolor import colored

import cirq_ionq as ionq
from cirq_ionq.ionq_exceptions import IonQException
from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable


def test_ionq_connection() -> bool:
    service = ionq.Service(default_target="simulator")
    try:
        service.list_jobs()
        return True
    except IonQException:
        print(colored("Wrong credentials", "red"))
        return False


def config_ionq_account():
    was_configured = get_env_variable("GOOGLE_IONQ_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "An IONQ account is already configured. Do you want to update it? [y/N]"
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    token = getpass("Enter your ionq token (hidden): ")
    if token == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    old_token = get_env_variable("IONQ_API_KEY")
    save_env_variable("IONQ_API_KEY", token)
    if test_ionq_connection():
        save_env_variable("GOOGLE_IONQ_CONFIGURED", "True")
        return "IONQ account correctly configured", []
    else:
        if was_configured:
            save_env_variable("IONQ_API_KEY", old_token)
        else:
            save_env_variable("GOOGLE_IONQ_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []


def get_google_account_info() -> str:
    ionq_api_key = get_env_variable("IONQ_API_KEY")

    return f"""   IONQ_api_key : '{ionq_api_key}'"""
