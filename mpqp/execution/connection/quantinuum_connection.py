from termcolor import colored

from mpqp.environment.env_manager import get_env_variable, save_env_variable


def setup_quantinuum_account():
    """set up the Quantinuum Nexus project used by MPQP."""

    was_configured = get_env_variable("QUANTINUUM_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "A Quantinuum Nexus account is already configured. Do you want to update it? [y/N] ",
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    print("\nChoose Quantinuum Nexus authentication method:")
    print("  1. Browser login")
    print("  2. Credentials prompt")
    method = input("Select option [1/2]: ").strip()

    try:
        import qnexus as qnx

        if method == "2":
            qnx.login_with_credentials()
        else:
            qnx.login()

        project_name = input("Enter your Quantinuum Nexus project name: ").strip()

        if project_name == "":
            print(colored("Empty project name", "red"))
            save_env_variable("QUANTINUUM_CONFIGURED", "False")
            return "", []

        project = qnx.projects.get_or_create(name=project_name)
        qnx.context.set_active_project(project)

        save_env_variable("QUANTINUUM_PROJECT_NAME", project_name)
        save_env_variable("QUANTINUUM_CONFIGURED", "True")

        return "Quantinuum account correctly configured.", []

    except Exception as err:
        save_env_variable("QUANTINUUM_CONFIGURED", "False")
        print(colored("Quantinuum Nexus authentication failed.", "red"))
        print(colored(str(err), "red"))
        input("Press 'Enter' to continue")
        return "", []


def get_quantinuum_account_info() -> str:
    """ "Return the locally configured Quantinuum Nexus project name."""

    project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")

    if project_name == "":
        return "Account not configured"

    return "   QUANTINUUM_PROJECT_NAME: " + project_name


def delete_quantinuum_account():
    """Delete the locally stored Quantinuum Nexus configuration."""

    decision = input(
        colored(
            "This will delete the local Quantinuum configuration and log out from Nexus. Continue? [y/N] ",
            "yellow",
        )
    )

    if decision.lower().strip() != "y":
        return "Canceled.", []

    try:
        import qnexus as qnx

        qnx.logout()
    except Exception:
        pass

    save_env_variable("QUANTINUUM_CONFIGURED", "False")
    save_env_variable("QUANTINUUM_PROJECT_NAME", "")

    print(colored("Quantinuum Nexus account deleted.", "green"))
    input("Press 'Enter' to continue")

    return "Quantinuum account deleted.", []


def get_quantinuum_config(device_name: str):
    """Return a Quantiunuum Nexus backend configuration."""

    import qnexus as qnx

    project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")

    qnx.login()

    if project_name != "":
        project = qnx.projects.get_or_create(project_name)
        qnx.context.set_active_project(project)

    return qnx.QuantinuumConfig(device_name=device_name)
