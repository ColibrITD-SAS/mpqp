from termcolor import colored

from mpqp.environment.env_manager import get_env_variable, save_env_variable
from mpqp.execution.devices import QUANTINUUMDevice


def setup_quantinuum_account():
    """Set up the Quantinuum Nexus project used by MPQP."""
    was_configured = get_env_variable("QUANTINUUM_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "A Quantinuum Nexus account is already configured. Do you want to update it? [y/N] ",
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    try:
        import qnexus as qnx

        qnx.login(force=was_configured)
        return _setup_quantinuum_project()

    except Exception as err:
        print(colored("Quantinuum Nexus configuration failed.", "red"))
        print(colored(str(err), "red"))
        input("Press 'Enter' to continue")
        return "", []


def _setup_quantinuum_project():
    """Choose an existing Quantinuum Nexus project or create a new one."""
    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    project_tree = QuestionNode(
        "Choose Quantinuum Nexus project option:",
        [
            AnswerNode(
                "Use existing project",
                lambda: _configure_quantinuum_project(create_new=False),
            ),
            AnswerNode(
                "Create new project",
                lambda: _configure_quantinuum_project(create_new=True),
            ),
        ],
    )
    for answer in project_tree.answers:
        answer.next_question = None

    run_choice_tree(project_tree)

    return "", []


def _configure_quantinuum_project(create_new: bool):
    """Select or create the Nexus project stored in the MPQP configuration."""
    try:
        import qnexus as qnx

        project_name = input("Enter your Quantinuum Nexus project name: ").strip()
        project_name = project_name.strip("\"'")
        if not project_name:
            return "Empty project name", []

        if create_new:
            project = qnx.projects.create(name=project_name)
        else:
            project = qnx.projects.get(name=project_name)

        qnx.context.set_active_project(project)
        save_env_variable("QUANTINUUM_PROJECT_NAME", project_name)
        save_env_variable("QUANTINUUM_CONFIGURED", "True")
        return "Quantinuum Nexus account correctly configured.", []
    except Exception as error:
        return f"Quantinuum Nexus configuration failed.\n{error}", []


def get_quantinuum_account_info() -> str:
    """Return the Nexus user, project, and current connection status."""
    configured = get_env_variable("QUANTINUUM_CONFIGURED") == "True"
    project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")

    if not configured or not project_name:
        return "Account not configured"

    try:
        import qnexus as qnx

        qnx.projects.get(name=project_name)
        user_name = qnx.users.get_self().display_name or "Not available"
        connection_status = "Connected"
    except Exception:
        user_name = "Not available"
        connection_status = "Unable to connect"

    return f"""    User name: {user_name}
    Project: {project_name}
    Connection status: {connection_status}"""


def check_quantinuum_connection() -> bool:
    """Return whether the configured Nexus project can be accessed."""
    try:
        import qnexus as qnx

        configured = get_env_variable("QUANTINUUM_CONFIGURED") == "True"
        project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")
        if not configured or not project_name:
            return False

        qnx.projects.get(name=project_name)
        return True
    except Exception:
        return False


def _activate_quantinuum_project() -> None:
    """Activate the configured Nexus project using existing authentication."""
    import qnexus as qnx

    configured = get_env_variable("QUANTINUUM_CONFIGURED") == "True"
    project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")
    if not configured or not project_name:
        raise RuntimeError(
            "No Quantinuum Nexus project is configured. Run setup_connections "
            "and configure Quantinuum Nexus first."
        )

    active_project = qnx.context.get_active_project()
    if active_project is not None and active_project.annotations.name == project_name:
        return

    try:
        project = qnx.projects.get(name=project_name)
    except Exception as error:
        raise RuntimeError(
            f"Unable to access the configured Quantinuum Nexus project "
            f"'{project_name}'. Check your authentication and project "
            "configuration."
        ) from error
    qnx.context.set_active_project(project)


def get_quantinuum_config(device: QUANTINUUMDevice):
    """Return the Nexus backend configuration associated with a device."""
    import qnexus as qnx

    _activate_quantinuum_project()

    if device == QUANTINUUMDevice.NEXUS_AER_SIMULATOR:
        return qnx.AerConfig()
    if device == QUANTINUUMDevice.NEXUS_AER_STATE_SIMULATOR:
        return qnx.AerStateConfig()
    if device == QUANTINUUMDevice.NEXUS_QULACS_SIMULATOR:
        return qnx.QulacsConfig()
    return qnx.QuantinuumConfig(device_name=device.value)


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
