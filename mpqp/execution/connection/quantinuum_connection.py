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

        qnx.login()
        return _setup_quantinuum_project()

    except Exception as err:
        save_env_variable("QUANTINUUM_CONFIGURED", "False")
        print(colored("Quantinuum Nexus configuration failed.", "red"))
        print(colored(str(err), "red"))
        input("Press 'Enter' to continue")
        return "", []


def _setup_quantinuum_project():
    """Choose an existing Quantinuum Nexus project or create a new one."""
    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    def setup_project(create_new: bool):
        try:
            import qnexus as qnx

            project_name = input("Enter your Quantinuum Nexus project name: ").strip()
            project_name = project_name.strip("\"'")

            if project_name == "":
                print(colored("Empty project name", "red"))
                save_env_variable("QUANTINUUM_CONFIGURED", "False")
                return "", []

            if create_new:
                project = qnx.projects.create(name=project_name)
            else:
                project = qnx.projects.get(name=project_name)

            qnx.context.set_active_project(project)

            save_env_variable("QUANTINUUM_PROJECT_NAME", project_name)
            save_env_variable("QUANTINUUM_CONFIGURED", "True")

            return "Quantinuum Nexus account correctly configured.", []

        except Exception as err:
            save_env_variable("QUANTINUUM_CONFIGURED", "False")
            return (
                f"Quantinuum Nexus configuration failed.\n{err}",
                [],
            )

    def use_existing_project():
        return setup_project(create_new=False)

    def create_new_project():
        return setup_project(create_new=True)

    project_tree = QuestionNode(
        "Choose Quantinuum Nexus project option:",
        [
            AnswerNode("Use existing project", use_existing_project),
            AnswerNode("Create new project", create_new_project),
        ],
    )
    for answer in project_tree.answers:
        answer.next_question = None

    run_choice_tree(project_tree)

    return "Quantinuum Nexus project configured successfully.", []


def get_quantinuum_account_info() -> str:
    """Return the locally configured Quantinuum Nexus project name."""
    configured = get_env_variable("QUANTINUUM_CONFIGURED") == "True"
    project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")

    if not configured or not project_name:
        return "Account not configured"

    return "   QUANTINUUM_PROJECT_NAME: " + project_name


def _activate_quantinuum_project() -> None:
    """Log in to Nexus and activate the project configured in MPQP."""
    import qnexus as qnx

    project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")

    if not project_name:
        raise RuntimeError(
            "No Quantinuum Nexus project configured. "
            "Run setup_connections and configure Quantinuum Nexus first."
        )

    qnx.login()

    project = qnx.projects.get(name=project_name)
    qnx.context.set_active_project(project)


def get_quantinuum_config(device: QUANTINUUMDevice):
    """Return the Nexus backend configuration associated with a device."""
    import qnexus as qnx

    _activate_quantinuum_project()

    if device == QUANTINUUMDevice.NEXUS_AER_SIMULATOR:
        return qnx.AerConfig()
    if device == QUANTINUUMDevice.NEXUS_AER_STATE_SIMULATOR:
        return qnx.AerStateConfig()
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
