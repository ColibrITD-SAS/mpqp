from termcolor import colored

from mpqp.environment.env_manager import get_env_variable, save_env_variable


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
            print(colored("Quantinuum Nexus configuration failed.", "red"))
            print(colored(str(err), "red"))
            input("Press 'Enter' to continue")
            return "", []

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

    project_tree.answers[0].next_question = None
    project_tree.answers[1].next_question = None

    run_choice_tree(project_tree)
    return "", []


def get_quantinuum_account_info() -> str:
    """Return the locally configured Quantinuum Nexus project name."""

    project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")

    if project_name == "":
        return "Account not configured"

    return "   QUANTINUUM_PROJECT_NAME: " + project_name


def get_quantinuum_config(device_name: str):
    """Return a Quantinuum Nexus backend configuration."""

    import qnexus as qnx

    project_name = get_env_variable("QUANTINUUM_PROJECT_NAME")

    if project_name == "":
        raise RuntimeError(
            "No Quantinuum Nexus project configured. "
            "Run setup_connections and configure Quantinuum Nexus first."
        )

    qnx.login()

    project = qnx.projects.get(name=project_name)
    qnx.context.set_active_project(project)

    return qnx.QuantinuumConfig(device_name=device_name)


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
