#! /usr/bin/env python3
"""The ``setup_connections`` script helps you configuring the connections for
all the supported remote backends. In time, it will also guide you to retrieve
the tokens, passwords, etc... but for now, it is a prerequisite that you already
have these credentials to use this script.

Information concerning which provider is configured and related credentials are
stored in the ``~/.mpqp/.env`` file."""

import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"


def print_config_info():
    """Displays the information stored for each provider."""
    import mpqp.environment.env_manager as env_m
    import mpqp.execution.connection.aws_connection as awsc
    import mpqp.execution.connection.azure_connection as azuc
    import mpqp.execution.connection.ibm_connection as ibmqc
    import mpqp.execution.connection.ionq_connection as ionqc
    from mpqp.tools.errors import IBMRemoteExecutionError

    """Prints the info concerning each provider's registered account."""
    print("===== IBM Quantum info : ===== ")
    try:
        print(ibmqc.get_active_account_info())
    except IBMRemoteExecutionError as err:
        if "Unable to find account" in str(err) or "No IBM account configured" in str(
            err
        ):
            print("Account not configured")
        else:
            print(f"{err}")

    print("===== Qaptiva QLMaaS info : ===== ")
    user_name = env_m.get_env_variable("QLM_USER")
    if user_name == "":
        print("Account not configured")
    else:
        print("Current user: ", env_m.get_env_variable("QLM_USER"))
    print("===== AWS Braket info : ===== ")
    try:
        print(awsc.get_aws_braket_account_info())
    except Exception as err:
        if "No AWS Braket account configured" in str(err):
            print("Account not configured")
        else:
            print("Error occurred when getting AWS account info.")
    print("===== IonQ info : ===== ")
    try:
        print(ionqc.get_ionq_account_info())
    except Exception as err:
        print("Error occurred when getting IonQ account info.")
    print("===== Azure info : ===== ")
    try:
        print(azuc.get_azure_account_info())
    except Exception as err:
        print("Error occurred when getting Azure account info.")
    input("Press 'Enter' to continue")
    return "", []


def delete_config():
    """Delete stored credentials for a selected provider."""
    from mpqp.environment.env_manager import get_env_variable
    from mpqp.execution.connection.aws_connection import delete_aws_braket_account
    from mpqp.execution.connection.azure_connection import delete_azure_account
    from mpqp.execution.connection.ibm_connection import delete_ibm_account
    from mpqp.execution.connection.ionq_connection import delete_ionq_account
    from mpqp.execution.connection.qlm_connection import delete_qlm_account
    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    def delete_all():
        if get_env_variable("IBM_CONFIGURED") == "True":
            delete_ibm_account()
        if get_env_variable("QLM_CONFIGURED") == "True":
            delete_qlm_account()
        if get_env_variable("AWS_CONFIGURED") == "True":
            delete_aws_braket_account()
        if get_env_variable("IONQ_CONFIGURED") == "True":
            delete_ionq_account()
        if get_env_variable("AZURE_CONFIGURED") == "True":
            delete_azure_account()

        return "All accounts deleted", []

    delete_tree = QuestionNode(
        "Select provider configuration to delete:",
        [
            AnswerNode("IBM", delete_ibm_account),
            AnswerNode("QLM", delete_qlm_account),
            AnswerNode("Braket", delete_aws_braket_account),
            AnswerNode("IonQ", delete_ionq_account),
            AnswerNode("Azure", delete_azure_account),
            AnswerNode("All", delete_all),
        ],
        leaf_loop_to_here=True,
    )

    run_choice_tree(delete_tree)
    return "", []


def main_setup():
    """Main function of the script, triggering the choice selection, and guiding
    you through the steps needed to configure each provider access. This
    function has to be executed from a terminal like environment, allowing you
    to type tokens and alike."""
    from mpqp.execution.connection.aws_connection import setup_aws_braket_account
    from mpqp.execution.connection.azure_connection import config_azure_account
    from mpqp.execution.connection.ibm_connection import setup_ibm_account
    from mpqp.execution.connection.ionq_connection import config_ionq_key
    from mpqp.execution.connection.qlm_connection import setup_qlm_account
    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    setup_tree = QuestionNode(
        "~~~~~ MPQP REMOTE CONFIGURATION ~~~~~",
        [
            AnswerNode("IBM", setup_ibm_account),
            AnswerNode("QLM", setup_qlm_account),
            AnswerNode("Amazon Braket", setup_aws_braket_account),
            AnswerNode("IonQ", config_ionq_key),
            AnswerNode("Azure", config_azure_account),
            AnswerNode("Recap", print_config_info),
            AnswerNode("Delete a configuration", delete_config),
        ],
        leaf_loop_to_here=True,
    )

    run_choice_tree(setup_tree)


if __name__ == "__main__":
    try:
        main_setup()
    except KeyboardInterrupt:
        exit()
