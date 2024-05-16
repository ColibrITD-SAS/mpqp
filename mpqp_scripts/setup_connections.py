#! /usr/bin/env python3
"""The ``setup_connections`` script helps you configuring the connections for
all the supported remote backends. In time, it will also guide you to retrieve
the tokens, passwords, etc... but for now, it is a prerequisite that you already
have these credentials to use this script.

Information concerning which provider is configured and related credentials are
stored in the ``~/.mpqp`` file."""

import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"


def print_config_info():
    """Displays the information stored for each provider."""
    import mpqp.execution.connection.aws_connection as awsc
    import mpqp.execution.connection.env_manager as env_m
    import mpqp.execution.connection.ibm_connection as ibmqc
    import mpqp.execution.connection.google_connection as cirqc
    from mpqp.tools.errors import IBMRemoteExecutionError

    """Prints the info concerning each provider's registered account."""
    print("===== IBM Quantum info : ===== ")
    try:
        print(ibmqc.get_active_account_info())
    except IBMRemoteExecutionError as err:
        if "No IBM Q account configured" in str(err):
            print("Account not configured")

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
    print("===== Cirq info : ===== ")
    try:
        print(cirqc.get_google_account_info())
    except Exception as err:
            print("Error occurred when getting Cirq account info.")
    input("Press 'Enter' to continue")
    return "", []


def main_setup():
    """Main function of the script, triggering the choice selection, and guiding
    you through the steps needed to configure each provider access. This
    function has to be executed from a terminal like environment, allowing you
    to type tokens and alike."""
    import mpqp.execution.connection.aws_connection as awsc
    import mpqp.execution.connection.ibm_connection as ibmqc
    import mpqp.execution.connection.qlm_connection as qlmc
    import mpqp.execution.connection.google_connection as cirqc
    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    def return_action():
        return "", []
    
    setup_tree = QuestionNode(
        "~~~~~ MPQP REMOTE CONFIGURATION ~~~~~",
        [
            AnswerNode("IBM configuration", ibmqc.setup_ibm_account),
            AnswerNode("QLM configuration", qlmc.setup_qlm_account),
            AnswerNode("Amazon Braket configuration", awsc.setup_aws_braket_account),
            AnswerNode("Cirq configuration", return_action),
            AnswerNode("Config information", print_config_info),
        ],
    )

    cirq_setup_tree = QuestionNode(
        "~~~~~ Cirq REMOTE CONFIGURATION ~~~~~",
        [
            AnswerNode("↩", return_action),
            AnswerNode("Ionq configuration", cirqc.config_ionq_account),
        ],
    )

    for answer in setup_tree.answers:
        if answer.label == "Cirq configuration":
            answer.next_question = cirq_setup_tree
        else:
            answer.next_question = setup_tree

    for answer in cirq_setup_tree.answers:
        if answer.label == "↩":
            answer.next_question = setup_tree
        else:
            answer.next_question = cirq_setup_tree

    run_choice_tree(setup_tree)

if __name__ == "__main__":
    try:
        main_setup()
    except KeyboardInterrupt:
        exit()
