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
    import mpqp.execution.connection.google_connection as cirqc
    import mpqp.execution.connection.ibm_connection as ibmqc
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
    print(cirqc.get_google_account_info())
    input("Press 'Enter' to continue")
    return "", []


def main_setup():
    """Main function of the script, triggering the choice selection, and guiding
    you through the steps needed to configure each provider access. This
    function has to be executed from a terminal like environment, allowing you
    to type tokens and alike."""
    from mpqp.execution.connection.aws_connection import setup_aws_braket_account
    from mpqp.execution.connection.ibm_connection import setup_ibm_account
    from mpqp.execution.connection.key_connection import config_ionq_key
    from mpqp.execution.connection.qlm_connection import setup_qlm_account
    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    # def no_op():
    #     return "", []

    setup_tree = QuestionNode(
        "~~~~~ MPQP REMOTE CONFIGURATION ~~~~~",
        [
            AnswerNode("IBM", setup_ibm_account),
            AnswerNode("QLM", setup_qlm_account),
            AnswerNode("Amazon Braket", setup_aws_braket_account),
            AnswerNode("IonQ", config_ionq_key),
            AnswerNode("Recap", print_config_info),
            # AnswerNode(
            #     "Cirq",
            #     no_op,
            #     next_question=QuestionNode(
            #         "~~~~~ Cirq REMOTE CONFIGURATION ~~~~~",
            #         [
            #             AnswerNode("↩ Return", no_op),
            #         ],
            #     ),
            # ),
        ],
    )

    # TODO: to avoid having to manually set that, we could add this as an option
    # to the run choice tree
    for answer in setup_tree.answers:
        if answer.label == "Cirq" and answer.next_question is not None:
            for answer in answer.next_question.answers:
                answer.next_question = setup_tree
        else:
            answer.next_question = setup_tree

    run_choice_tree(setup_tree)


if __name__ == "__main__":
    try:
        main_setup()
    except KeyboardInterrupt:
        exit()
