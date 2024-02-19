#! /usr/bin/env python3

import mpqp.execution.connection.aws_connection as awsc
import mpqp.execution.connection.env_manager as env_m
import mpqp.execution.connection.ibm_connection as ibmqc
import mpqp.execution.connection.qlm_connection as qlmc
from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree
from mpqp.tools.errors import IBMRemoteExecutionError


def print_config_info():
    """
    Prints the info concerning each provider's registered account
    """
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
    input("Press 'Enter' to continue")
    return "", []


def main_setup():
    """Function called by the script bellow to setup all connections or to get
    information about the existing ones."""

    setup_tree = QuestionNode(
        "~~~~~ MPQP REMOTE CONFIGURATION ~~~~~",
        [
            AnswerNode("IBM configuration", ibmqc.setup_ibm_account),
            AnswerNode("QLM configuration", qlmc.setup_qlm_account),
            AnswerNode("Amazon Braket configuration", awsc.setup_aws_braket_account),
            AnswerNode("Config information", print_config_info),
        ],
    )

    for answer in setup_tree.answers:
        answer.next_question = setup_tree

    run_choice_tree(setup_tree)


if __name__ == "__main__":
    main_setup()
