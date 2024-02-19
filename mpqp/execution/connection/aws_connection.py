import os
from typing import Any

from botocore.exceptions import NoRegionError
from braket.aws import AwsDevice, AwsSession
from braket.devices import LocalSimulator
from braket.devices.device import Device as BraketDevice
from termcolor import colored
from typeguard import typechecked

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.execution.devices import AWSDevice
from mpqp.tools.errors import AWSBraketRemoteExecutionError


def setup_aws_braket_account() -> tuple[str, list[Any]]:
    """Setups the connection to an Amazon Braket account using user input.

    This function checks whether an Amazon Braket account is already configured
    and prompts the user to update it if needed. It then collects the user's AWS
    access key, AWS secret key (hidden input), and the AWS region for Amazon
    Braket. The function attempts to configure the Amazon Braket account using
    the provided credentials.

    Returns:
        A tuple containing a message indicating the result of the setup (e.g.,
        success, cancelled, or error, ...) and an empty list. The list is
        included for consistency with the existing code structure.
    """

    already_configured = get_env_variable("BRAKET_CONFIGURED") == "True"

    if already_configured:
        decision = input(
            "An Amazon Braket account is already configured. Do you want to update it? [y/N] "
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    try:
        os.system("aws configure")
        save_env_variable("BRAKET_CONFIGURED", "True")
        return "Amazon Braket account correctly configured", []

    except Exception as e:
        print(colored("Error configuring Amazon Braket account", "red"))
        print(colored(str(e), "red"))
        input("Press 'Enter' to continue")
        return "", []


def get_aws_braket_account_info() -> str:
    """Get AWS Braket credentials information including access key ID,
    obfuscated secret access key, and region.

    Example:
        >>> get_aws_braket_account_info()
            access_key_id: 'AKIA26NYJ***********'
            secret_access_key: 'sMDad***********************************'
            region: 'us-east-1'

    Returns:
        A formatted string containing AWS credentials information with an
        obfuscated secret access key.

    """
    if get_env_variable("BRAKET_CONFIGURED") == "False":
        raise AWSBraketRemoteExecutionError(
            "Error when trying to get AWS credentials. No AWS Braket account configured."
        )

    try:
        session = AwsSession()

        # get the AWS Braket user access key, secret key and region
        access_key_id = session.boto_session.get_credentials().access_key

        secret_access_key = session.boto_session.get_credentials().secret_key
        obfuscate_key = secret_access_key[:5] + "*" * (len(secret_access_key) - 5)

        region_name = session.boto_session.region_name
    except Exception as e:
        raise AWSBraketRemoteExecutionError(
            "Error when trying to get AWS credentials. No AWS Braket account configured.\n Trace:"
            + str(e)
        )

    return f"""    access_key_id: '{access_key_id}'
    secret_access_key: '{obfuscate_key}' 
    region: '{region_name}'"""


@typechecked
def get_braket_device(device: AWSDevice) -> BraketDevice:
    """
    Returns the AwsDevice device associate with the AWSDevice in parameter.

    Example:
        >>> device = get_braket_device(AWSDevice.BRAKET_RIGETTI_ASPEN_M_3)
        >>> device.properties.action['braket.ir.jaqcd.program'].supportedResultTypes
        [ResultType(name='Sample', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=100000),
         ResultType(name='Expectation', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=100000),
         ResultType(name='Variance', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=100000),
         ResultType(name='Probability', observables=None, minShots=10, maxShots=100000)]

    Args:
        device: AWSDevice element describing which remote/local AwsDevice we want.

    Raises:
        AWSBraketRemoteExecutionError
    """

    if not device.is_remote():
        return LocalSimulator()

    try:
        return AwsDevice(device.get_arn())
    except ValueError as ve:
        raise AWSBraketRemoteExecutionError(
            "Failed to retrieve remote AWS device. Please check the arn, or if the "
            "device is accessible from your region..\nTrace: " + str(ve)
        )
    except NoRegionError as err:
        raise AWSBraketRemoteExecutionError(
            "Failed to find the region related with your aws credentials. Make sure you"
            "configured correctly your AWS account with 'setup_connections.py' script."
            "\nTrace: " + str(err)
        )


def get_all_task_ids() -> list[str]:
    """Retrieves all the task ids of this account/group from AWS.

    Example:
        >>> get_all_task_ids()
        ['arn:aws:braket:us-east-1:752542621531:quantum-task/6a46ae9a-d02f-4a23-b46f-eae43471bc22',
         'arn:aws:braket:us-east-1:752542621531:quantum-task/11db7e68-2b17-4b00-a4ec-20f662fd4876',
         'arn:aws:braket:us-east-1:752542621531:quantum-task/292d329f-727c-4b92-83e1-7d4bedd4b243',
         'arn:aws:braket:us-east-1:752542621531:quantum-task/4b94c703-2ce8-480b-b3f3-ecb2580dbb82',
         'arn:aws:braket:us-east-1:752542621531:quantum-task/edc094aa-23e8-4a8c-87be-f2e09281d79d',
         'arn:aws:braket:us-east-1:752542621531:quantum-task/af9e623a-dd1c-4ecb-9db6-dbbd1af08110']
    """
    return [
        task["quantumTaskArn"]
        for task in (
            AwsSession().braket_client.search_quantum_tasks(filters=[])["quantumTasks"]
        )
    ]


def get_all_partial_ids() -> list[str]:
    """Retrieves all the task ids of this account/group from AWS and extracts the
    significant part.

    Example:
        >>> get_all_partial_ids()
        ['6a46ae9a-d02f-4a23-b46f-eae43471bc22',
         '11db7e68-2b17-4b00-a4ec-20f662fd4876',
         '292d329f-727c-4b92-83e1-7d4bedd4b243',
         '4b94c703-2ce8-480b-b3f3-ecb2580dbb82',
         'edc094aa-23e8-4a8c-87be-f2e09281d79d',
         'af9e623a-dd1c-4ecb-9db6-dbbd1af08110']
    """
    return [id.split("/")[-1] for id in get_all_task_ids()]
