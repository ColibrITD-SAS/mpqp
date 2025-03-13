import os
from typing import TYPE_CHECKING, Any, Optional, Union

from termcolor import colored
from typeguard import typechecked

if TYPE_CHECKING:
    from braket.devices.device import Device as BraketDevice

from configparser import ConfigParser
from getpass import getpass
from pathlib import Path

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.execution.devices import AWSDevice
from mpqp.tools.errors import AWSBraketRemoteExecutionError


def setup_aws_braket_account() -> tuple[str, list[Any]]:
    """Set-up the connection to an Amazon Braket account using user input.

    This function checks whether an Amazon Braket account is already configured
    and prompts the user to update it if needed. The function attempts to configure
    the Amazon Braket account using two authentication methods:

    IAM (Identity and Access Management):
        - The user is guided to enter their AWS access key, secret access key, and region.
        - Credentials are stored in the default AWS credentials file.

    SSO (Single Sign-On):
        - The user is guided through the process of configuring SSO authentication.
        - SSO credentials, including the session token, are retrieved and provided by the user
            to complete the authentication process.

    It then collects the user's AWS access key, AWS secret key (hidden input),
    AWS session token (hidden input) in case of SSO auth and the AWS region for Amazon Braket.

    Returns:
        A tuple containing a message indicating the result of the setup (e.g.,
        success, cancelled, or error, ...) and an empty list. The list is
        included for consistency with the existing code structure.
    """
    from braket.aws import AwsSession

    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    if get_env_variable("BRAKET_CONFIGURED") == "True":
        decision = input(
            "An Amazon Braket account is already configured. Do you want to update it? [y/N] "
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    braket_auth_choices = QuestionNode(
        "Choose your Amazon Braket authentication method: ",
        [
            AnswerNode("IAM (Identity and Access Management)", configure_account_iam),
            AnswerNode("SSO (Single Sign-On)", configure_account_sso),
        ],
    )
    run_choice_tree(braket_auth_choices)

    try:
        session = AwsSession()
        save_env_variable("AWS_DEFAULT_REGION", session.region)

        return "Amazon Braket account correctly configured", []

    except Exception as e:
        print(colored("Error configuring Amazon Braket account", "red"))
        print(colored(str(e), "red"))
        input("Press 'Enter' to continue")
        return "", []


def update_aws_credentials_file(
    profile_name: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: Optional[str],
    region: str,
):
    """Create or update the ``~/.aws/credentials`` file with the provided credentials.
    Ensure that the directory and file exist before making changes.
    """

    credentials_file = Path.home() / ".aws" / "credentials"

    credentials_dir = credentials_file.parent
    if not credentials_dir.exists():
        credentials_dir.mkdir(parents=True, exist_ok=True)

    config = ConfigParser()
    if credentials_file.exists():
        config.read(credentials_file)

        if config.has_section(profile_name):
            config.remove_section(profile_name)

    config.add_section(profile_name)
    config[profile_name]["aws_access_key_id"] = access_key_id
    config[profile_name]["aws_secret_access_key"] = secret_access_key

    if session_token:
        config[profile_name]["aws_session_token"] = session_token

    config[profile_name]["region"] = region

    with open(credentials_file, "w") as f:
        config.write(f)


def configure_account_iam() -> tuple[str, list[Any]]:
    """Configure IAM authentication for Amazon Braket."""

    print("Configuring IAM authentication for Amazon Braket...")
    os.system("aws configure")

    print("IAM authentication configured successfully.")
    save_env_variable("BRAKET_AUTH_METHOD", "IAM")
    save_env_variable("BRAKET_CONFIGURED", "True")

    credentials_file = Path.home() / ".aws" / "credentials"

    config = ConfigParser()
    config.read(credentials_file)

    access_key_id = config.get("default", "aws_access_key_id", fallback="")
    secret_access_key = config.get("default", "aws_secret_access_key", fallback="")
    region = config.get("default", "region", fallback="us-east-1")

    update_aws_credentials_file(
        profile_name="default",
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=None,
        region=region,
    )

    return "IAM configuration successful.", []


def get_user_sso_credentials() -> Union[dict[str, str], None]:

    print("Please enter your AWS SSO credentials:")

    try:
        access_key_id = input("Enter AWS access key ID: ").strip()
        secret_access_key = getpass("Enter AWS secret access key (hidden): ").strip()
        session_token = getpass("Enter SSO session token (hidden): ").strip()
        region = input("Enter SSO region: ").strip()

        return {
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
            "session_token": session_token,
            "region": region,
        }

    except Exception as e:
        print(f"An error occurred while getting credentials: {e}")
        return None


def configure_account_sso() -> tuple[str, list[Any]]:
    """Configure SSO authentication for Amazon Braket.
    This function guides the user through the Amazon Braket SSO configuration process.
    """
    print("Configuring SSO authentication for Amazon Braket...")

    sso_credentials = get_user_sso_credentials()
    if not sso_credentials:
        raise Exception("Failed to retrieve SSO credentials after configuration.")

    print("SSO authentication configured successfully.")

    update_aws_credentials_file(
        profile_name="default",
        access_key_id=sso_credentials["access_key_id"],
        secret_access_key=sso_credentials["secret_access_key"],
        session_token=sso_credentials["session_token"],
        region=sso_credentials["region"],
    )

    save_env_variable("BRAKET_AUTH_METHOD", "SSO")
    save_env_variable("BRAKET_CONFIGURED", "True")

    return "SSO configuration successful.", []


def get_aws_braket_account_info() -> str:
    """Retrieves AWS Braket credentials information including access key ID, secret access key,
    and region. For SSO authentication, the session token is also included.

    Returns:
        A formatted string containing AWS credentials information with an
        obfuscated secret access key.

    Examples:
        >>> print(get_aws_braket_account_info())
            Authentication method: IAM
            Access Key ID: 'AKIA26JFZI8JFZ18FI4N'
            Secret Access Key: 'E9oF9*********************************'
            Region: 'us-east-1'

        >>> print(get_aws_braket_account_info())
            Authentication method: SSO
            Access Key ID: 'ASIA26JFEZ6JEOZ9JC7K'
            Secret Access Key: 'FiZp3***********************************'
            SSO Session Token: 'EfGkf2kbI3nfC5V...IIZUf79jofZNF=='
            Region: 'us-east-1'

    Note:
        This function assumes that the AWS credentials are already configured
        in the AWS credentials/config file ``~/.aws/credentials``.

    """
    if get_env_variable("BRAKET_CONFIGURED") == "False":
        raise AWSBraketRemoteExecutionError(
            "Error when trying to get AWS credentials. No AWS Braket account configured."
        )
    import boto3
    from braket.aws import AwsSession

    try:
        boto3_session = boto3.Session(profile_name="default")
        session = AwsSession(boto_session=boto3_session)

        credentials = session.boto_session.get_credentials()
        if credentials is None:
            raise AWSBraketRemoteExecutionError("Could not retrieve AWS credentials")

        access_key_id = credentials.access_key
        secret_access_key = credentials.secret_key
        obfuscated_key = secret_access_key[:5] + "*" * (len(secret_access_key) - 5)

        session_token = credentials.token
        if session_token:
            auth_method = "SSO"
            token_length = len(session_token)
            obfuscated_token = (
                f"{session_token[:15]}...{session_token[-15:]}"
                if token_length > 30
                else session_token
            )

        else:
            obfuscated_token = ""
            auth_method = "IAM"

        region_name = session.boto_session.region_name

    except Exception as e:
        raise AWSBraketRemoteExecutionError(
            "Error when trying to get AWS credentials. Possibly no AWS Braket account configured.\n Trace:"
            + str(e)
        )

    result = f"""    Authentication method: {auth_method}  
    Access Key ID: '{access_key_id}'
    Secret Access Key: '{obfuscated_key}'"""
    if session_token:
        result += f"\n    SSO Session Token: '{obfuscated_token}'"

    result += f"\n    Region: '{region_name}'"
    return result


@typechecked
def get_braket_device(device: AWSDevice, is_noisy: bool = False) -> "BraketDevice":
    """Returns the AwsDevice device associate with the AWSDevice in parameter.

    Args:
        device: AWSDevice element describing which remote/local AwsDevice we want.
        is_noisy: If the expected device is noisy or not.

    Raises:
        AWSBraketRemoteExecutionError: If the device or the region could not be
            retrieved.

    Example:
        >>> device = get_braket_device(AWSDevice.RIGETTI_ANKAA_2)
        >>> device.properties.action['braket.ir.openqasm.program'].supportedResultTypes
        [ResultType(name='Sample', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=50000),
         ResultType(name='Expectation', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=50000),
         ResultType(name='Variance', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=50000),
         ResultType(name='Probability', observables=None, minShots=10, maxShots=50000)]

    """
    from braket.devices import LocalSimulator

    if not device.is_remote():
        if is_noisy:
            return LocalSimulator("braket_dm")
        else:
            return LocalSimulator()

    import boto3
    import pkg_resources
    from botocore.exceptions import NoRegionError
    from braket.aws import AwsDevice, AwsSession

    try:
        braket_client = boto3.client("braket", region_name=device.get_region())
        aws_session = AwsSession(braket_client=braket_client)
        mpqp_version = pkg_resources.get_distribution("mpqp").version[:3]
        aws_session.add_braket_user_agent(
            user_agent="APN/1.0 ColibriTD/1.0 MPQP/" + mpqp_version
        )
        return AwsDevice(device.get_arn(), aws_session=aws_session)
    except ValueError as ve:
        raise AWSBraketRemoteExecutionError(
            "Failed to retrieve remote AWS device. Please check the arn, or if the "
            "device is accessible from your region.\nTrace: " + str(ve)
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
    from braket.aws import AwsSession

    if get_env_variable("BRAKET_CONFIGURED") == "True":
        return [
            task["quantumTaskArn"]
            for task in (
                AwsSession().braket_client.search_quantum_tasks(filters=[])[
                    "quantumTasks"
                ]
            )
        ]
    return []


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
