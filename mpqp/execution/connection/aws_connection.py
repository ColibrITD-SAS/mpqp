from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from termcolor import colored

if TYPE_CHECKING:
    from boto3.session import Session as Boto3Session
    from braket.devices.device import Device as BraketDevice

from mpqp.environment.env_manager import (
    get_env_variable,
    remove_env_variable,
    save_env_variable,
)
from mpqp.execution.devices import AWSDevice
from mpqp.tools.errors import AWSBraketRemoteExecutionError


def _get_aws_files() -> tuple[Path, Path]:
    """Return the AWS credentials and config file paths."""

    import os

    default_aws_directory = Path.home() / ".aws"
    credentials_file = Path(
        os.environ.get("AWS_SHARED_CREDENTIALS_FILE")
        or default_aws_directory / "credentials"
    ).expanduser()
    config_file = Path(
        os.environ.get("AWS_CONFIG_FILE") or default_aws_directory / "config"
    ).expanduser()
    return credentials_file, config_file


def _load_aws_file(path: Path) -> ConfigParser:
    """Read an AWS credentials or config file."""
    parser = ConfigParser(interpolation=None)
    if path.exists():
        with path.open(encoding="utf-8") as aws_file:
            parser.read_file(aws_file)
    return parser


def _save_aws_file(path: Path, parser: ConfigParser) -> None:
    """Write an AWS credentials or config file."""

    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            parser.write(temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        assert temporary_path is not None
        if os.name != "nt":
            temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _get_boto3_session(
    *,
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
) -> "Boto3Session":
    """Create a Boto3 session using the AWS profile selected in MPQP."""

    import os

    import boto3

    selected_profile = profile_name or get_env_variable("BRAKET_PROFILE") or "default"

    # AWS_DEFAULT_REGION was previously set to an empty string when the local
    # AWS configuration was deleted. Remove that old value so Boto3 reads the
    # region from the selected AWS profile.
    if os.environ.get("AWS_DEFAULT_REGION") == "":
        os.environ.pop("AWS_DEFAULT_REGION")

    return boto3.Session(profile_name=selected_profile, region_name=region_name)


def _get_aws_profile_region(profile_name: str) -> Optional[str]:
    """Return the region set in an AWS profile."""

    from botocore.session import Session as BotocoreSession

    region = BotocoreSession(profile=profile_name).get_scoped_config().get("region")
    if not isinstance(region, str) or not region.strip():
        return None
    return region.strip()


def _is_identity_center_profile(profile_name: str) -> bool:
    """Check whether the AWS profile is configured for IAM Identity Center."""

    from botocore.session import Session as BotocoreSession

    profile = BotocoreSession(profile=profile_name).get_scoped_config()
    if profile.get("sso_session"):
        return True
    return all(
        profile.get(key)
        for key in ("sso_start_url", "sso_region", "sso_account_id", "sso_role_name")
    )


def setup_aws_braket_account() -> tuple[str, list[Any]]:
    """Set up or update the Amazon Braket connection.

    The user can select an existing AWS profile or provide short-term
    credentials.

    For an existing AWS profile, MPQP saves only its name and does not write to
    the profile. Short-term credentials entered in MPQP are stored in the
    ``mpqp-temporary`` AWS profile.

    Returns:
        A tuple containing a message indicating whether the setup finished or
        was canceled, and an empty list. The list is included for consistency
        with the other connection setup functions.
    """

    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    was_configured = get_env_variable("BRAKET_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "An Amazon Braket account is already configured. Do you want to update "
            "it? [y/N] "
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    authentication_tree = QuestionNode(
        "Choose an Amazon Braket connection method:",
        [
            AnswerNode(
                "Option 1: Existing AWS profile",
                configure_account_profile,
            ),
            AnswerNode(
                "Option 2: Short-term credentials",
                configure_account_temporary_credentials,
            ),
        ],
    )
    for answer in authentication_tree.answers:
        answer.next_question = None

    run_choice_tree(authentication_tree)
    return "Amazon Braket setup finished.", []


def update_aws_credentials_file(
    profile_name: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: str,
    region: str,
) -> None:
    """Save AWS credentials and configuration under a profile name.

    The access keys are saved in the AWS credentials file, and the region is
    saved in the AWS config file.
    """

    credentials_file, config_file = _get_aws_files()

    credentials = _load_aws_file(credentials_file)

    if credentials.has_section(profile_name):
        credentials.remove_section(profile_name)

    credentials.add_section(profile_name)
    credentials[profile_name]["aws_access_key_id"] = access_key_id
    credentials[profile_name]["aws_secret_access_key"] = secret_access_key
    credentials[profile_name]["aws_session_token"] = session_token

    config = _load_aws_file(config_file)

    config_profile_name = (
        "default" if profile_name == "default" else f"profile {profile_name}"
    )
    if not config.has_section(config_profile_name):
        config.add_section(config_profile_name)
    config[config_profile_name]["region"] = region

    _save_aws_file(credentials_file, credentials)
    _save_aws_file(config_file, config)


def configure_account_temporary_credentials() -> tuple[str, list[Any]]:
    """Configure short-term credentials in the ``mpqp-temporary`` profile."""

    from getpass import getpass

    print("Please enter your AWS access keys, session token, and region:")
    access_key_id = input("Enter AWS access key ID: ").strip()
    secret_access_key = getpass("Enter AWS secret access key (hidden): ").strip()
    session_token = getpass("Enter AWS session token (hidden): ").strip()
    region = input("Enter AWS region: ").strip()

    if not access_key_id or not secret_access_key or not session_token or not region:
        return (
            "AWS access key, secret access key, session token, and region are "
            "required.",
            [],
        )

    import boto3

    try:
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
            region_name=region,
        )
        session.client("sts").get_caller_identity()
    except Exception as error:
        return f"Invalid or expired AWS credentials.\nTrace: {error}", []

    profile_name = "mpqp-temporary"
    try:
        update_aws_credentials_file(
            profile_name=profile_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            region=region,
        )
    except Exception as error:
        return f"Unable to save the AWS profile.\nTrace: {error}", []
    remove_env_variable("AWS_DEFAULT_REGION")

    save_env_variable("BRAKET_AUTH_METHOD", "TEMPORARY")
    save_env_variable("BRAKET_PROFILE", profile_name)
    save_env_variable("BRAKET_CONFIGURED", "True")

    return "AWS temporary credentials configured successfully.", []


def delete_aws_braket_account() -> tuple[str, list[Any]]:
    """Remove the Amazon Braket configuration saved in MPQP.

    The local profile created by MPQP for temporary credentials is deleted. If
    the user selected an existing AWS profile, only its name is removed from
    MPQP's configuration. The AWS profile and account are kept.

    Returns:
        A tuple containing a message describing the result and an empty list.
    """

    authentication_method = get_env_variable("BRAKET_AUTH_METHOD")
    configured_profile = get_env_variable("BRAKET_PROFILE")
    profile_name = configured_profile or "default"
    managed_profile = {"TEMPORARY": "mpqp-temporary"}.get(authentication_method)
    profile_is_managed = (
        managed_profile is not None and configured_profile == managed_profile
    )
    if profile_is_managed:
        deletion_description = (
            "This will delete the temporary AWS credentials stored in the "
            f"'{profile_name}' profile."
        )
    else:
        deletion_description = (
            "This will remove the AWS configuration saved in MPQP. "
            f"The AWS '{profile_name}' profile will not be deleted."
        )
    decision = input(
        colored(
            f"{deletion_description} Continue? [y/N] ",
            "yellow",
        )
    )
    if decision.lower().strip() != "y":
        return "Canceled.", []

    errors: list[str] = []

    if profile_is_managed:
        credentials_file, config_file = _get_aws_files()
        config_profile = f"profile {profile_name}"
        for path, section, label in (
            (credentials_file, profile_name, "credentials"),
            (config_file, config_profile, "config"),
        ):
            try:
                if not path.exists():
                    continue
                parser = _load_aws_file(path)
                if parser.remove_section(section):
                    _save_aws_file(path, parser)
            except Exception as error:
                errors.append(f"Failed to update AWS {label} file: {error}")

    if errors:
        return "Unable to delete the AWS profile.\n" + "\n".join(errors), []

    save_env_variable("BRAKET_CONFIGURED", "False")
    save_env_variable("BRAKET_AUTH_METHOD", "")
    save_env_variable("BRAKET_PROFILE", "")
    remove_env_variable("AWS_DEFAULT_REGION")

    if profile_is_managed:
        return (
            f"AWS profile '{profile_name}' deleted and its MPQP configuration "
            "removed.",
            [],
        )
    return (
        "Amazon Braket configuration removed from MPQP. "
        f"AWS profile '{profile_name}' remains available.",
        [],
    )


def configure_account_profile(
    profile_name: Optional[str] = None,
) -> tuple[str, list[Any]]:
    """Configure MPQP to use an existing AWS profile."""

    if profile_name is None:
        profile_name = input("Enter AWS profile name: ").strip()
    if not profile_name:
        return "AWS profile name is required.", []

    identity_center_profile = False
    try:
        identity_center_profile = _is_identity_center_profile(profile_name)
        profile_region = _get_aws_profile_region(profile_name)
        if profile_region is None:
            return (
                f"AWS profile '{profile_name}' has no default region. Add a region "
                "to the profile before selecting it in MPQP.",
                [],
            )
        session = _get_boto3_session(
            profile_name=profile_name,
            region_name=profile_region,
        )
        session.client("sts").get_caller_identity()
    except Exception as error:
        message = (
            f"Unable to authenticate with AWS profile '{profile_name}'. Verify "
            "that the profile exists and its credentials are valid."
        )
        if identity_center_profile:
            message += (
                "\nIf its IAM Identity Center session has expired, run "
                f"'aws sso login --profile {profile_name}', then try again."
            )
        return f"{message}\nTrace: {error}", []

    remove_env_variable("AWS_DEFAULT_REGION")
    save_env_variable("BRAKET_AUTH_METHOD", "PROFILE")
    save_env_variable("BRAKET_PROFILE", profile_name)
    save_env_variable("BRAKET_CONFIGURED", "True")

    return f"AWS profile '{profile_name}' configured successfully.", []


def get_aws_braket_account_info() -> str:
    """Retrieve information about the AWS configuration used for Amazon Braket.

    The selected profile is validated before its information is returned.

    Returns:
        A formatted string containing the credential source, profile name,
        profile authentication type when applicable, region, and connection
        status.

    Raises:
        AWSBraketRemoteExecutionError: If Amazon Braket is not configured or the
            selected AWS profile cannot be validated.

    Examples:
        >>> print(get_aws_braket_account_info())
            Credential source: Existing AWS profile
            Profile name: 'aws-sso-profile'
            Profile authentication: IAM Identity Center (SSO)
            Region: 'us-east-1'
            Connection status: Connected

        >>> print(get_aws_braket_account_info())
            Credential source: Short-term credentials
            Profile name: 'mpqp-temporary'
            Region: 'us-east-1'
            Connection status: Connected
    """

    if get_env_variable("BRAKET_CONFIGURED") != "True":
        raise AWSBraketRemoteExecutionError("No AWS Braket account configured.")

    try:
        session = _get_boto3_session()
        session.client("sts").get_caller_identity()
    except Exception as error:
        raise AWSBraketRemoteExecutionError(
            "Error when validating the configured AWS profile. Possibly no AWS "
            "Braket account is configured.\nTrace: " + str(error)
        ) from error

    profile_name = get_env_variable("BRAKET_PROFILE") or "default"
    stored_auth_method = get_env_variable("BRAKET_AUTH_METHOD")
    credential_source = {
        "PROFILE": "Existing AWS profile",
        "TEMPORARY": "Short-term credentials",
    }.get(stored_auth_method, "Unknown")
    information = [
        f"    Credential source: {credential_source}",
        f"    Profile name: '{profile_name}'",
    ]
    if stored_auth_method == "PROFILE":
        credentials = session.get_credentials()
        credential_method = getattr(credentials, "method", "")
        if not isinstance(credential_method, str):
            credential_method = ""
        authentication_type = {
            "sso": "IAM Identity Center (SSO)",
            "assume-role": "Assume role",
            "shared-credentials-file": "Access keys",
            "config-file": "Access keys",
            "custom-process": "Credential process",
        }.get(credential_method, "Other")
        information.append(f"    Profile authentication: {authentication_type}")
    information.extend(
        [
            f"    Region: '{session.region_name}'",
            "    Connection status: Connected",
        ]
    )
    return "\n".join(information)


def get_braket_device(
    device: AWSDevice,
    is_noisy: bool = False,
    is_gate_model: bool = True,
) -> "BraketDevice":
    """Return the Braket device associated with the selected MPQP device.

    Args:
        device: AWS device selected for execution.
        is_noisy: Whether to use a simulator that supports noise.
        is_gate_model: Whether to require a device that supports gate-based
            circuits.

    Returns:
        The associated Braket device.

    Raises:
        AWSBraketRemoteExecutionError: If the remote device or its region cannot
            be retrieved.
        DeviceJobIncompatibleError: If the selected device is retired or does
            not support gate-based circuits.

    Example:
        >>> device = get_braket_device(AWSDevice.RIGETTI_ANKAA_3)
        >>> device.properties.action['braket.ir.openqasm.program'].supportedResultTypes
        [ResultType(name='Sample', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=50000),
         ResultType(name='Expectation', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=50000),
         ResultType(name='Variance', observables=['x', 'y', 'z', 'h', 'i'], minShots=10, maxShots=50000),
         ResultType(name='Probability', observables=None, minShots=10, maxShots=50000)]
    """

    from mpqp.tools.errors import (
        AWSBraketRemoteExecutionError,
        DeviceJobIncompatibleError,
    )

    if device.is_retired():
        raise DeviceJobIncompatibleError(
            f"{device.name} has been retired by Amazon Braket and cannot run new "
            "jobs."
        )
    if is_gate_model and not device.is_gate_based():
        raise DeviceJobIncompatibleError(
            f"{device.name} supports Analog Hamiltonian Simulation, not MPQP "
            "QCircuit jobs."
        )

    from braket.devices import LocalSimulator

    if not device.is_remote():
        if is_noisy:
            return LocalSimulator("braket_dm")
        else:
            return LocalSimulator()

    from botocore.exceptions import NoRegionError
    from braket.aws import AwsDevice, AwsSession

    import mpqp

    try:
        boto3_session = _get_boto3_session()
        if device.is_simulator() and not boto3_session.region_name:
            raise NoRegionError()
        region = device.get_region(boto3_session.region_name)

        if boto3_session.region_name != region:
            boto3_session = _get_boto3_session(region_name=region)
        braket_client = boto3_session.client("braket", region_name=region)

        aws_session = AwsSession(
            boto_session=boto3_session,
            braket_client=braket_client,
        )
        aws_session.add_braket_user_agent(
            user_agent="APN/1.0 ColibriTD/1.0 MPQP/" + mpqp.__version__
        )
        braket_device = AwsDevice(device.get_arn(), aws_session=aws_session)

    except ValueError as ve:
        raise AWSBraketRemoteExecutionError(
            "Failed to retrieve remote AWS device. Please check the arn, or if the "
            "device is accessible from your region.\nTrace: " + str(ve)
        )
    except NoRegionError as err:
        raise AWSBraketRemoteExecutionError(
            "Failed to find the region associated with your AWS credentials. Make "
            "sure your AWS account is correctly configured using "
            "'setup_connections'."
            "\nTrace: " + str(err)
        )

    return braket_device


def get_all_task_ids() -> list[str]:
    """Retrieve all Braket task IDs as full ARNs for the selected AWS profile.

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
        boto3_session = _get_boto3_session()
        return [
            task["quantumTaskArn"]
            for task in (
                AwsSession(
                    boto_session=boto3_session
                ).braket_client.search_quantum_tasks(filters=[])["quantumTasks"]
            )
        ]
    return []


def get_all_partial_ids() -> list[str]:
    """Retrieve all Braket task IDs for the selected AWS profile without their
    ARN prefixes.

    Example:
        >>> get_all_partial_ids()
        ['6a46ae9a-d02f-4a23-b46f-eae43471bc22',
         '11db7e68-2b17-4b00-a4ec-20f662fd4876',
         '292d329f-727c-4b92-83e1-7d4bedd4b243',
         '4b94c703-2ce8-480b-b3f3-ecb2580dbb82',
         'edc094aa-23e8-4a8c-87be-f2e09281d79d',
         'af9e623a-dd1c-4ecb-9db6-dbbd1af08110']
    """

    return [task_id.split("/")[-1] for task_id in get_all_task_ids()]
