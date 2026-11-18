from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

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


class _AWSFileState(NamedTuple):
    content: bytes
    mode: int


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


def _serialize_aws_file(parser: ConfigParser) -> bytes:
    """Serialize the AWS configuration so it can be written to a file."""

    from io import StringIO

    serialized_file = StringIO()
    parser.write(serialized_file)
    return serialized_file.getvalue().encode("utf-8")


def _stage_aws_file(path: Path, content: bytes, mode: int) -> Path:
    """Write the new AWS file contents to a temporary file before replacing the original."""

    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    try:
        with temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if os.name != "nt":
            temporary_path.chmod(mode)
        return temporary_path
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_aws_file_state(path: Path) -> Optional[_AWSFileState]:
    """Read an AWS file's contents and permissions."""

    import stat

    if not path.exists():
        return None
    return _AWSFileState(
        content=path.read_bytes(),
        mode=stat.S_IMODE(path.stat().st_mode),
    )


def _aws_file_matches(
    path: Path,
    expected: Optional[_AWSFileState],
) -> bool:
    """Check whether an AWS file matches the expected contents and permissions.

    File permissions are compared only on POSIX systems.
    """

    import os

    current = _read_aws_file_state(path)
    if current is None or expected is None:
        return current is expected
    return current.content == expected.content and (
        os.name == "nt" or current.mode == expected.mode
    )


def _restore_aws_file(
    path: Path,
    original: Optional[_AWSFileState],
    saved: _AWSFileState,
) -> None:
    """Restore an AWS file unless another process has changed it."""

    import os

    if _aws_file_matches(path, original):
        return
    if not _aws_file_matches(path, saved):
        raise OSError(
            f"Cannot restore AWS file '{path}' because it changed after MPQP updated it."
        )
    if original is None:
        path.unlink(missing_ok=True)
        return

    rollback_file = _stage_aws_file(path, original.content, original.mode)
    try:
        # Recheck in case the file changed while preparing the rollback.
        if _aws_file_matches(path, original):
            return
        if not _aws_file_matches(path, saved):
            raise OSError(
                f"Cannot restore AWS file '{path}' because it changed after MPQP updated it."
            )
        os.replace(rollback_file, path)
    finally:
        rollback_file.unlink(missing_ok=True)


def _save_aws_files(
    credentials_file: Path,
    credentials: Optional[ConfigParser],
    config_file: Path,
    config: Optional[ConfigParser],
    *,
    complete_update: Optional[Callable[[], None]] = None,
) -> None:
    """Save the AWS credentials and config files safely.

    Both files are prepared before they are updated. If something fails during
    the update, MPQP tries to restore the previous files.

    If an AWS file is a symlink, MPQP updates the file it points to
    without replacing the symlink.
    """

    import os

    try:
        credentials_file = credentials_file.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise OSError("Could not resolve the AWS credentials file path.") from error
    try:
        config_file = config_file.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise OSError("Could not resolve the AWS config file path.") from error

    if credentials_file == config_file:
        raise ValueError(
            "AWS credentials and config files must use different paths. Check "
            "AWS_SHARED_CREDENTIALS_FILE and AWS_CONFIG_FILE."
        )

    files = (
        (credentials_file, credentials),
        (config_file, config),
    )
    originals: dict[Path, Optional[_AWSFileState]] = {}
    staged_files: dict[Path, Path] = {}
    staged_states: dict[Path, _AWSFileState] = {}
    replaced_files: list[Path] = []
    for path, _ in files:
        originals[path] = _read_aws_file_state(path)

    try:
        for path, parser in files:
            if parser is None:
                continue
            content = _serialize_aws_file(parser)
            staged_states[path] = _AWSFileState(content=content, mode=0o600)
            staged_files[path] = _stage_aws_file(
                path,
                content,
                0o600,
            )

        try:
            for path, temporary_path in staged_files.items():
                if not _aws_file_matches(path, originals[path]):
                    raise OSError(
                        f"AWS file '{path}' changed before MPQP could update it."
                    )
                os.replace(temporary_path, path)
                replaced_files.append(path)
            if complete_update is not None:
                complete_update()
        except Exception as operation_error:
            rollback_errors: list[str] = []
            for path in reversed(replaced_files):
                try:
                    _restore_aws_file(
                        path,
                        originals[path],
                        staged_states[path],
                    )
                except Exception as rollback_error:
                    rollback_errors.append(f"{path}: {rollback_error}")

            if rollback_errors:
                raise OSError(
                    "Could not restore the AWS files after the update failed: "
                    + ", ".join(rollback_errors)
                ) from operation_error
            raise
    finally:
        for temporary_path in staged_files.values():
            temporary_path.unlink(missing_ok=True)


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


def _get_mpqp_aws_settings() -> dict[str, str]:
    """Return the AWS connection values currently saved in MPQP."""

    return {
        "BRAKET_AUTH_METHOD": get_env_variable("BRAKET_AUTH_METHOD"),
        "BRAKET_PROFILE": get_env_variable("BRAKET_PROFILE"),
        "BRAKET_CONFIGURED": get_env_variable("BRAKET_CONFIGURED"),
        "AWS_DEFAULT_REGION": get_env_variable("AWS_DEFAULT_REGION"),
    }


def _save_mpqp_aws_settings(
    authentication_method: str,
    profile_name: str,
    configured: bool,
    previous_settings: dict[str, str],
) -> None:
    """Save MPQP's AWS settings and restore their previous values on failure."""

    new_settings = {
        "BRAKET_AUTH_METHOD": authentication_method,
        "BRAKET_PROFILE": profile_name,
        "BRAKET_CONFIGURED": str(configured),
    }
    try:
        for key, value in new_settings.items():
            if save_env_variable(key, value) is False:
                raise OSError(f"Unable to save MPQP configuration value '{key}'.")
        if remove_env_variable("AWS_DEFAULT_REGION") is False:
            raise OSError(
                "Unable to remove MPQP configuration value 'AWS_DEFAULT_REGION'."
            )
    except Exception as settings_error:
        rollback_errors: list[str] = []
        for key, value in previous_settings.items():
            try:
                restored = (
                    save_env_variable(key, value) if value else remove_env_variable(key)
                )
                if restored is False:
                    rollback_errors.append(key)
            except Exception:
                rollback_errors.append(key)
        if rollback_errors:
            raise OSError(
                "Unable to restore MPQP AWS configuration values: "
                + ", ".join(rollback_errors)
            ) from settings_error
        raise


def _switch_aws_profile(
    profile_name: str,
    previous_settings: dict[str, str],
) -> None:
    """Select an AWS profile and remove old credentials managed by MPQP."""

    def save_profile_selection() -> None:
        _save_mpqp_aws_settings(
            authentication_method="PROFILE",
            profile_name=profile_name,
            configured=True,
            previous_settings=previous_settings,
        )

    previous_profile_is_managed = (
        previous_settings["BRAKET_AUTH_METHOD"] == "TEMPORARY"
        and previous_settings["BRAKET_PROFILE"] == "mpqp-temporary"
        and profile_name != "mpqp-temporary"
    )
    if not previous_profile_is_managed:
        save_profile_selection()
        return

    credentials_file, config_file = _get_aws_files()
    credentials = (
        _load_aws_file(credentials_file) if credentials_file.exists() else None
    )
    config = _load_aws_file(config_file) if config_file.exists() else None
    temporary_profile_removed = False
    if credentials is not None:
        temporary_profile_removed = credentials.remove_section("mpqp-temporary")
    if config is not None:
        temporary_profile_removed = (
            config.remove_section("profile mpqp-temporary") or temporary_profile_removed
        )

    if not temporary_profile_removed:
        save_profile_selection()
        return

    _save_aws_files(
        credentials_file,
        credentials,
        config_file,
        config,
        complete_update=save_profile_selection,
    )


def get_boto3_session(
    *,
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
) -> "Boto3Session":
    """Create a Boto3 session using MPQP's profile or Boto3's default AWS credentials."""

    import os

    import boto3

    selected_profile = profile_name or get_env_variable("BRAKET_PROFILE") or None

    # AWS_DEFAULT_REGION was previously set to an empty string when the local
    # AWS configuration was deleted. Remove that old value so Boto3 reads the
    # region from the selected AWS profile.
    if os.environ.get("AWS_DEFAULT_REGION") == "":
        os.environ.pop("AWS_DEFAULT_REGION")

    return boto3.Session(profile_name=selected_profile, region_name=region_name)


def setup_aws_braket_account() -> tuple[str, list[Any]]:
    """Set up or update the Amazon Braket connection.

    The user can select an existing AWS profile or provide short-term
    credentials.

    For an existing AWS profile, MPQP saves only its name and does not modify
    the profile. Short-term credentials entered in MPQP are stored in an AWS
    profile used only by MPQP.

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
    return "Amazon Braket setup ended.", []


def update_aws_credentials_file(
    profile_name: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: str,
    region: str,
    *,
    overwrite_existing: bool = False,
    complete_update: Optional[Callable[[], None]] = None,
) -> None:
    """Save AWS credentials and configuration under a profile name.

    The access keys are saved in the AWS credentials file, and the region is
    saved in the AWS config file. Existing profiles are left unchanged unless
    ``overwrite_existing`` is enabled. If ``complete_update`` fails, both AWS
    files are restored.
    """

    credentials_file, config_file = _get_aws_files()

    credentials = _load_aws_file(credentials_file)
    config = _load_aws_file(config_file)
    config_profile_name = (
        "default" if profile_name == "default" else f"profile {profile_name}"
    )

    profile_exists = credentials.has_section(profile_name) or config.has_section(
        config_profile_name
    )
    if profile_exists and not overwrite_existing:
        raise ValueError(f"AWS profile '{profile_name}' already exists.")

    if credentials.has_section(profile_name):
        credentials.remove_section(profile_name)
    credentials.add_section(profile_name)
    credentials[profile_name]["aws_access_key_id"] = access_key_id
    credentials[profile_name]["aws_secret_access_key"] = secret_access_key
    credentials[profile_name]["aws_session_token"] = session_token

    if not config.has_section(config_profile_name):
        config.add_section(config_profile_name)
    config[config_profile_name]["region"] = region

    _save_aws_files(
        credentials_file,
        credentials,
        config_file,
        config,
        complete_update=complete_update,
    )


def configure_account_profile(
    profile_name: Optional[str] = None,
) -> tuple[str, list[Any]]:
    """Configure MPQP to use an existing AWS profile."""

    if profile_name is None:
        profile_name = input("Enter AWS profile name: ").strip()
    if not profile_name:
        return "AWS profile name is required.", []

    previous_settings = _get_mpqp_aws_settings()

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
        session = get_boto3_session(
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

    try:
        _switch_aws_profile(profile_name, previous_settings)
    except Exception as error:
        return (
            "Unable to select the AWS profile without safely removing the "
            f"previous configuration.\nTrace: {error}",
            [],
        )

    return f"AWS profile '{profile_name}' configured successfully.", []


def configure_account_temporary_credentials() -> tuple[str, list[Any]]:
    """Configure short-term credentials in a profile managed by MPQP."""

    profile_name = "mpqp-temporary"
    previous_settings = _get_mpqp_aws_settings()
    configured_profile = previous_settings["BRAKET_PROFILE"]
    reuse_managed_profile = (
        previous_settings["BRAKET_AUTH_METHOD"] == "TEMPORARY"
        and configured_profile == profile_name
    )
    credentials_file, config_file = _get_aws_files()
    credentials = _load_aws_file(credentials_file)
    config = _load_aws_file(config_file)
    profile_exists = credentials.has_section(profile_name) or config.has_section(
        f"profile {profile_name}"
    )
    if profile_exists and not reuse_managed_profile:
        return (
            f"AWS profile '{profile_name}' already exists and was not created by "
            "MPQP. Rename or remove it before configuring short-term credentials.",
            [],
        )

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

    def save_temporary_selection() -> None:
        _save_mpqp_aws_settings(
            authentication_method="TEMPORARY",
            profile_name=profile_name,
            configured=True,
            previous_settings=previous_settings,
        )

    try:
        update_aws_credentials_file(
            profile_name=profile_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            region=region,
            overwrite_existing=reuse_managed_profile,
            complete_update=save_temporary_selection,
        )
    except Exception as error:
        return f"Unable to save the AWS profile.\nTrace: {error}", []

    return "AWS temporary credentials configured successfully.", []


def delete_aws_braket_account() -> tuple[str, list[Any]]:
    """Remove the Amazon Braket configuration saved in MPQP.

    The local profile created by MPQP for temporary credentials is deleted.
    If the user selected an existing AWS profile, only its name is removed from
    MPQP's configuration. The AWS profile and account are kept.

    Returns:
        A tuple containing a message describing the result and an empty list.
    """

    previous_settings = _get_mpqp_aws_settings()
    authentication_method = previous_settings["BRAKET_AUTH_METHOD"]
    configured_profile = previous_settings["BRAKET_PROFILE"]
    profile_name = configured_profile or "default"
    profile_is_managed = (
        authentication_method == "TEMPORARY" and configured_profile == "mpqp-temporary"
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

    def clear_mpqp_settings() -> None:
        _save_mpqp_aws_settings(
            authentication_method="",
            profile_name="",
            configured=False,
            previous_settings=previous_settings,
        )

    try:
        if profile_is_managed:
            credentials_file, config_file = _get_aws_files()
            config_profile = f"profile {profile_name}"
            credentials = (
                _load_aws_file(credentials_file) if credentials_file.exists() else None
            )
            config = _load_aws_file(config_file) if config_file.exists() else None
            profile_removed = False
            if credentials is not None:
                profile_removed = credentials.remove_section(profile_name)
            if config is not None:
                profile_removed = (
                    config.remove_section(config_profile) or profile_removed
                )
            if profile_removed:
                _save_aws_files(
                    credentials_file,
                    credentials,
                    config_file,
                    config,
                    complete_update=clear_mpqp_settings,
                )
            else:
                clear_mpqp_settings()
        else:
            clear_mpqp_settings()
    except Exception as error:
        action = (
            "delete the AWS profile"
            if profile_is_managed
            else "remove the Amazon Braket configuration"
        )
        return f"Unable to {action}.\nTrace: {error}", []

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


def get_aws_braket_account_info() -> str:
    """Retrieve information about the AWS configuration used for Amazon Braket.

    The selected AWS credentials are validated before their information is
    returned.

    Returns:
        A formatted string containing the credential source, profile details
        when applicable, region, and connection status.

    Raises:
        AWSBraketRemoteExecutionError: If Amazon Braket is not configured or the
            AWS credentials cannot be validated.

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
        session = get_boto3_session()
        session.client("sts").get_caller_identity()
    except Exception as error:
        raise AWSBraketRemoteExecutionError(
            "Error when validating the configured AWS credentials. Possibly no AWS "
            "Braket account is configured.\nTrace: " + str(error)
        ) from error

    profile_name = get_env_variable("BRAKET_PROFILE")
    stored_auth_method = get_env_variable("BRAKET_AUTH_METHOD")
    if profile_name:
        credential_source = {
            "PROFILE": "Existing AWS profile",
            "TEMPORARY": "Short-term credentials",
        }.get(stored_auth_method, "Unknown")
    else:
        credential_source = "Other AWS credentials"

    information = [f"    Credential source: {credential_source}"]
    if profile_name:
        information.append(f"    Profile name: '{profile_name}'")
    if stored_auth_method == "PROFILE" and profile_name:
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

    from botocore.exceptions import BotoCoreError, ClientError, NoRegionError
    from braket.aws import AwsDevice, AwsSession

    import mpqp

    try:
        boto3_session = get_boto3_session()
        if device.is_simulator() and not boto3_session.region_name:
            raise NoRegionError()
        region = device.get_region(boto3_session.region_name)

        if boto3_session.region_name != region:
            boto3_session = get_boto3_session(region_name=region)
        braket_client = boto3_session.client("braket", region_name=region)

        aws_session = AwsSession(
            boto_session=boto3_session,
            braket_client=braket_client,
        )
        aws_session.add_braket_user_agent(
            user_agent="APN/1.0 ColibriTD/1.0 MPQP/" + mpqp.__version__
        )
        braket_device = AwsDevice(device.get_arn(), aws_session=aws_session)

    except NoRegionError as error:
        raise AWSBraketRemoteExecutionError(
            "Failed to find the region associated with your AWS credentials. Make "
            "sure your AWS account is correctly configured using "
            "'setup_connections'."
            "\nTrace: " + str(error)
        ) from error
    except (BotoCoreError, ClientError, ValueError) as error:
        raise AWSBraketRemoteExecutionError(
            "Failed to retrieve the remote AWS device. Verify that the selected "
            "AWS profile and region are valid and that the device is accessible."
            "\nTrace: " + str(error)
        ) from error

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

    if get_env_variable("BRAKET_CONFIGURED") == "True":
        boto3_session = get_boto3_session()
        braket_client = boto3_session.client("braket")
        pages = braket_client.get_paginator("search_quantum_tasks").paginate(filters=[])
        return [
            task["quantumTaskArn"]
            for page in pages
            for task in page.get("quantumTasks", [])
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
