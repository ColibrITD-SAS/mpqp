# pyright: reportPrivateUsage=false

import configparser
import os
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

import mpqp.environment.env_manager as env_manager
import mpqp.execution.connection.aws_connection as aws_connection
from mpqp.execution.devices import AWSDevice

pytestmark = pytest.mark.provider("braket")

_MPQP_AWS_KEYS = (
    "BRAKET_AUTH_METHOD",
    "BRAKET_CONFIGURED",
    "BRAKET_PROFILE",
    "AWS_DEFAULT_REGION",
)


@pytest.fixture
def mpqp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_manager, "MPQP_ENV", tmp_path / ".mpqp" / ".env")
    for key in _MPQP_AWS_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def aws_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    credentials_file = tmp_path / "credentials"
    config_file = tmp_path / "config"
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    return credentials_file, config_file


def _save_mpqp_aws_config(
    *,
    authentication_method: str,
    profile_name: str,
    configured: bool = True,
) -> None:
    env_manager.save_env_variable("BRAKET_AUTH_METHOD", authentication_method)
    env_manager.save_env_variable("BRAKET_PROFILE", profile_name)
    env_manager.save_env_variable("BRAKET_CONFIGURED", str(configured))


def _get_mpqp_aws_config() -> dict[str, str]:
    return {key: env_manager.get_env_variable(key) for key in _MPQP_AWS_KEYS}


def test_update_aws_profile_preserves_existing_profiles(
    aws_files: tuple[Path, Path],
):
    credentials_file, config_file = aws_files
    credentials_file.write_text(
        "[user-profile]\naws_access_key_id = user-key\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "[profile user-profile]\n"
        "region = us-east-1\n"
        "output = text\n"
        "[profile mpqp-temporary]\n"
        "output = json\n",
        encoding="utf-8",
    )

    aws_connection.update_aws_credentials_file(
        profile_name="mpqp-temporary",
        access_key_id="temporary-key",
        secret_access_key="temporary-secret",
        session_token="temporary-token",
        region="eu-west-2",
        overwrite_existing=True,
    )

    credentials = configparser.ConfigParser()
    credentials.read(credentials_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    assert dict(credentials["mpqp-temporary"]) == {
        "aws_access_key_id": "temporary-key",
        "aws_secret_access_key": "temporary-secret",
        "aws_session_token": "temporary-token",
    }
    assert dict(credentials["user-profile"]) == {
        "aws_access_key_id": "user-key",
    }
    assert config["profile mpqp-temporary"]["region"] == "eu-west-2"
    assert config["profile mpqp-temporary"]["output"] == "json"
    assert dict(config["profile user-profile"]) == {
        "region": "us-east-1",
        "output": "text",
    }
    if os.name != "nt":
        assert credentials_file.stat().st_mode & 0o777 == 0o600
        assert config_file.stat().st_mode & 0o777 == 0o600


def test_boto3_session_uses_environment_credentials(
    mpqp_config: None,
    aws_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "environment-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")

    session = aws_connection.get_boto3_session()
    credentials = session.get_credentials()

    assert credentials is not None
    assert credentials.method == "env"
    assert credentials.access_key == "environment-key"
    assert session.region_name == "eu-west-1"


def test_boto3_session_uses_selected_profile(
    mpqp_config: None,
    aws_files: tuple[Path, Path],
):
    credentials_file, config_file = aws_files
    credentials_file.write_text(
        "[team-profile]\n"
        "aws_access_key_id = team-key\n"
        "aws_secret_access_key = team-secret\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "[profile team-profile]\nregion = eu-west-1\n",
        encoding="utf-8",
    )
    _save_mpqp_aws_config(
        authentication_method="PROFILE",
        profile_name="team-profile",
    )

    session = aws_connection.get_boto3_session()
    credentials = session.get_credentials()

    assert session.profile_name == "team-profile"
    assert credentials is not None
    assert credentials.access_key == "team-key"


def test_configure_short_term_credentials(
    mpqp_config: None,
    aws_files: tuple[Path, Path],
):
    import boto3

    credentials_file, config_file = aws_files
    boto3_session = Mock()
    with (
        patch("builtins.input", side_effect=["temporary-key", "eu-west-2"]),
        patch(
            "getpass.getpass",
            side_effect=["temporary-secret", "temporary-token"],
        ),
        patch.object(boto3, "Session", return_value=boto3_session),
    ):
        setup_response = aws_connection.configure_account_temporary_credentials()

    assert setup_response == ("AWS temporary credentials configured successfully.", [])
    boto3_session.client.assert_called_once_with("sts")
    boto3_session.client.return_value.get_caller_identity.assert_called_once_with()
    credentials = configparser.ConfigParser()
    credentials.read(credentials_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    assert dict(credentials["mpqp-temporary"]) == {
        "aws_access_key_id": "temporary-key",
        "aws_secret_access_key": "temporary-secret",
        "aws_session_token": "temporary-token",
    }
    assert config["profile mpqp-temporary"]["region"] == "eu-west-2"
    assert _get_mpqp_aws_config() == {
        "BRAKET_AUTH_METHOD": "TEMPORARY",
        "BRAKET_CONFIGURED": "True",
        "BRAKET_PROFILE": "mpqp-temporary",
        "AWS_DEFAULT_REGION": "",
    }


def test_short_term_credentials_do_not_overwrite_user_profile(
    mpqp_config: None,
    aws_files: tuple[Path, Path],
):
    credentials_file, config_file = aws_files
    credentials_file.write_text(
        "[mpqp-temporary]\n"
        "aws_access_key_id = existing-key\n"
        "aws_secret_access_key = existing-secret\n"
        "aws_session_token = existing-token\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "[profile mpqp-temporary]\nregion = us-east-1\n",
        encoding="utf-8",
    )
    message, _ = aws_connection.configure_account_temporary_credentials()

    credentials = configparser.ConfigParser()
    credentials.read(credentials_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    assert "was not created by MPQP" in message
    assert credentials["mpqp-temporary"]["aws_access_key_id"] == "existing-key"
    assert credentials["mpqp-temporary"]["aws_secret_access_key"] == "existing-secret"
    assert credentials["mpqp-temporary"]["aws_session_token"] == "existing-token"
    assert config["profile mpqp-temporary"]["region"] == "us-east-1"


def test_short_term_credentials_update_mpqp_profile(
    mpqp_config: None,
    aws_files: tuple[Path, Path],
):
    import boto3

    _save_mpqp_aws_config(
        authentication_method="TEMPORARY",
        profile_name="mpqp-temporary",
    )
    credentials_file, config_file = aws_files
    credentials_file.write_text(
        "[mpqp-temporary]\n"
        "aws_access_key_id = existing-key\n"
        "aws_secret_access_key = existing-secret\n"
        "aws_session_token = existing-token\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "[profile mpqp-temporary]\nregion = us-east-1\n",
        encoding="utf-8",
    )
    boto3_session = Mock()
    with (
        patch("builtins.input", side_effect=["new-key", "eu-west-2"]),
        patch("getpass.getpass", side_effect=["new-secret", "new-token"]),
        patch.object(boto3, "Session", return_value=boto3_session),
    ):
        setup_response = aws_connection.configure_account_temporary_credentials()

    credentials = configparser.ConfigParser()
    credentials.read(credentials_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    assert setup_response == ("AWS temporary credentials configured successfully.", [])
    assert credentials["mpqp-temporary"]["aws_access_key_id"] == "new-key"
    assert credentials["mpqp-temporary"]["aws_secret_access_key"] == "new-secret"
    assert credentials["mpqp-temporary"]["aws_session_token"] == "new-token"
    assert config["profile mpqp-temporary"]["region"] == "eu-west-2"


def test_select_profile_removes_short_term_credentials(
    mpqp_config: None,
    aws_files: tuple[Path, Path],
):
    import boto3

    _save_mpqp_aws_config(
        authentication_method="TEMPORARY",
        profile_name="mpqp-temporary",
    )
    credentials_file, config_file = aws_files
    credentials_file.write_text(
        "[user-profile]\n"
        "aws_access_key_id = user-key\n"
        "[mpqp-temporary]\n"
        "aws_access_key_id = temporary-key\n"
        "aws_secret_access_key = temporary-secret\n"
        "aws_session_token = temporary-token\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "[profile mpqp-temporary]\nregion = us-east-1\n"
        "[profile team-profile]\nregion = eu-west-1\n",
        encoding="utf-8",
    )
    boto3_session = Mock(region_name="eu-west-1")

    with patch.object(boto3, "Session", return_value=boto3_session):
        setup_response = aws_connection.configure_account_profile("team-profile")

    credentials = configparser.ConfigParser()
    credentials.read(credentials_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    assert setup_response == ("AWS profile 'team-profile' configured successfully.", [])
    assert credentials["user-profile"]["aws_access_key_id"] == "user-key"
    assert not credentials.has_section("mpqp-temporary")
    assert config.has_section("profile team-profile")
    assert not config.has_section("profile mpqp-temporary")
    assert _get_mpqp_aws_config() == {
        "BRAKET_AUTH_METHOD": "PROFILE",
        "BRAKET_CONFIGURED": "True",
        "BRAKET_PROFILE": "team-profile",
        "AWS_DEFAULT_REGION": "",
    }


def test_delete_short_term_credentials_preserves_other_profiles(
    mpqp_config: None,
    aws_files: tuple[Path, Path],
):
    _save_mpqp_aws_config(
        authentication_method="TEMPORARY",
        profile_name="mpqp-temporary",
    )
    credentials_file, config_file = aws_files
    credentials_file.write_text(
        "[user-profile]\naws_access_key_id = user-key\n"
        "[mpqp-temporary]\naws_access_key_id = temporary-key\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "[profile user-profile]\nregion = us-east-1\n"
        "[profile mpqp-temporary]\nregion = eu-west-2\n",
        encoding="utf-8",
    )

    with patch("builtins.input", return_value="y"):
        message, _ = aws_connection.delete_aws_braket_account()

    credentials = configparser.ConfigParser()
    credentials.read(credentials_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    assert message == (
        "AWS profile 'mpqp-temporary' deleted and its MPQP configuration removed."
    )
    assert credentials["user-profile"]["aws_access_key_id"] == "user-key"
    assert not credentials.has_section("mpqp-temporary")
    assert config["profile user-profile"]["region"] == "us-east-1"
    assert not config.has_section("profile mpqp-temporary")
    assert env_manager.get_env_variable("BRAKET_CONFIGURED") == "False"


def test_remove_connection_preserves_aws_profile(
    mpqp_config: None,
    aws_files: tuple[Path, Path],
):
    _save_mpqp_aws_config(
        authentication_method="PROFILE",
        profile_name="team-profile",
    )
    _, config_file = aws_files
    aws_config_before_removal = (
        "[profile team-profile]\n"
        "sso_session = team-session\n"
        "region = us-east-1\n"
        "\n"
        "[sso-session team-session]\n"
        "sso_start_url = https://example.awsapps.com/start\n"
        "sso_region = us-east-1\n"
    )
    config_file.write_text(aws_config_before_removal, encoding="utf-8")

    with patch("builtins.input", return_value="y"):
        message, _ = aws_connection.delete_aws_braket_account()

    assert message == (
        "Amazon Braket configuration removed from MPQP. AWS profile 'team-profile' "
        "remains available."
    )
    assert config_file.read_text(encoding="utf-8") == aws_config_before_removal
    assert env_manager.get_env_variable("BRAKET_CONFIGURED") == "False"


def test_remote_simulator_uses_profile_region(
    mpqp_config: None,
):
    import boto3

    _save_mpqp_aws_config(
        authentication_method="PROFILE",
        profile_name="team-profile",
    )
    boto3_session = Mock(region_name="us-west-2")
    braket_client = boto3_session.client.return_value
    braket_session = Mock()
    braket_device = Mock()

    with (
        patch.object(
            boto3,
            "Session",
            return_value=boto3_session,
        ) as create_boto3_session,
        patch(
            "braket.aws.AwsSession",
            return_value=braket_session,
        ) as create_braket_session,
        patch("braket.aws.AwsDevice", return_value=braket_device),
    ):
        selected_device = aws_connection.get_braket_device(
            AWSDevice.BRAKET_SV1_SIMULATOR
        )

    assert selected_device is braket_device
    create_boto3_session.assert_called_once_with(
        profile_name="team-profile",
        region_name=None,
    )
    boto3_session.client.assert_called_once_with("braket", region_name="us-west-2")
    create_braket_session.assert_called_once_with(
        boto_session=boto3_session,
        braket_client=braket_client,
    )


def test_qpu_uses_device_region(mpqp_config: None):
    import boto3

    _save_mpqp_aws_config(
        authentication_method="PROFILE",
        profile_name="team-profile",
    )
    profile_boto3_session = Mock(region_name="us-east-1")
    device_boto3_session = Mock(region_name="us-west-1")
    braket_client = device_boto3_session.client.return_value
    braket_session = Mock()
    braket_device = Mock()

    with (
        patch.object(
            boto3,
            "Session",
            side_effect=[profile_boto3_session, device_boto3_session],
        ) as create_boto3_session,
        patch(
            "braket.aws.AwsSession",
            return_value=braket_session,
        ) as create_braket_session,
        patch("braket.aws.AwsDevice", return_value=braket_device),
    ):
        selected_device = aws_connection.get_braket_device(AWSDevice.RIGETTI_ANKAA_3)

    assert selected_device is braket_device
    assert create_boto3_session.call_args_list == [
        call(profile_name="team-profile", region_name=None),
        call(profile_name="team-profile", region_name="us-west-1"),
    ]
    device_boto3_session.client.assert_called_once_with(
        "braket",
        region_name="us-west-1",
    )
    create_braket_session.assert_called_once_with(
        boto_session=device_boto3_session,
        braket_client=braket_client,
    )
