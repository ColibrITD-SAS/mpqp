import os
from pathlib import Path

import pytest

from mpqp.environment import env_manager


def test_remove_env_variable_clears_unsaved_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_file = tmp_path / ".env"
    config_file.touch()
    monkeypatch.setattr(env_manager, "MPQP_ENV", config_file)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    assert env_manager.remove_env_variable("AWS_DEFAULT_REGION")
    assert "AWS_DEFAULT_REGION" not in os.environ
