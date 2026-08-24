# pyright: reportPrivateUsage=false

import sys
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import mpqp.execution.connection.quantinuum_connection as qnx_connection

pytestmark = pytest.mark.provider("quantinuum")


@pytest.fixture
def quantinuum_config() -> Iterator[dict[str, str]]:
    values = {
        "QUANTINUUM_CONFIGURED": "True",
        "QUANTINUUM_PROJECT_NAME": "mpqp-project",
    }

    def save_value(key: str, value: str) -> bool:
        values[key] = value
        return True

    with (
        patch.object(qnx_connection, "get_env_variable", side_effect=values.get),
        patch.object(qnx_connection, "save_env_variable", side_effect=save_value),
    ):
        yield values


def _project_ref(name: str = "mpqp-project") -> SimpleNamespace:
    return SimpleNamespace(annotations=SimpleNamespace(name=name))


@pytest.mark.parametrize("configured", [False, True])
def test_setup_quantinuum_account_login(
    quantinuum_config: dict[str, str],
    configured: bool,
):
    import qnexus as qnx

    quantinuum_config["QUANTINUUM_CONFIGURED"] = str(configured)
    qnx_login = Mock()
    with (
        patch("builtins.input", return_value="y"),
        patch.object(qnx, "login", qnx_login),
        patch.object(
            qnx_connection,
            "_setup_quantinuum_project",
            return_value=("configured", []),
        ),
    ):
        result = qnx_connection.setup_quantinuum_account()

    assert result == ("configured", [])
    qnx_login.assert_called_once_with(force=configured)


@pytest.mark.parametrize("create_new", [False, True])
def test_configure_quantinuum_project(
    quantinuum_config: dict[str, str],
    create_new: bool,
):
    import qnexus as qnx

    quantinuum_config["QUANTINUUM_CONFIGURED"] = "False"
    quantinuum_config["QUANTINUUM_PROJECT_NAME"] = ""
    project = _project_ref()
    get_project = Mock(return_value=project)
    create_project = Mock(return_value=project)
    set_active_project = Mock()

    with (
        patch("builtins.input", return_value="mpqp-project"),
        patch.object(qnx.projects, "get", get_project),
        patch.object(qnx.projects, "create", create_project),
        patch.object(qnx.context, "set_active_project", set_active_project),
    ):
        message, _ = qnx_connection._configure_quantinuum_project(create_new)

    assert message == "Quantinuum Nexus account correctly configured."
    if create_new:
        create_project.assert_called_once_with(name="mpqp-project")
        get_project.assert_not_called()
    else:
        get_project.assert_called_once_with(name="mpqp-project")
        create_project.assert_not_called()
    set_active_project.assert_called_once_with(project)
    assert quantinuum_config == {
        "QUANTINUUM_CONFIGURED": "True",
        "QUANTINUUM_PROJECT_NAME": "mpqp-project",
    }


@pytest.mark.usefixtures("quantinuum_config")
@pytest.mark.parametrize("can_access_project", [True, False])
def test_check_quantinuum_connection(
    can_access_project: bool,
):
    import qnexus as qnx

    get_project = Mock(return_value=_project_ref())
    if not can_access_project:
        get_project.side_effect = RuntimeError("Project cannot be accessed.")

    with patch.object(qnx.projects, "get", get_project):
        result = qnx_connection.check_quantinuum_connection()

    assert result is can_access_project
    get_project.assert_called_once_with(name="mpqp-project")


@pytest.mark.usefixtures("quantinuum_config")
def test_activate_quantinuum_project():
    import qnexus as qnx

    project = _project_ref()
    get_project = Mock(return_value=project)
    set_active_project = Mock()
    with (
        patch.object(qnx.context, "get_active_project", return_value=None),
        patch.object(qnx.projects, "get", get_project),
        patch.object(qnx.context, "set_active_project", set_active_project),
    ):
        qnx_connection._activate_quantinuum_project()

    get_project.assert_called_once_with(name="mpqp-project")
    set_active_project.assert_called_once_with(project)


@pytest.mark.usefixtures("quantinuum_config")
def test_project_activation_fails_if_inaccessible():
    import qnexus as qnx

    with (
        patch.object(qnx.context, "get_active_project", return_value=None),
        patch.object(
            qnx.projects,
            "get",
            side_effect=RuntimeError("Project cannot be accessed."),
        ),
    ):
        with pytest.raises(RuntimeError, match="Unable to access.*'mpqp-project'"):
            qnx_connection._activate_quantinuum_project()


@pytest.mark.usefixtures("quantinuum_config")
def test_get_all_job_ids():
    import qnexus as qnx

    get_all_jobs = Mock(
        return_value=[SimpleNamespace(id="job-id-1"), SimpleNamespace(id="job-id-2")]
    )
    with (
        patch.object(qnx_connection, "_activate_quantinuum_project"),
        patch.object(qnx.jobs, "get_all", get_all_jobs),
    ):
        result = qnx_connection.get_all_job_ids()

    assert result == ["job-id-1", "job-id-2"]
    get_all_jobs.assert_called_once_with()


def test_delete_quantinuum_account(quantinuum_config: dict[str, str]):
    import qnexus as qnx

    qnx_logout = Mock()
    with (
        patch("builtins.input", side_effect=["y", ""]),
        patch.object(qnx, "logout", qnx_logout),
    ):
        result = qnx_connection.delete_quantinuum_account()

    assert result == ("Quantinuum account deleted.", [])
    qnx_logout.assert_called_once_with()
    assert quantinuum_config == {
        "QUANTINUUM_CONFIGURED": "False",
        "QUANTINUUM_PROJECT_NAME": "",
    }


@pytest.mark.skipif(
    "--long" not in sys.argv,
    reason="requires a configured Quantinuum Nexus account and project",
)
def test_remote_quantinuum_connection():
    assert qnx_connection.check_quantinuum_connection()
