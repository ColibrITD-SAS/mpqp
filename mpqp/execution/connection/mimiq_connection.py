"""Explicit account setup and noninteractive connections to the MIMIQ cloud."""

from __future__ import annotations

from functools import lru_cache, partial
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Protocol, cast

from mpqp.environment.env_manager import get_env_variable, save_env_variable

if TYPE_CHECKING:
    from mimiqcircuits import Circuit, QCSResults


class _MimiqRequestInfo(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def status(self) -> str: ...


class _MimiqConnection(Protocol):
    """The SDK's dynamically forwarded cloud methods used by this provider."""

    def connect(self) -> object: ...
    def savetoken(self, filepath: str) -> object: ...
    def loadtoken(self, filepath: str) -> object: ...
    def submit(self, circuit: Circuit, **kwargs: Any) -> str: ...
    def get_result(self, job_id: str) -> QCSResults: ...
    def get_inputs(self, job_id: str) -> tuple[list[Circuit], dict[str, Any]]: ...
    def requestInfo(self, job_id: str) -> _MimiqRequestInfo: ...
    def requests(self, **kwargs: Any) -> Iterable[_MimiqRequestInfo]: ...


def _token_path() -> Path:
    configured = get_env_variable('MIMIQ_TOKEN_FILE')
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / '.mpqp' / 'mimiq-token.json'
    )


def _new_mimiq_connection(platform: str) -> _MimiqConnection:
    if platform not in ('v2', 'legacy'):
        raise ValueError(
            'Unknown MIMIQ platform. Choose "v2" or "legacy" during setup.'
        )
    from mimiqcircuits import QPERFECT_CLOUD, QPERFECT_CLOUD_V2, MimiqConnection

    url = QPERFECT_CLOUD_V2 if platform == 'v2' else QPERFECT_CLOUD
    return cast(_MimiqConnection, MimiqConnection(url))


def setup_mimiq_account(platform: Literal['v2', 'legacy'] = 'v2') -> None:
    """Open browser login and save the selected MIMIQ account for future sessions.

    V2 and the original platform have separate accounts and job histories.
    New setups use V2 by default. Accounts saved before platform selection was
    introduced continue to use the original service until reconfigured.

    The token grants access to the account. Keep it private. Set
    ``MIMIQ_TOKEN_FILE`` to choose its location; the default is
    ``~/.mpqp/mimiq-token.json``. Execution never opens a login browser itself.
    You can also run ``setup_connections`` and choose MIMIQ to configure it.

    Args:
        platform: Cloud platform to authenticate against: ``"v2"`` or ``"legacy"``.

    Raises:
        ValueError: If the platform is unrecognized.
        RuntimeError: If the platform or account configuration cannot be saved.
    """
    connection = _new_mimiq_connection(platform)
    connection.connect()
    token_path = _token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    connection.savetoken(str(token_path))
    get_mimiq_connection.cache_clear()
    if not save_env_variable('MIMIQ_PLATFORM', platform) or not save_env_variable(
        'MIMIQ_CONFIGURED', 'True'
    ):
        raise RuntimeError('Unable to save the MIMIQ account configuration.')


def config_mimiq_account() -> tuple[str, list[object]]:
    """Choose a MIMIQ platform and configure it through browser authentication."""
    from mpqp.tools.choice_tree import AnswerNode, QuestionNode, run_choice_tree

    def configure(platform: Literal['v2', 'legacy']) -> tuple[str, list[object]]:
        try:
            setup_mimiq_account(platform)
        except Exception as error:
            return f'MIMIQ configuration failed: {error}', []
        return 'MIMIQ account correctly configured.', []

    question = QuestionNode(
        'Choose MIMIQ cloud platform:',
        [
            AnswerNode('MIMIQ V2 (Quantum Hive)', partial(configure, 'v2')),
            AnswerNode('Original MIMIQ', partial(configure, 'legacy')),
        ],
    )
    for answer in question.answers:
        answer.next_question = None
    run_choice_tree(question)
    return '', []


def get_mimiq_account_info() -> str:
    """Describe the saved platform and token availability without logging in.

    This reports local configuration only; it does not verify authentication
    with the cloud service or display the token.
    """
    if get_env_variable('MIMIQ_CONFIGURED') != 'True':
        return 'Account not configured'
    platform = get_env_variable('MIMIQ_PLATFORM') or 'legacy'
    labels = {'v2': 'MIMIQ V2 (Quantum Hive)', 'legacy': 'Original MIMIQ'}
    if platform not in labels:
        return 'Unknown MIMIQ platform. Configure MIMIQ again.'
    if not _token_path().is_file():
        return 'Saved MIMIQ token not found. Configure MIMIQ again.'
    return f'Platform: {labels[platform]}\nSaved token available (authentication not checked).'


def delete_mimiq_account() -> tuple[str, list[object]]:
    """Confirm removal of the saved local MIMIQ token and configuration.

    This clears MPQP's cached connection. It does not delete the cloud account,
    revoke tokens remotely, or remove jobs from the service.
    """
    decision = input('Delete the saved local MIMIQ configuration? [y/N] ')
    if decision.strip().lower() != 'y':
        return 'Canceled.', []
    try:
        _token_path().unlink(missing_ok=True)
    except OSError:
        return 'Unable to remove the saved MIMIQ token.', []
    get_mimiq_connection.cache_clear()
    for key, value in (
        ('MIMIQ_CONFIGURED', 'False'),
        ('MIMIQ_PLATFORM', ''),
        ('MIMIQ_TOKEN_FILE', ''),
    ):
        if not save_env_variable(key, value):
            return 'Unable to clear the MIMIQ configuration.', []
    return 'Local MIMIQ configuration deleted.', []


@lru_cache(maxsize=1)
def get_mimiq_connection() -> _MimiqConnection:
    """Load a saved cloud account without prompting or opening a browser.

    Returns:
        An authenticated SDK connection, reused within the Python session.

    Raises:
        RuntimeError: If no token is available or authentication fails.
    """
    token_path = _token_path()
    if not token_path.is_file():
        raise RuntimeError(
            'No MIMIQ account configured. Call setup_mimiq_account() first.'
        )
    connection = _new_mimiq_connection(get_env_variable('MIMIQ_PLATFORM') or 'legacy')
    try:
        connection.loadtoken(str(token_path))
    except Exception as error:
        raise RuntimeError(
            'MIMIQ authentication failed. Run setup_mimiq_account() again.'
        ) from error
    return connection


def get_all_job_ids() -> list[str]:
    """Return execution IDs for a configured MIMIQ account, or an empty list."""
    if (
        get_env_variable('MIMIQ_CONFIGURED') != 'True'
        or find_spec("mimiqcircuits") is None
    ):
        return []
    return [info.id for info in get_mimiq_connection().requests()]
