"""Explicit account setup and noninteractive connections to the MIMIQ cloud."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol, cast

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


def setup_mimiq_account() -> None:
    """Open MIMIQ's browser login and save its token outside the repository.

    The token grants access to the account. Keep it private. Set
    ``MIMIQ_TOKEN_FILE`` to choose its location; the default is
    ``~/.mpqp/mimiq-token.json``. Execution never opens a login browser itself.
    """
    from mimiqcircuits import MimiqConnection

    connection = MimiqConnection()
    connection.connect()
    token_path = _token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    connection.savetoken(str(token_path))
    save_env_variable('MIMIQ_CONFIGURED', 'True')
    get_mimiq_connection.cache_clear()


@lru_cache(maxsize=1)
def get_mimiq_connection() -> _MimiqConnection:
    """Load a saved cloud account without prompting or opening a browser.

    Returns:
        An authenticated SDK connection, reused within the Python session.

    Raises:
        RuntimeError: If no token is available or authentication fails.
    """
    from mimiqcircuits import MimiqConnection

    token_path = _token_path()
    if not token_path.is_file():
        raise RuntimeError(
            'No MIMIQ account configured. Call setup_mimiq_account() first.'
        )
    connection = MimiqConnection()
    try:
        connection.loadtoken(str(token_path))
    except Exception as error:
        raise RuntimeError(
            'MIMIQ authentication failed. Run setup_mimiq_account() again.'
        ) from error
    return cast(_MimiqConnection, connection)


def get_all_job_ids() -> list[str]:
    """Return execution IDs for a configured MIMIQ account, or an empty list."""
    if (
        get_env_variable('MIMIQ_CONFIGURED') != 'True'
        or find_spec("mimiqcircuits") is None
    ):
        return []
    return [info.id for info in get_mimiq_connection().requests()]
