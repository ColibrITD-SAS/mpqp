"""MIMIQ platform selection and local credential lifecycle, without network calls."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest

from mpqp.execution.connection import mimiq_connection as connection
from mpqp.tools.choice_tree import QuestionNode

mc = pytest.importorskip('mimiqcircuits')
pytestmark = pytest.mark.provider('mimiq')


@pytest.fixture
def mimiq_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[dict[str, str]]:
    values = {
        'MIMIQ_TOKEN_FILE': str(tmp_path / 'mimiq-token.json'),
        'IBM_CONFIGURED': 'True',
    }

    def save(key: str, value: str) -> bool:
        values[key] = value
        return True

    monkeypatch.setattr(connection.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(connection, 'get_env_variable', lambda key: values.get(key, ''))
    monkeypatch.setattr(connection, 'save_env_variable', save)
    connection.get_mimiq_connection.cache_clear()
    yield values
    connection.get_mimiq_connection.cache_clear()


@pytest.mark.parametrize('platform', ['v2', 'legacy'])
def test_mimiq_setup_and_token_reload_use_same_platform(
    monkeypatch: pytest.MonkeyPatch, mimiq_config: dict[str, str], platform: str
):
    token_path = Path(mimiq_config['MIMIQ_TOKEN_FILE'])
    factory = Mock()
    factory.return_value.savetoken.side_effect = lambda filename: Path(
        filename
    ).write_text('private-test-token', encoding='utf-8')
    monkeypatch.setattr(mc, 'MimiqConnection', factory)
    if platform == 'v2':
        connection.setup_mimiq_account()
    else:
        connection.setup_mimiq_account('legacy')
    endpoint = mc.QPERFECT_CLOUD_V2 if platform == 'v2' else mc.QPERFECT_CLOUD
    factory.assert_called_once_with(endpoint)
    factory.return_value.connect.assert_called_once_with()
    assert mimiq_config['MIMIQ_PLATFORM'] == platform
    assert mimiq_config['MIMIQ_CONFIGURED'] == 'True'

    factory.reset_mock()
    client = connection.get_mimiq_connection()
    assert connection.get_mimiq_connection() is client
    factory.assert_called_once_with(endpoint)
    factory.return_value.loadtoken.assert_called_once_with(str(token_path))
    factory.return_value.connect.assert_not_called()
    connection.get_mimiq_connection.cache_clear()
    connection.get_mimiq_connection()
    assert factory.call_count == 2


def test_mimiq_existing_token_defaults_to_original_platform(
    monkeypatch: pytest.MonkeyPatch, mimiq_config: dict[str, str]
):
    Path(mimiq_config['MIMIQ_TOKEN_FILE']).write_text('{}', encoding='utf-8')
    factory = Mock()
    monkeypatch.setattr(mc, 'MimiqConnection', factory)
    connection.get_mimiq_connection()
    factory.assert_called_once_with(mc.QPERFECT_CLOUD)


@pytest.mark.parametrize('operation', ['connect', 'savetoken'])
def test_mimiq_failed_setup_does_not_mark_account_configured(
    monkeypatch: pytest.MonkeyPatch, mimiq_config: dict[str, str], operation: str
):
    factory = Mock()
    getattr(factory.return_value, operation).side_effect = RuntimeError('Failed login')
    monkeypatch.setattr(mc, 'MimiqConnection', factory)
    with pytest.raises(RuntimeError, match='Failed login'):
        connection.setup_mimiq_account()
    assert 'MIMIQ_CONFIGURED' not in mimiq_config
    assert 'MIMIQ_PLATFORM' not in mimiq_config


def test_mimiq_unknown_saved_platform_is_rejected_before_authentication(
    monkeypatch: pytest.MonkeyPatch, mimiq_config: dict[str, str]
):
    mimiq_config['MIMIQ_PLATFORM'] = 'unknown'
    Path(mimiq_config['MIMIQ_TOKEN_FILE']).write_text('{}', encoding='utf-8')
    factory = Mock()
    monkeypatch.setattr(mc, 'MimiqConnection', factory)
    with pytest.raises(ValueError, match='Unknown MIMIQ platform'):
        connection.get_mimiq_connection()
    factory.assert_not_called()


def test_mimiq_recap_is_offline_and_never_displays_token(
    monkeypatch: pytest.MonkeyPatch, mimiq_config: dict[str, str]
):
    mimiq_config.update(MIMIQ_PLATFORM='v2', MIMIQ_CONFIGURED='True')
    Path(mimiq_config['MIMIQ_TOKEN_FILE']).write_text('secret-value', encoding='utf-8')
    factory = Mock()
    monkeypatch.setattr(mc, 'MimiqConnection', factory)
    info = connection.get_mimiq_account_info()
    assert 'MIMIQ V2' in info
    assert 'authentication not checked' in info
    assert 'secret-value' not in info
    factory.assert_not_called()


@pytest.mark.parametrize('confirm', ['y', 'n'])
def test_mimiq_delete_respects_confirmation_and_clears_cached_credentials(
    monkeypatch: pytest.MonkeyPatch, mimiq_config: dict[str, str], confirm: str
):
    token_path = Path(mimiq_config['MIMIQ_TOKEN_FILE'])
    token_path.write_text('secret-value', encoding='utf-8')
    mimiq_config.update(MIMIQ_PLATFORM='v2', MIMIQ_CONFIGURED='True')
    monkeypatch.setattr(mc, 'MimiqConnection', Mock())
    connection.get_mimiq_connection()
    monkeypatch.setattr('builtins.input', lambda _: confirm)
    message, _ = connection.delete_mimiq_account()
    assert mimiq_config['IBM_CONFIGURED'] == 'True'
    if confirm == 'y':
        assert 'deleted' in message
        assert not token_path.exists()
        assert mimiq_config['MIMIQ_CONFIGURED'] == 'False'
        assert mimiq_config['MIMIQ_PLATFORM'] == ''
        assert mimiq_config['MIMIQ_TOKEN_FILE'] == ''
        with pytest.raises(RuntimeError, match='No MIMIQ account configured'):
            connection.get_mimiq_connection()
    else:
        assert message == 'Canceled.'
        assert token_path.exists()
        assert mimiq_config['MIMIQ_CONFIGURED'] == 'True'


@pytest.mark.parametrize('selected', [0, 1])
def test_mimiq_interactive_setup_selects_platform_and_returns_menu_result(
    monkeypatch: pytest.MonkeyPatch, selected: int
):
    setup = Mock()
    monkeypatch.setattr(connection, 'setup_mimiq_account', setup)

    def choose(question: QuestionNode):
        answer = question.answers[selected]
        assert answer.next_question is None
        message, arguments = answer.action()
        assert 'correctly configured' in message
        assert arguments == []

    monkeypatch.setattr('mpqp.tools.choice_tree.run_choice_tree', choose)
    assert connection.config_mimiq_account() == ('', [])
    setup.assert_called_once_with('v2' if selected == 0 else 'legacy')


def test_mimiq_interactive_setup_reports_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        connection,
        'setup_mimiq_account',
        Mock(side_effect=RuntimeError('Login failed')),
    )

    def choose(question: QuestionNode):
        message, arguments = question.answers[0].action()
        assert 'configuration failed' in message
        assert arguments == []

    monkeypatch.setattr('mpqp.tools.choice_tree.run_choice_tree', choose)
    connection.config_mimiq_account()


def test_mimiq_reconfiguration_discards_cached_original_connection(
    monkeypatch: pytest.MonkeyPatch, mimiq_config: dict[str, str]
):
    token_path = Path(mimiq_config['MIMIQ_TOKEN_FILE'])
    token_path.write_text('{}', encoding='utf-8')
    original, setup, reloaded = Mock(), Mock(), Mock()
    factory = Mock(side_effect=[original, setup, reloaded])
    monkeypatch.setattr(mc, 'MimiqConnection', factory)
    assert connection.get_mimiq_connection() is original
    connection.setup_mimiq_account('v2')
    assert connection.get_mimiq_connection() is reloaded
    assert [call.args[0] for call in factory.call_args_list] == [
        mc.QPERFECT_CLOUD,
        mc.QPERFECT_CLOUD_V2,
        mc.QPERFECT_CLOUD_V2,
    ]
    reloaded.loadtoken.assert_called_once_with(str(token_path))


def test_mimiq_configuration_save_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, mimiq_config: dict[str, str]
):
    monkeypatch.setattr(mc, 'MimiqConnection', Mock())
    monkeypatch.setattr(connection, 'save_env_variable', Mock(return_value=False))
    with pytest.raises(RuntimeError, match='Unable to save'):
        connection.setup_mimiq_account()
    assert mimiq_config.get('MIMIQ_CONFIGURED') != 'True'
