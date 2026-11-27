"""Offline coverage of provider actions exposed by the setup menu."""

from unittest.mock import Mock

import pytest

from mpqp.execution.connection import mimiq_connection
from mpqp.tools.choice_tree import QuestionNode
from mpqp_scripts import setup_connections


def test_setup_menu_exposes_mimiq(monkeypatch: pytest.MonkeyPatch):
    configure = Mock(return_value=('', []))
    monkeypatch.setattr(mimiq_connection, 'config_mimiq_account', configure)

    def choose(question: QuestionNode):
        next(answer for answer in question.answers if answer.label == 'MIMIQ').action()

    monkeypatch.setattr('mpqp.tools.choice_tree.run_choice_tree', choose)
    setup_connections.main_setup()
    configure.assert_called_once_with()


@pytest.mark.parametrize('selected', ['MIMIQ', 'All'])
def test_delete_menu_includes_mimiq(monkeypatch: pytest.MonkeyPatch, selected: str):
    delete = Mock(return_value=('', []))
    monkeypatch.setattr(mimiq_connection, 'delete_mimiq_account', delete)
    monkeypatch.setattr(
        'mpqp.environment.env_manager.get_env_variable',
        lambda key: 'True' if key == 'MIMIQ_CONFIGURED' else 'False',
    )

    def choose(question: QuestionNode):
        next(answer for answer in question.answers if answer.label == selected).action()

    monkeypatch.setattr('mpqp.tools.choice_tree.run_choice_tree', choose)
    setup_connections.delete_config()
    delete.assert_called_once_with()


@pytest.mark.parametrize('ibm_sdk_present', [True, False])
def test_recap_includes_mimiq(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ibm_sdk_present: bool,
):
    for module, function in (
        ('ibm_connection', 'get_active_account_info'),
        ('aws_connection', 'get_aws_braket_account_info'),
        ('ionq_connection', 'get_ionq_account_info'),
        ('azure_connection', 'get_azure_account_info'),
        ('quantinuum_connection', 'get_quantinuum_account_info'),
    ):
        monkeypatch.setattr(
            f'mpqp.execution.connection.{module}.{function}',
            Mock(return_value='Not configured'),
        )
    if not ibm_sdk_present:
        monkeypatch.setattr(
            'mpqp.execution.connection.ibm_connection.get_active_account_info',
            Mock(side_effect=ModuleNotFoundError('qiskit_ibm_runtime')),
        )
    monkeypatch.setattr('mpqp.environment.env_manager.get_env_variable', lambda _: '')
    monkeypatch.setattr('builtins.input', lambda _: '')
    monkeypatch.setattr(
        mimiq_connection, 'get_mimiq_account_info', lambda: 'MIMIQ V2 saved token'
    )
    setup_connections.print_config_info()
    assert 'MIMIQ V2 saved token' in capsys.readouterr().out
