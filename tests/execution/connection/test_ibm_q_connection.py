# from getpass import getpass
# from unittest.mock import MagicMock, patch
#
# import pytest
# from qiskit_ibm_provider.api.exceptions import RequestsApiError
#
# from mpqp.execution.connection.ibm_connection import (
#     IBMProvider,
#     config_ibm_account,
#     setup_ibm_account,
#     test_connection,
# )
#
#
# @pytest.fixture
# def mock_ibm_provider(monkeypatch):
#     mock_ibm_provider = MagicMock()
#     monkeypatch.setattr(
#         "mpqp.execution.connection.ibm_connection.IBMProvider", mock_ibm_provider
#     )
#     return mock_ibm_provider
#
#
# @pytest.fixture
# def mock_getpass(monkeypatch):
#     mock_getpass = MagicMock(return_value="test_token")
#     monkeypatch.setattr("getpass.getpass", mock_getpass)
#     return mock_getpass
#
#
# def test_config_ibm_account():
#     token = "test_token"
#     mock_ibm_provider = MagicMock()
#
#     # successful configuration
#     with patch(
#         "mpqp.execution.connection.ibm_connection.IBMProvider", mock_ibm_provider
#     ):
#         config_ibm_account(token)
#
#     mock_ibm_provider.save_account.assert_called_once_with(token=token, overwrite=True)
#
#
# def test_setup_ibm_account_configured():
#     # test case when an IBMQ account is already configured
#     with patch(
#         "mpqp.execution.connection.env_manager.get_env_variable", return_value="True"
#     ), patch("builtins.input", side_effect=["n"]):
#         result, _ = setup_ibm_account()
#         assert result == "Canceled."
#
#
# def test_setup_ibm_account_not_configured():
#     # test case when an IBMQ account is not configured
#     with patch(
#         "mpqp.execution.connection.env_manager.get_env_variable", return_value=None
#     ), patch("builtins.input", side_effect=["y"]), patch(
#         "mpqp.execution.connection.ibm_connection.test_connection", return_value=True
#     ):
#         result, _ = setup_ibm_account()
#         assert result == "IBM Q account correctly configured"
#
#
# def test_setup_ibm_account_failed_and_not_configured():
#     # test case when an IBMQ account is not configured and test_connection fails
#     with patch(
#         "mpqp.execution.connection.env_manager.get_env_variable", return_value=None
#     ), patch("builtins.input", side_effect=["y"]), patch(
#         "mpqp.execution.connection.ibm_connection.test_connection", return_value=False
#     ):
#         result, _ = setup_ibm_account()
#         assert result == ""
#
#
# def test_test_connection():
#     if test_connection():
#         print("Test connection success - unexpected")
#
#     with patch(
#         "mpqp.execution.connection.ibm_connection.IBMProvider",
#         side_effect=RequestsApiError("Login failed"),
#     ):
#         try:
#             test_connection()
#         except RequestsApiError:
#             pass
#
#
# def test_mock_getpass(mock_getpass):
#     with patch("getpass.getpass", mock_getpass):
#         assert getpass() == "test_token"
#
#
# def test_mock_ibm_provider(mock_ibm_provider):
#     with patch(
#         "mpqp.execution.connection.ibm_connection.IBMProvider", mock_ibm_provider
#     ):
#         with pytest.raises(RequestsApiError, match="401 Client Error: Unauthorized"):
#             IBMProvider()

# TODO
