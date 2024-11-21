from getpass import getpass
from typing import TYPE_CHECKING

from termcolor import colored
from typeguard import typechecked

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.execution.devices import IBMDevice
from mpqp.tools.errors import IBMRemoteExecutionError

if TYPE_CHECKING:
    from qiskit.providers.backend import BackendV2
    from qiskit_ibm_runtime import QiskitRuntimeService


Runtime_Service = None


@typechecked
def config_ibm_account(token: str):
    """Configure and save locally IBM Quantum account's information.

    Args:
        token: IBM Quantum API token.

    Raises:
        IBMRemoteExecutionError: If the account could not be saved.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum", token=token, overwrite=True
        )
        save_env_variable("IBM_CONFIGURED", "True")
        save_env_variable("IBM_TOKEN", token)
    except Exception as err:
        # if an error occurred, we put False in the mpqp config file
        save_env_variable("IBM_CONFIGURED", "False")
        raise IBMRemoteExecutionError(
            "Error when saving the account.\nTrace: " + str(err)
        )


def setup_ibm_account():
    """Setups and updates the IBM Quantum account using the existing
    configuration (or by asking for the token if not already configured)."""
    was_configured = get_env_variable("IBM_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "An IBMQ account is already configured. Do you want to update it? [y/N]"
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    token = getpass("Enter your IBMQ token (hidden): ")
    if token == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    old_token = get_env_variable("IBM_TOKEN")
    config_ibm_account(token)
    if test_connection():
        return "IBMQ account correctly configured", []
    else:
        if was_configured:
            config_ibm_account(old_token)
        else:
            save_env_variable("IBM_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []


def test_connection() -> bool:
    """Tests if the connection to the provider works.

    Returns:
        ``False`` if login failed.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.exceptions import IBMNotAuthorizedError

    global Runtime_Service
    try:
        Runtime_Service = QiskitRuntimeService(channel="ibm_quantum")
    except IBMNotAuthorizedError as err:
        if "Login failed" in str(err):
            print(colored("Wrong credentials", "red"))
            return False
        else:
            raise err
    return True


def get_QiskitRuntimeService() -> "QiskitRuntimeService":
    """Returns the QiskitRuntimeService needed for remote connection and
    execution.

    Raises:
        IBMRemoteExecutionError: When the ``qiskit`` runtime is not configured
            or the configuration cannot be retrieved.

    Example:
        >>> service = get_QiskitRuntimeService()
        >>> service.jobs()
        [<RuntimeJob('cmdj3b4nktricigarn8g', 'estimator')>,
         <RuntimeJob('cmdj3a74mi97k7j7ujv0', 'sampler')>,
         <RuntimeJob('cmama29054sir2cq94og', 'estimator')>,
         <RuntimeJob('cmama14pduldih1q4ktg', 'sampler')>,
         <RuntimeJob('cm7vds4pduldih1k1mq0', 'sampler')>]

    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    global Runtime_Service
    if Runtime_Service is None:
        if get_env_variable("IBM_CONFIGURED") == "False":
            raise IBMRemoteExecutionError(
                "Error when instantiating QiskitRuntimeService. No IBM account configured."
            )
        try:
            Runtime_Service = QiskitRuntimeService(channel="ibm_quantum")
        except Exception as err:
            raise IBMRemoteExecutionError(
                "Error when instantiating QiskitRuntimeService (probably wrong token saved "
                "in the account).\nTrace: " + str(err)
            )
    return Runtime_Service


def get_active_account_info() -> str:
    """Returns the information concerning the active IBM Quantum account.

    Returns:
        The description containing the account information.

    Example:
        >>> print(get_active_account_info())
            Channel: ibm_quantum
            Instance: ibm-q-startup/colibritd/default
            Token: bf5e5*****
            URL: https://auth.quantum-computing.ibm.com/api
            Verify: True

    """
    service = get_QiskitRuntimeService()
    account = service.active_account()
    if TYPE_CHECKING:
        assert account is not None
    return f"""    Channel: {account["channel"]}
    Token: {account["token"][:5]}*****
    URL: {account["url"]}
    Verify: {account["verify"]}"""


@typechecked
def get_backend(device: IBMDevice) -> "BackendV2":
    """Retrieves the corresponding ``qiskit`` remote device.

    Args:
        device: The device to get from the qiskit Runtime service.

    Returns:
        The requested backend.

    Raises:
        ValueError: If the required backend is a local simulator.
        IBMRemoteExecutionError: If the device was not found.

    Example:
        >>> brisbane = get_backend(IBMDevice.IBM_BRISBANE)
        >>> brisbane.properties().gates[0].parameters
        [Nduv(datetime.datetime(2024, 1, 9, 11, 3, 18, tzinfo=tzlocal()), gate_error, , 0.00045619997922344296),
         Nduv(datetime.datetime(2024, 1, 9, 15, 41, 39, tzinfo=tzlocal()), gate_length, ns, 60)]

    """
    if not device.is_remote():
        raise ValueError("Expected a remote IBM device but got a local simulator.")
    from qiskit.providers.exceptions import QiskitBackendNotFoundError

    service = get_QiskitRuntimeService()

    try:
        if device == IBMDevice.IBM_LEAST_BUSY:
            return service.least_busy(operational=True)
        return service.backend(device.value)
    except QiskitBackendNotFoundError as err:
        raise IBMRemoteExecutionError(
            f"Requested device {device} not found. Verify if your instances "
            "allows to access this machine, or the device's name.\n"
            f"Trace: {err}"
        )


def get_all_job_ids() -> list[str]:
    """Retrieves all the job ids of this account.

    Returns:
        The list of job ids.

    Example:
        >>> get_all_job_ids()
        ['cm6pp7e879ps6bbo7m30', 'cm6ou0q70abqioeudkd0', 'cm6opgcpduldih1hq7j0', 'cm01vp4pduldih0uoi2g',
        'cnvw8z3b08x0008y3e4g', 'cnvw7qyb08x0008y3e0g', 'cnvw7fdvn4c0008a6ztg', 'cnvw79dvn4c0008a6zt0',
        'cnvw64rb08x0008y3dx0', 'cnvw5z7wsx00008wybcg', 'cmdj3b4nktricigarn8g', 'cmdj3a74mi97k7j7ujv0',
        'cmama29054sir2cq94og', 'cmama14pduldih1q4ktg', 'cm80qmi70abqiof0o170', 'cm80qlkpduldih1k4png',
        'cm80pb1054sir2ck9i3g', 'cm80pa6879ps6bbqg2pg', 'cm7vdugiidfp3m8rg02g', 'cm7vds4pduldih1k1mq0']

    """
    if get_env_variable("IBM_CONFIGURED") == "True":
        return [job.job_id() for job in get_QiskitRuntimeService().jobs(limit=None)]
    return []
