from getpass import getpass
from typing import TYPE_CHECKING

from termcolor import colored

from mpqp.environment.env_manager import get_env_variable, save_env_variable
from mpqp.execution.devices import IBMDevice
from mpqp.tools.errors import IBMRemoteExecutionError

if TYPE_CHECKING:
    from qiskit.providers.backend import BackendV2
    from qiskit_ibm_runtime import QiskitRuntimeService


Runtime_Service = None


def config_ibm_account(token: str, channel: str):
    """Configure and save locally IBM Quantum account's information.

    Args:
        token: IBM Quantum API token (API key).
        channel: The channel to use for the account (default is "ibm_quantum_platform").
    Raises:
        IBMRemoteExecutionError: If the account could not be saved.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    channel = channel.strip()

    try:
        QiskitRuntimeService.save_account(
            channel=channel,  # pyright: ignore[reportArgumentType]
            token=token.strip(),
            overwrite=True,
            set_as_default=True,
        )
        save_env_variable("IBM_CONFIGURED", "True")
        save_env_variable("IBM_TOKEN", token.strip())
        save_env_variable("IBM_CHANNEL", channel)
    except Exception as err:
        # if an error occurred, we put False in the mpqp config file
        save_env_variable("IBM_CONFIGURED", "False")
        raise IBMRemoteExecutionError(
            "Error when saving the account.\nTrace: " + str(err)
        )


def setup_ibm_account():
    """Set up and update the IBM Quantum account using the existing
    configuration (or by asking for the token if not already configured)."""
    was_configured = get_env_variable("IBM_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "An IBMQ account is already configured. Do you want to update it? [y/N]"
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    DEFAULT_CHANNEL = "ibm_quantum_platform"
    OTHER_CHANNELS = {"ibm_cloud"}
    channel = input(
        f"Enter the channel ({colored(DEFAULT_CHANNEL, attrs=["underline"])}, "
        f"{', '.join(OTHER_CHANNELS)}): "
    ).strip()
    if channel == "":
        channel = DEFAULT_CHANNEL
        print(colored(f"set to {DEFAULT_CHANNEL}", "yellow"))

    if channel not in OTHER_CHANNELS.union(DEFAULT_CHANNEL):
        print(
            colored(
                f"Invalid channel. Use one of {OTHER_CHANNELS.union(DEFAULT_CHANNEL)}.", "red"
            )
        )
        getpass("Press 'Enter' to continue")
        return "", []

    token = getpass("Enter your IBM Quantum / IBM Cloud API key (hidden): ").strip()
    if token == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []

    old_token = get_env_variable("IBM_TOKEN")
    old_channel = get_env_variable("IBM_CHANNEL")

    config_ibm_account(token, channel)
    if test_connection():
        return "IBMQ account correctly configured", []
    else:
        if was_configured:
            config_ibm_account(old_token, old_channel)
        else:
            save_env_variable("IBM_CONFIGURED", "False")
        getpass("Press 'Enter' to continue")
        return "", []


def test_connection() -> bool:
    """Test if the connection to the provider works.

    Returns:
        ``False`` if login failed.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.exceptions import IBMNotAuthorizedError

    global Runtime_Service
    try:
        Runtime_Service = QiskitRuntimeService()
    except IBMNotAuthorizedError as err:
        if "Login failed" in str(err):
            print(colored("Wrong credentials", "red"))
            return False
        else:
            raise err
    return True


def get_QiskitRuntimeService() -> "QiskitRuntimeService":
    """Return the QiskitRuntimeService needed for remote connection and
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
            Runtime_Service = QiskitRuntimeService()
        except Exception as err:
            raise IBMRemoteExecutionError(
                "Error when instantiating QiskitRuntimeService (probably wrong token saved "
                "in the account).\nTrace: " + str(err)
            )
    return Runtime_Service


def get_active_account_info() -> str:
    """Return the information concerning the active IBM Quantum account.

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
    if account is None:
        return "Account not configured"
    return f"""    Channel: {account["channel"]}
    Token: {account["token"][:5]}*****
    URL: {account["url"]}
    Verify: {account["verify"]}"""


def delete_ibm_account():
    """Delete the locally stored IBM Quantum account configuration."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    global Runtime_Service

    decision = input(
        colored(
            "This will delete the local IBM Quantum configuration. Continue? [y/N] ",
            "yellow",
        )
    )
    if decision.lower().strip() != "y":
        return "Canceled.", []

    try:
        QiskitRuntimeService.delete_account()
    except Exception:
        pass

    save_env_variable("IBM_CONFIGURED", "False")
    save_env_variable("IBM_TOKEN", "")
    save_env_variable("IBM_CHANNEL", "")

    Runtime_Service = None

    print(colored("IBM Quantum account deleted.", "green"))
    input("Press 'Enter' to continue")

    return "IBM account deleted", []


def get_backend(device: IBMDevice) -> "BackendV2":
    """Retrieve the corresponding ``qiskit`` remote device.

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
    """Retrieve all the job ids of this account.

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
