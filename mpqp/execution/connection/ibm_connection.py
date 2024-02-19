from getpass import getpass

from qiskit.providers import QiskitBackendNotFoundError
from qiskit.providers.backend import BackendV1
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_provider.accounts import AccountNotFoundError
from qiskit_ibm_provider.api.exceptions import RequestsApiError
from qiskit_ibm_runtime import QiskitRuntimeService
from termcolor import colored
from typeguard import typechecked

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.execution.devices import IBMDevice
from mpqp.tools.errors import IBMRemoteExecutionError

Ibm_Provider = None
Runtime_Service = None


@typechecked
def config_ibm_account(token: str):
    """
    Configure and save locally IBM Quantum account's information.

    Args:
        token: IBM Quantum API token.

    Raises:
        IBMRemoteExecutionError
    """

    try:
        IBMProvider.save_account(token=token, overwrite=True)
        save_env_variable("IBM_CONFIGURED", "True")
        save_env_variable("IBM_TOKEN", token)
    except Exception as err:
        # if an error occurred, we put False in the mpqp config file
        save_env_variable("IBM_CONFIGURED", "False")
        raise IBMRemoteExecutionError(
            "Error when saving the account.\nTrace: " + str(err)
        )


def setup_ibm_account():
    """Setups and updates the IBM Q account using the existing configuration and by asking for the token ."""
    was_configured = get_env_variable("IBM_CONFIGURED") == "True"

    if was_configured:
        decision = input(
            "An IBMQ account is already configured. Do you want to update it? [y/N]"
        )
        if decision.lower().strip() != "y":
            return "Canceled.", []

    token = getpass("Enter your IBM token (hidden): ")
    if token == "":
        print(colored("Empty credentials", "red"))
        getpass("Press 'Enter' to continue")
        return "", []
    old_token = get_env_variable("IBM_TOKEN")
    config_ibm_account(token)
    if test_connection():
        return "IBM Q account correctly configured", []
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
    try:
        IBMProvider()
    except RequestsApiError as err:
        if "Login failed" in str(err):
            print(colored("Wrong credentials", "red"))
            return False
        else:
            raise err
    return True


def get_IBMProvider() -> IBMProvider:
    """Returns the IBMProvider needed to get one or several backends for
    execution.

    Example:
        >>> instance = get_IBMProvider()
        >>> instance.backends()
        [<IBMBackend('ibmq_qasm_simulator')>,
         <IBMBackend('simulator_extended_stabilizer')>,
         <IBMBackend('simulator_mps')>,
         <IBMBackend('simulator_stabilizer')>,
         <IBMBackend('simulator_statevector')>,
         <IBMBackend('ibm_brisbane')>,
         <IBMBackend('ibm_kyoto')>,
         <IBMBackend('ibm_osaka')>]

    Raises:
        IBMRemoteExecutionError
    """
    global Ibm_Provider
    if Ibm_Provider is None:
        if get_env_variable("IBM_CONFIGURED") == "False":
            raise IBMRemoteExecutionError(
                "Error when instantiating IBM Provider. No IBM Q account configured."
            )
        try:
            Ibm_Provider = IBMProvider()
        except RequestsApiError as err:
            raise IBMRemoteExecutionError(
                "Error when instantiating IBM Provider (probably wrong token saved "
                "in the account).\nTrace: " + str(err)
            )
        except AccountNotFoundError as err:
            raise IBMRemoteExecutionError(
                "Error when instantiating IBM Provider. No IBM Q account configured.\nTrace: "
                + str(err)
            )
    return Ibm_Provider


def get_QiskitRuntimeService() -> QiskitRuntimeService:
    """
    Returns the QiskitRuntimeService needed for remote connection and execution

    Example:
        >>> service = get_QiskitRuntimeService()
        >>> service.jobs()
        [<RuntimeJob('cmdj3b4nktricigarn8g', 'estimator')>,
         <RuntimeJob('cmdj3a74mi97k7j7ujv0', 'sampler')>,
         <RuntimeJob('cmama29054sir2cq94og', 'estimator')>,
         <RuntimeJob('cmama14pduldih1q4ktg', 'sampler')>,
         <RuntimeJob('cm7vds4pduldih1k1mq0', 'sampler')>]

    Raises:
        IBMRemoteExecutionError
    """
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
    """
    Returns the information concerning the active IBMQ account

    Example:
        >>> print(get_active_account_info())
            Channel: ibm_quantum
            Instance: ibm-q-startup/colibritd/default
            Token: bf5e5*****
            URL: https://auth.quantum-computing.ibm.com/api
            Verify: True

    Returns:
        A string describing the account info.
    """
    provider = get_IBMProvider()
    account = provider.active_account()
    assert account is not None
    return f"""    Channel: {account["channel"]}
    Instance: {account["instance"]}
    Token: {account["token"][:5]}*****
    URL: {account["url"]}
    Verify: {account["verify"]}"""


@typechecked
def get_backend(device: IBMDevice) -> BackendV1:
    """
    Retrieves the IBM Q remote device corresponding to the device in parameter

    Args:
        device: The IBMDevice to get from IBMQ provider.

    Example:
        >>> brisbane = get_backend(IBMDevice.IBM_BRISBANE)
        >>> brisbane.properties().gates[0].parameters
        [Nduv(datetime.datetime(2024, 1, 9, 11, 3, 18, tzinfo=tzlocal()), gate_error, , 0.00045619997922344296),
         Nduv(datetime.datetime(2024, 1, 9, 15, 41, 39, tzinfo=tzlocal()), gate_length, ns, 60)]

    Returns:
        A qiskit.providers.backend.Backend object that will be use to execute circuit.

    Raises:
        IBMRemoteExecutionError
    """
    # NOTE:
    #       Question : when a backend is present in several IBMQ instances, which instance does it use to submit jobs
    # on this backend ? Typically if with colibritd instance i have more priority and by default it uses ibmq
    # instance, then i lose something here.
    #       Answer : it takes the default instance attached to the account (the higher plan, usually).
    if not device.is_remote():
        raise ValueError("Expected a remote IBMQ device but got a local simulator.")

    provider = get_IBMProvider()

    try:
        backend = provider.get_backend(device.value)

    except QiskitBackendNotFoundError as err:
        raise IBMRemoteExecutionError(
            f"Requested device {device} not found. Verify if your instances "
            "allows to access this machine, or the device's name.\n"
            f"Trace: {err}"
        )

    return backend


def get_all_job_ids() -> list[str]:
    """
    Retrieves all the job ids of this account from the several IBM remote providers
    (IBMProvider, QiskitRuntimeService, ...)

    Example:
        >>> get_all_job_ids()
        ['cm6pp7e879ps6bbo7m30', 'cm6ou0q70abqioeudkd0', 'cm6opgcpduldih1hq7j0', 'cm01vp4pduldih0uoi2g',
        'cnvw8z3b08x0008y3e4g', 'cnvw7qyb08x0008y3e0g', 'cnvw7fdvn4c0008a6ztg', 'cnvw79dvn4c0008a6zt0',
        'cnvw64rb08x0008y3dx0', 'cnvw5z7wsx00008wybcg', 'cmdj3b4nktricigarn8g', 'cmdj3a74mi97k7j7ujv0',
        'cmama29054sir2cq94og', 'cmama14pduldih1q4ktg', 'cm80qmi70abqiof0o170', 'cm80qlkpduldih1k4png',
        'cm80pb1054sir2ck9i3g', 'cm80pa6879ps6bbqg2pg', 'cm7vdugiidfp3m8rg02g', 'cm7vds4pduldih1k1mq0']
    """
    all_job_ids = []

    ibm_provider = get_IBMProvider()  # using IBMProvider
    service = get_QiskitRuntimeService()  # using QiskitRuntimeService

    if ibm_provider:
        all_job_ids.extend([job.job_id() for job in ibm_provider.jobs()])

    if service:
        all_job_ids.extend([job.job_id() for job in service.jobs()])

    return all_job_ids
