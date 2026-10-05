from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
)
from mpqp.environment.var_cache import (
    InstalledProviders,
)

if InstalledProviders.QISKIT in _INSTALLED_MPQP_PROVIDERS:
    from qiskit_ibm_catalog.serverless import RunnableQiskitFunction

    def get_qiskit_function(func_name: str, crn: str) -> RunnableQiskitFunction:
        """Returns the QiskitFunction corresponding to the inputted name and CRN.

        Note: This function requires that you have an initiated Qiskit account through setup_connections.
            For more information about it see `setup_connections`

        Args:
            func_name: Name of the function to search in the Catalog.
            crn: IBM's "Cloud Resource Names" more information can be found here: https://cloud.ibm.com/docs/account?topic=account-crn.
        """
        from mpqp.environment.env_manager import get_env_variable
        from qiskit_ibm_catalog import QiskitFunctionsCatalog

        if (
            get_env_variable("IBM_CONFIGURED") == "False"
            or get_env_variable("IBM_CONFIGURED") == ""
        ):
            from mpqp.tools.errors import IBMAccountInitializationMissing

            raise IBMAccountInitializationMissing(
                "The IBM account is not currently configured, please run setup_connections to use this feature."
            )
        catalog = QiskitFunctionsCatalog(
            token=get_env_variable("IBM_TOKEN"),
            channel=get_env_variable("IBM_CHANNEL"),
            instance=crn,
        )
        function = catalog.load(func_name)
        if function is None:
            raise ValueError(
                f"Function name: {func_name} was not found in the requested catalog."
            )
        return function
