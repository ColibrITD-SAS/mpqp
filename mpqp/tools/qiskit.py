from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from qiskit_ibm_catalog.serverless import (  # pyright: ignore[reportMissingImports]
        RunnableQiskitFunction,
    )


def get_qiskit_function(
    func_name: str, crn: Optional[str] = None
) -> RunnableQiskitFunction:
    """Return a Qiskit Function from an IBM Catalog instance.

    Args:
        func_name: Name of the function to load from the Catalog.
        crn: IBM Cloud Resource Name identifying the Catalog instance. When
            omitted, the ``IBM_INSTANCE`` saved by ``setup_connections`` is
            used.

    Returns:
        The requested runnable Qiskit Function.

    Raises:
        IBMAccountInitializationMissing: If no IBM account is configured.
        ValueError: If no Catalog instance is configured or the requested
            function is not available.

    Example:
        Load the Q-CTRL performance-management function using the instance
        configured by ``setup_connections``, then execute a circuit through
        that function. This remote example is not executed as a doctest.

        >>> from mpqp import BasisMeasure, H, IBMDevice, QCircuit, run
        >>> from mpqp.execution.providers.providers_params import QiskitParams
        >>> from mpqp.tools import get_qiskit_function  # doctest: +REMOTE_RUN
        >>> function = get_qiskit_function("q-ctrl/performance-management")
        >>> params = QiskitParams(
        ...     qctrl_function=function,
        ...     options={
        ...         "environment": {
        ...             "job_tags": ["experiment", "qctrl-run"],
        ...         }
        ...     },
        ... )
        >>> circuit = QCircuit([H(0), BasisMeasure([0], shots=1024)])
        >>> result = run(
        ...     circuit,
        ...     IBMDevice.IBM_FEZ,
        ...     provider_params=params,
        ... )

    Note:
        This function requires an IBM account configured with
        ``setup_connections``.
    """
    from qiskit_ibm_catalog import (  # pyright: ignore[reportMissingImports]
        QiskitFunctionsCatalog,
    )

    from mpqp.environment.env_manager import get_env_variable
    from mpqp.tools.errors import IBMAccountInitializationMissing

    if get_env_variable("IBM_CONFIGURED") not in {"True", "true"}:
        raise IBMAccountInitializationMissing(
            "The IBM account is not configured. Run setup_connections before "
            "loading a Qiskit Function."
        )

    instance = crn or get_env_variable("IBM_INSTANCE") or None
    if instance is None:
        raise ValueError(
            "No IBM Catalog instance was provided. Pass crn explicitly or "
            "configure a default IBM_INSTANCE with setup_connections."
        )

    catalog = QiskitFunctionsCatalog(
        token=get_env_variable("IBM_TOKEN"),
        channel=get_env_variable("IBM_CHANNEL"),
        instance=instance,
    )
    function = catalog.load(func_name)
    if function is None:
        raise ValueError(
            f"Function name: {func_name} was not found in the requested catalog."
        )
    return function
