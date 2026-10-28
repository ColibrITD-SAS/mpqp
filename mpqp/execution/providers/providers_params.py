"""This file regroups all provider specific parametrization needed
to configure more precisely the run on local or remote devices."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pytket.partition import PauliPartitionStrat
    from qiskit_ibm_catalog.serverless import (  # pyright: ignore[reportMissingImports]
        RunnableQiskitFunction,
    )


class ProviderParams:
    """Base class meant to regroup a set of provider specific parameters needed at runtime."""

    pass


class QiskitParams(ProviderParams):
    """
    Class meant to regroup all IBM specific parameters for remote execution.

    Args:
        instance: IBM Quantum instance on which the job(s) should be sent.
        options: Options passed to IBM Runtime primitives, or adapted for the
            selected Q-CTRL function.
        qctrl_function: Optional Q-CTRL Qiskit Function used instead of the
            standard IBM Runtime primitives.

    Example:
        Configure tags and private execution for an IBM Runtime job:

        >>> from mpqp.execution.providers.providers_params import QiskitParams
        >>> params = QiskitParams(
        ...     options={
        ...         "environment": {
        ...             "private": True,
        ...             "job_tags": ["experiment", "run-1"],
        ...         }
        ...     }
        ... )
        >>> params.options["environment"]["job_tags"]
        ['experiment', 'run-1']

    """

    def __init__(
        self,
        instance: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        qctrl_function: Optional["RunnableQiskitFunction"] = None,
    ):
        self.instance = instance
        self.options = deepcopy(options) if options is not None else {}
        self.qctrl_function = qctrl_function

    def get_options(self) -> dict[str, Any]:
        """Return an isolated option dictionary for the selected execution mode.

        Returns:
            A deep copy of the Runtime options, adapted to the Q-CTRL schema
            when a Q-CTRL function is configured by moving every
            ``environment`` option to the top level.

        Raises:
            TypeError: If ``environment`` is not a dictionary.
            ValueError: If an option is defined with different values in
                ``environment`` and at the top level.

        Example:
            Return isolated IBM Runtime options:

            >>> from mpqp.execution.providers.providers_params import QiskitParams
            >>> params = QiskitParams(
            ...     options={
            ...         "environment": {
            ...             "private": True,
            ...             "job_tags": ["experiment", "run-1"],
            ...         }
            ...     }
            ... )
            >>> params.get_options()["environment"]["private"]
            True
        """
        options = deepcopy(self.options)
        if self.qctrl_function is None:
            return options

        environment = options.pop("environment", None)
        if environment is None:
            return options
        if not isinstance(environment, dict):
            raise TypeError("The environment option must be a dictionary.")

        for name, value in environment.items():
            if name in options and options[name] != value:
                raise ValueError(
                    f"Conflicting option '{name}' was provided at the top level "
                    "and in environment."
                )
            options[name] = value
        return options


class QuantinuumParams(ProviderParams):
    """Configuration parameters for Quantinuum execution.

    Args:
        optimisation_level: Level of optimisation applied when compiling the
            circuit. Defaults to 0.
        commutation_strategy: Strategy used by TKET to group Pauli terms for
            sampled observable jobs on local devices. If ``None``, MPQP chooses
            a strategy based on the commuting type defined by the measurement.
            This parameter is not used for Nexus jobs.
    """

    def __init__(
        self,
        optimisation_level: int = 0,
        commutation_strategy: Optional["PauliPartitionStrat"] = None,
    ):
        self.optimisation_level = optimisation_level
        self.commutation_strategy = commutation_strategy
