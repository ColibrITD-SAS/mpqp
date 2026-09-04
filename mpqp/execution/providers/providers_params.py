"""This file regroups all provider specific parametrization needed
to configure more precisely the run on local or remote devices."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pytket.partition import PauliPartitionStrat


class ProviderParams:
    """Abstract class meant to regroup a set of provider specific parameters needed at runtime."""

    pass


class QiskitParams(ProviderParams):
    """
    Class meant to regroup all IBM specific parameters for remote execution.

    Args:
        instance: IBM Quantum instance on which the job(s) should be sent.

    """

    def __init__(self, instance: Optional[str] = None):
        self.instance = instance


class TketParams(ProviderParams):
    """
    Class meant to regroup all IBM specific parameters for remote execution.

    Args:
        optimisation_level: Optimisation level with which the circuit should be compiled (default at 0)
    """

    def __init__(
        self,
        optimisation_level: Optional[int] = None,
        optimisation_strategy: Optional["PauliPartitionStrat"] = None,
    ):
        self.optimisation_level = (
            optimisation_level if optimisation_level is not None else 0
        )
        self.optimisation_strategy = optimisation_strategy
