"""This file regroups all provider specific parametrization needed
to configure more precisely the run on local or remote devices."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pytket.partition import PauliPartitionStrat


class ProviderParams:
    """Base class meant to regroup a set of provider specific parameters needed at runtime."""

    pass


class QiskitParams(ProviderParams):
    """
    Class meant to regroup all IBM specific parameters for remote execution.

    Args:
        instance: IBM Quantum instance on which the job(s) should be sent.

    """

    def __init__(self, instance: Optional[str] = None):
        self.instance = instance


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
