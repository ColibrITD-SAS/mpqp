"""This file regroups all provider specific parametrization needed
to configure more precisely the run on local or remote devices."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pytket.partition import PauliPartitionStrat


@dataclass
class ProviderParams:
    """Base class meant to regroup a set of provider specific parameters needed at runtime."""

    pass


@dataclass
class QiskitParams(ProviderParams):
    """
    Class meant to regroup all IBM specific parameters for remote execution.

    Args:
        instance: IBM Quantum instance on which the job(s) should be sent.

    """

    instance: Optional[str] = None


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


@dataclass
class AWSParams(ProviderParams):
    """AWS Braket-specific execution parameters.

    Args:
        reservation_arn: ARN of the Braket direct reservation to use.
    """

    reservation_arn: Optional[str] = None
