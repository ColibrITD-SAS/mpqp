"""This file regroups all provider specific parametrization needed
to configure more precisely the run on local or remote devices."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderParams:
    """Abstract class meant to regroup a set of provider specific parameters needed at runtime."""

    pass


@dataclass
class QiskitParams(ProviderParams):
    """
    Class meant to regroup all IBM specific parameters for remote execution.

    Args:
        instance: IBM Quantum instance on which the job(s) should be sent.

    """

    instance: Optional[str] = None


@dataclass
class AWSParams(ProviderParams):
    """AWS Braket-specific execution parameters.

    Args:
        reservation_arn: ARN of the Braket direct reservation to use.
    """

    reservation_arn: Optional[str] = None
