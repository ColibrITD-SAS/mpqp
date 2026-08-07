"""This file regroups all provider specific parametrization needed
to configure more precisely the run on local or remote devices."""

from typing import Optional


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
