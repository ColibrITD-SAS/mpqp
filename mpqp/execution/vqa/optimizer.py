"""For now, our minimizer is a wrapper around ``scipy``'s minimizer. The
:class:`Optimizer` enum lists all the methods validated with the rest of the
library."""

from enum import Enum, auto


class Optimizer(Enum):
    """Enum used to select the optimizer for the VQA."""

    BFGS = auto()
    COBYLA = auto()
    CMAES = auto()
