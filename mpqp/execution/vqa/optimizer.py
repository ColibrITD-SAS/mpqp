"""For now, our minimizer is a wrapper around ``scipy``'s minimizer. The
:class:`Optimizer` enum lists all the methods validated with the rest of the
library."""

from enum import Enum


class Optimizer(Enum):
    """Enum used to select the optimizer for the VQA."""

    BFGS = "BFGS"
    L_BFGS_B = "L-BFGS-B"
    COBYLA = "COBYLA"
    POWELL = "POWELL"
    NELDER_MEAD = "Nelder-Mead"
    CMAES = "CMAES"
    SLSQP = "SLSQP"


##TODO CMAES and SLSQP implementation
