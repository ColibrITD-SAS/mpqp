from __future__ import annotations

# import math
# import numpy as np

# from qat.hardware.default import DefaultGatesSpecification, HardwareModel
# from qiskit_aer.noise import NoiseModel

# from .noise_methods import GateNoise, GateNoiseCombination, NoiseModules
from mpqp.core.languages import Language


class MPQPNoiseModel:
    def __init__(self):
        # 3M-TODO : implement and comment
        self.pp = 1

    def no_idea_what_i_am_doing(
        self, language: Language = Language.QISKIT
    ):  # can be also in MYQLM
        pass
