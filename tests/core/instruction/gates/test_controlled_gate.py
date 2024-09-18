from typing import Optional

import numpy as np
import numpy.typing as npt
from qiskit.circuit import Parameter

from mpqp.core.instruction.gates.controlled_gate import ControlledGate
from mpqp.core.languages import Language


class CustomControlledGate(ControlledGate):
    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        assert self.non_controlled_gate is not None
        return self.non_controlled_gate.to_other_language(language, qiskit_parameters)

    def to_matrix(self) -> npt.NDArray[np.complex64]:
        return np.array([[1, 0], [0, 1]], dtype=np.complex64)


# TODO
