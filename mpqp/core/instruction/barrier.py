"""A barrier is a purely cosmetic instruction. In fact, at execution it is
removed because it could have a negative impact on the execution speed of the
circuit (since it artificially increases the depth)."""

from typing import Optional
from qiskit.circuit.library import Barrier as QiskitBarrier

from qiskit.circuit import Parameter

from mpqp.core.languages import Language
from .instruction import Instruction


class Barrier(Instruction):
    """Visual indicator of the grouping of circuit sections"""

    def __init__(self):
        super().__init__([0])
        self.size = 0
        """Size of the barrier (set to 0 by default)."""

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        if language == Language.QISKIT:
            return QiskitBarrier(self.size)
