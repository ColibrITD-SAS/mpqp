"""A barrier is a purely cosmetic instruction. In fact, at execution it is
removed because it could have a negative impact on the execution speed of the
circuit (since it artificially increases the depth)."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from qiskit.circuit import Parameter

from mpqp.core.languages import Language

from .instruction import Instruction


class Barrier(Instruction):
    """Visual indicator of how the circuit can be divided.

    Args:
        size: Size of the barrier. If set to 0, it will be a dynamic
            instruction, always spanning the entirety of the circuit.
    """

    def __init__(self, size: int = 0):
        super().__init__(list(range(size)))
        self.size = size
        """See parameter description."""
        if size == 0:
            self._dynamic = True

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QISKIT:
            from qiskit.circuit.library import Barrier as QiskitBarrier

            return QiskitBarrier(self.size)
        elif language == Language.QASM2:
            qubits = ",".join([f"q[{j}]" for j in self.targets])
            return "barrier " + qubits + ";"
        else:
            raise NotImplementedError(f"{language} is not supported")

    def __repr__(self):
        return f"{type(self).__name__}({self.size})"
