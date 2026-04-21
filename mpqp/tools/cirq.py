import sys
from typing import Any, Optional

if "cirq" in sys.modules:  # if cirq is imported instanciate the following classes

    from cirq import Gate as cirqGate
    from cirq import Qid
    from mpqp.tools.generics import Matrix

    class cirqCustomGate(cirqGate):
        def __init__(
            self, matrix: Matrix, decomposition: Any = None, label: Optional[str] = None
        ):
            import numpy as np

            self.decomposition = decomposition
            self.matrix = matrix
            self.label = label
            self._nb_qubits = int(np.log2(len(matrix)))
            super(cirqCustomGate, self)

        def _num_qubits_(self) -> int:
            return self._nb_qubits

        def _unitary_(self) -> Matrix:
            return self.matrix

        def _decompose_(self, qubits: list[Qid]):
            if self.decomposition is None:
                pass
            else:
                return self.decomposition(qubits)

        def _circuit_diagram_info_(self, args: list[float]) -> str | list[str]:
            # we keep args for later implementation
            if self.label:
                return [self.label] * self._nb_qubits
            return ["CustomGate"] * self._nb_qubits
