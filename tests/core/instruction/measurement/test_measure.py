from typing import Optional, TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from qiskit.circuit import Parameter

from mpqp.core.instruction.measurement import Measure
from mpqp.core.languages import Language
from mpqp.tools.generics import flatten


class CustomMeasure(Measure):
    def __init__(self, targets: list[int], shots: int = 0, label: Optional[str] = None):
        if shots < 0:
            raise ValueError(f"Negative number of shots makes no sense, given {shots}")

        super().__init__(targets, shots, label)
        self.c_targets = []

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        if language == Language.QISKIT:
            pass
        else:
            raise ValueError(f"Unsupported language: {language}")

    @property
    def nb_cbits(self):
        return len(flatten(self.c_targets))


def test_measure_init():
    targets = [0, 1, 2]
    shots = 1
    label = "test_measure"
    measure = CustomMeasure(targets, shots, label)

    assert measure.targets == targets
    assert measure.shots == shots
    assert measure.label == label


def test_negative_shots_raises_error():
    with pytest.raises(ValueError):
        CustomMeasure([0, 1], shots=-1)


def test_nb_qubits():
    measure = CustomMeasure([0, 1])
    assert measure.nb_qubits == 2


def test_nb_cbits_multiple_c_targets():
    custom_measure = CustomMeasure([0, 1], shots=10, label="test_measure")
    custom_measure.c_targets = [[0, 1], [2, 3]]
    assert custom_measure.nb_cbits == len(flatten(custom_measure.c_targets))
