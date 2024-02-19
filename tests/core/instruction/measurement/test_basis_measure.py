import pytest

from mpqp.core.instruction.measurement.basis import ComputationalBasis, HadamardBasis
from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.languages import Language


def test_basis_measure_init():
    measure = BasisMeasure([0, 1], shots=1024, basis=ComputationalBasis())
    assert measure.targets == [0, 1]
    assert measure.shots == 1024
    assert isinstance(measure.basis, ComputationalBasis)


def test_basis_measure_init_fails_duplicate_c_targets():
    with pytest.raises(ValueError, match="Duplicate registers in targets"):
        BasisMeasure(targets=[0, 1], c_targets=[2, 2, 3], shots=1024)


def test_basis_measure_to_other_language_not_implemented():
    measure = BasisMeasure([0], basis=HadamardBasis)
    with pytest.raises(NotImplementedError):
        measure.to_other_language(language=Language.QISKIT)


def test_basis_measure_repr():
    measure = BasisMeasure([0, 1], shots=1024)
    representation = repr(measure)
    assert representation == "BasisMeasure([0, 1], shots=1024)"
