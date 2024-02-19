from __future__ import annotations

from numpy import pi
import pytest

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.measures import BasisMeasure
from mpqp.execution import Result, StateVector, Sample, Job, JobType
from mpqp.execution.devices import IBMDevice


@pytest.mark.parametrize(
    "job_type, data",
    [
        (JobType.STATE_VECTOR, 0.4),
        (JobType.STATE_VECTOR, []),
        (JobType.OBSERVABLE, []),
        (JobType.OBSERVABLE, StateVector([1])),
        (JobType.SAMPLE, 0.4),
        (JobType.SAMPLE, StateVector([1])),
    ],
)
def test_result_wrong_type(job_type: JobType, data: float | StateVector | list[Sample]):
    c = QCircuit(3)
    c.add(Rx(pi / 2, 1))
    j = Job(job_type, c, IBMDevice.AER_SIMULATOR)
    with pytest.raises(TypeError):
        Result(j, data)


@pytest.mark.parametrize(
    "job_type, data",
    [
        (JobType.STATE_VECTOR, StateVector([0.5, 0.5, 0.5, 0.5])),
        (JobType.OBSERVABLE, 0.4),
        (
            JobType.SAMPLE,
            [Sample(3, index=3, count=250), Sample(3, index=6, count=250)],
        ),
    ],
)
def test_result_right_type(job_type: JobType, data: float | StateVector | list[Sample]):
    size = 3
    c = QCircuit(size)
    c.add(Rx(pi / 2, 1))
    measure = BasisMeasure(list(range(size))) if job_type == JobType.SAMPLE else None
    j = Job(job_type, c, IBMDevice.AER_SIMULATOR, measure)
    if job_type == JobType.SAMPLE:
        assert isinstance(data, list)
        assert all(sample.count is not None for sample in data)
        shots = sum(sample.count for sample in data if sample.count is not None)
    else:
        shots = 0
    r = Result(j, data, shots=shots)
    assert r.device == r.job.device
