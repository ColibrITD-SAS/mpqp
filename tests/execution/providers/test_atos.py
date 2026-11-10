"""Add ``--long`` to the cli args to run this test (disabled by default because
too slow)"""

import sys
from unittest.mock import MagicMock, Mock

# 3M-TODO test everything
import numpy as np
import pytest

from mpqp import ATOSDevice, BasisMeasure, ExpectationMeasure, Observable, QCircuit, run
from mpqp.gates import *


def _sample_qlm_result() -> MagicMock:
    sample = Mock(err=0.0, probability=1.0)
    sample.state.int = 0
    sample.state.bitstring = "0"

    result = MagicMock(value=None, amplitude=None)
    result.__iter__.return_value = [sample]
    return result


def _remote_qlm_job(job_id: str, result: object) -> Mock:
    remote_job = Mock()
    remote_job.get_info.return_value.id = job_id
    remote_job.join.return_value = result
    return remote_job


@pytest.fixture
def remote_qlm_qpu(monkeypatch: pytest.MonkeyPatch) -> Mock:
    qpu = Mock()
    monkeypatch.setattr(
        "mpqp.execution.providers.atos.get_remote_qpu",
        Mock(return_value=qpu),
    )
    return qpu


@pytest.mark.provider("myqlm")
def test_run_qlm_binds_single_job_id(remote_qlm_qpu: Mock):
    job_id = "qlm_job_id"
    remote_qlm_qpu.submit.return_value = _remote_qlm_job(
        job_id,
        _sample_qlm_result(),
    )

    result = run(
        QCircuit([H(0), BasisMeasure([0], shots=10)]),
        ATOSDevice.QLM_LINALG,
    )

    assert result.job.id == job_id


@pytest.mark.provider("myqlm")
def test_run_qlm_binds_multiple_job_ids(remote_qlm_qpu: Mock):
    job_ids = ["qlm_job_id_1", "qlm_job_id_2"]
    remote_qlm_qpu.submit.side_effect = [
        _remote_qlm_job(job_ids[0], Mock(value=1.0, error=None)),
        _remote_qlm_job(job_ids[1], Mock(value=-1.0, error=None)),
    ]
    measurement = ExpectationMeasure(
        [
            Observable(np.array([[0, 1], [1, 0]])),
            Observable(np.array([[1, 0], [0, -1]])),
        ],
        targets=[0],
        shots=10,
    )

    result = run(
        QCircuit([H(0), measurement]),
        ATOSDevice.QLM_LINALG,
    )

    assert result.job.id == job_ids


@pytest.mark.parametrize(
    "circuit",
    [
        # SAMPLE JOB
        QCircuit(
            [
                T(0),
                CNOT(0, 1),
                Ry(np.pi / 2, 2),
                S_dagger(1),
                CZ(2, 1),
                SWAP(2, 0),
                BasisMeasure(list(range(3)), shots=2000),
            ]
        ),
        # STATEVECTOR JOB
        QCircuit(
            [
                T(0),
                CNOT(0, 1),
                Ry(np.pi / 2, 2),
                S_dagger(1),
                CZ(2, 1),
                SWAP(2, 0),
                BasisMeasure(list(range(3)), shots=0),
            ]
        ),
        # OBSERVABLE JOB
        QCircuit(
            [
                H(0),
                Rx(1.76, 1),
                ExpectationMeasure(
                    observable=Observable(
                        np.array(
                            [
                                [0.63, 0.5, 1, 1],
                                [0.5, 0.82, 1, 1],
                                [1, 1, 1, 0.33],
                                [1, 1, 0.33, 0.3],
                            ],
                        )
                    ),
                    targets=[0, 1],
                    shots=1000,
                ),
            ]
        ),
    ],
)
def remote_qlm_execution_binds_job_ids(circuit: QCircuit):
    result = run(circuit, ATOSDevice.QLM_LINALG)

    job_ids = result.job.id
    assert job_ids
    if isinstance(job_ids, list):
        assert all(isinstance(job_id, str) and job_id for job_id in job_ids)
    else:
        assert isinstance(job_ids, str)


if "--long" in sys.argv:
    test_remote_qlm_execution_binds_job_ids = remote_qlm_execution_binds_job_ids
