"""Offline execution against the native Exaqt simulator."""

import sys
from copy import deepcopy
from numbers import Complex
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from sympy import symbols

from mpqp import (
    CNOT,
    TOF,
    BasisMeasure,
    CustomControlledGate,
    CustomGate,
    Depolarizing,
    ExpectationMeasure,
    H,
    HadamardBasis,
    JobStatus,
    MIMIQDevice,
    Observable,
    QCircuit,
    Rxx,
    Ry,
    S,
    X,
    run,
    submit,
)
from mpqp.execution.providers import mimiq
from mpqp.execution.providers.providers_params import MimiqParams, QiskitParams
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError

pytest.importorskip('exaqt')
pytestmark = pytest.mark.provider('mimiq')
DEVICE = MIMIQDevice.LOCAL_STATEVECTOR


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mimiq,
        'get_mimiq_connection',
        Mock(side_effect=AssertionError('Local execution contacted the cloud.')),
    )


@pytest.mark.parametrize(
    'circuit, indices, amplitudes',
    [
        (QCircuit([X(0)], nb_qubits=3), [4], [1]),
        (QCircuit([X(2)], nb_qubits=3), [1], [1]),
        (QCircuit([X(2), CNOT(2, 0)], nb_qubits=3), [5], [1]),
        (QCircuit([X(0), X(2), TOF([2, 0], 1)]), [7], [1]),
        (
            QCircuit([H(0), S(0)], nb_qubits=3),
            [0, 4],
            [1 / np.sqrt(2), 1j / np.sqrt(2)],
        ),
        (
            QCircuit([Rxx(0.37, 2, 0)], nb_qubits=3),
            [0, 5],
            [np.cos(0.37 / 2), -1j * np.sin(0.37 / 2)],
        ),
        (
            QCircuit(
                [
                    CustomGate(
                        np.asarray(
                            np.kron([[0, 1], [1, 0]], H(0).to_canonical_matrix()),
                            dtype=np.complex128,
                        ),
                        [2, 0],
                    )
                ],
                nb_qubits=3,
            ),
            [1, 5],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
        ),
        (
            QCircuit([X(0), X(2), CustomControlledGate([2, 0], Ry(np.pi, 1))]),
            [7],
            [1],
        ),
        (QCircuit(3), [0], [1]),
    ],
)
def test_mimiq_local_statevector_order_and_phase(
    circuit: QCircuit, indices: list[int], amplitudes: list[complex]
):
    original = deepcopy(circuit)
    circuit.input_g_phase = 0.23
    original.input_g_phase = 0.23
    expected = np.zeros(2**circuit.nb_qubits, dtype=np.complex128)
    expected[indices] = np.asarray(amplitudes) * np.exp(0.23j)

    result = run(circuit, DEVICE)

    np.testing.assert_allclose(result.amplitudes, expected, atol=1e-12)
    assert result.amplitudes.dtype == np.complex128
    assert result.job.status == JobStatus.DONE
    assert result.job.id is None
    assert circuit == original


@pytest.mark.parametrize(
    'targets, c_targets, expected',
    [
        ([2, 0], None, '01'),
        ([2, 0], [3, 1], '01'),
        ([0, 2], [3, 1], '10'),
        ([2], [4], '0'),
    ],
)
def test_mimiq_local_partial_and_reordered_sampling(
    targets: list[int], c_targets: list[int] | None, expected: str
):
    circuit = QCircuit(
        [X(0), BasisMeasure(targets, c_targets=c_targets, shots=32)], nb_qubits=3
    )
    original = deepcopy(circuit)

    result = run(circuit, DEVICE, provider_params=MimiqParams(seed=0))

    assert [(sample.bin_str, sample.count) for sample in result.samples] == [
        (expected, 32)
    ]
    assert result.shots == 32
    assert result.job.status == JobStatus.DONE
    assert circuit == original


def test_mimiq_local_sampling_basis_change():
    circuit = QCircuit(
        [H(2), H(0), BasisMeasure([2, 0], shots=32, basis=HadamardBasis())],
        nb_qubits=3,
    )
    original = deepcopy(circuit)
    result = run(circuit, DEVICE, provider_params=MimiqParams(seed=0))
    assert [(sample.bin_str, sample.count) for sample in result.samples] == [('00', 32)]
    assert circuit == original


def test_mimiq_local_zero_shots_returns_full_statevector():
    circuit = QCircuit([X(0), BasisMeasure([2], shots=0)], nb_qubits=3)
    result = run(circuit, DEVICE)
    expected = np.zeros(8, dtype=np.complex128)
    expected[4] = 1
    np.testing.assert_allclose(result.amplitudes, expected, atol=1e-12)
    assert result.shots == 0


def test_mimiq_local_zero_shots_applies_basis_change():
    circuit = QCircuit(
        [H(2), H(0), BasisMeasure([2, 0], shots=0, basis=HadamardBasis())],
        nb_qubits=3,
    )
    expected = np.zeros(8, dtype=np.complex128)
    expected[[0, 2]] = 1 / np.sqrt(2)
    np.testing.assert_allclose(run(circuit, DEVICE).amplitudes, expected, atol=1e-12)


def test_mimiq_local_seed_and_reuse():
    circuit = QCircuit([H(0), H(1), BasisMeasure(shots=128)])
    original = deepcopy(circuit)
    params = MimiqParams(seed=0)
    first = run(circuit, DEVICE, provider_params=params)
    second = run(circuit, DEVICE, provider_params=params)
    assert first.counts == second.counts
    assert sum(first.counts) == 128
    assert circuit == original


def test_mimiq_local_symbolic_binding():
    theta = symbols('theta')
    circuit = QCircuit([Ry(theta, 2)], nb_qubits=3)
    result = run(circuit, DEVICE, values={theta: cast(Complex, np.pi)})
    expected = np.zeros(8, dtype=np.complex128)
    expected[1] = 1
    np.testing.assert_allclose(result.amplitudes, expected, atol=1e-12)
    assert theta in circuit.variables()
    with pytest.raises(ValueError, match='Bind all circuit parameters'):
        run(circuit, DEVICE)


def test_mimiq_local_capabilities():
    assert not DEVICE.is_remote()
    assert DEVICE.supports_state_vector()
    assert DEVICE.supports_samples()
    assert not DEVICE.supports_observable()
    assert not DEVICE.is_noisy_simulator()


def test_mimiq_local_rejects_unsupported_jobs():
    with pytest.raises(DeviceJobIncompatibleError):
        run(QCircuit([H(0), Depolarizing(0.1)]), DEVICE)
    with pytest.raises(DeviceJobIncompatibleError):
        run(
            QCircuit([ExpectationMeasure(Observable(np.array([[0, 1], [1, 0]])))]),
            DEVICE,
        )
    with pytest.raises(RemoteExecutionError):
        submit(QCircuit([X(0)]), DEVICE)
    with pytest.raises(ValueError, match='at least one qubit'):
        run(QCircuit(0), DEVICE)


def test_mimiq_local_missing_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, 'exaqt', None)
    with pytest.raises(ModuleNotFoundError, match=r'mpqp\[mimiq\]'):
        run(QCircuit([X(0)]), DEVICE)


def test_mimiq_local_rejects_other_provider_parameters():
    with pytest.raises(TypeError, match='MimiqParams'):
        run(QCircuit([X(0)]), DEVICE, provider_params=QiskitParams())
