"""Public MIMIQ translation contracts and offline cloud execution."""

import subprocess
import sys
from copy import deepcopy
from numbers import Complex
from pathlib import Path
from typing import Any, Callable, Optional, cast
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest
from numpy.typing import NDArray
from sympy import symbols

from mpqp import (
    CNOT,
    CP,
    CZ,
    PRX,
    SWAP,
    TOF,
    BasisMeasure,
    CustomControlledGate,
    CustomGate,
    ExpectationMeasure,
    Gate,
    H,
    HadamardBasis,
    JobStatus,
    Language,
    MIMIQDevice,
    Observable,
    P,
    QCircuit,
    Rk,
    Rk_dagger,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    S,
    S_dagger,
    T,
    U,
    X,
    Y,
    Z,
    get_remote_result,
    run,
    submit,
)
from mpqp.execution.providers import mimiq
from mpqp.execution.providers.providers_params import MimiqParams, QiskitParams
from mpqp.tools.errors import DeviceJobIncompatibleError, RemoteExecutionError
from mpqp.translation.mimiq import mpqp_to_mimiq

mc = pytest.importorskip('mimiqcircuits')
pytestmark = pytest.mark.provider('mimiq')


def translated_matrix(circuit: QCircuit) -> NDArray[np.complex128]:
    translated = mpqp_to_mimiq(circuit, skip_measurements=True)
    width = circuit.nb_qubits
    dimension = 2**width
    matrix = np.eye(dimension, dtype=np.complex128)
    for instruction in translated:
        if not isinstance(instruction.operation, mc.Gate):
            continue
        targets = tuple(instruction.qubits)
        axes = targets + tuple(q for q in range(width) if q not in targets) + (width,)
        tensor = matrix.reshape((2,) * width + (dimension,)).transpose(axes)
        operation = instruction.operation
        raw_matrix = (
            operation.matrix
            if isinstance(operation, mc.GateCustom)
            else operation.matrix()
        )
        gate = np.asarray(raw_matrix.tolist(), dtype=np.complex128)
        tensor = (gate @ tensor.reshape(2 ** len(targets), -1)).reshape(tensor.shape)
        matrix = tensor.transpose(np.argsort(axes)).reshape(dimension, dimension)
    return matrix


@pytest.mark.parametrize(
    'gate',
    [
        X(0),
        Y(2),
        Z(1),
        H(0),
        S(1),
        S_dagger(2),
        T(0),
        P(0.3, 1),
        Rx(0.4, 2),
        Ry(-0.2, 0),
        Rz(0.8, 1),
        PRX(0.3, -0.5, 2),
        U(0.2, 0.4, -0.7, 0),
        Rk(3, 1),
        Rk_dagger(3, 1),
        Rxx(0.3, 2, 0),
        Ryy(0.7, 0, 2),
        Rzz(-0.6, 2, 1),
        CNOT(2, 0),
        CP(0.3, 2, 0),
        CZ(0, 2),
        SWAP(2, 0),
        TOF([2, 0], 1),
        CustomControlledGate([2, 0], Ry(0.7, 1)),
        CustomGate(np.array([[0, 1j], [1j, 0]]), [2]),
    ],
)
def test_translation_matches_circuit_matrix(gate: Gate):
    circuit = QCircuit([H(0), Ry(0.43, 1), Rx(0.27, 2), gate], nb_qubits=4)
    expected = circuit.to_matrix()[:, 0]
    original = deepcopy(circuit)
    actual = translated_matrix(circuit)[:, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert circuit == original


def test_global_phase_and_unused_qubits():
    circuit = QCircuit([X(0)], nb_qubits=3)
    circuit.input_g_phase = 0.37
    expected = np.zeros(8, dtype=complex)
    expected[4] = np.exp(0.37j)
    np.testing.assert_allclose(translated_matrix(circuit)[:, 0], expected, atol=1e-12)
    assert circuit.input_g_phase == 0.37


@pytest.mark.parametrize(
    'targets,c_targets,expected',
    [
        ([2, 0], None, '01'),
        ([2, 0], [3, 1], '01'),
        ([0, 2], [3, 1], '10'),
        ([2], [4], '0'),
    ],
)
def test_partial_and_reordered_sampling(
    cloud: Mock, targets: list[int], c_targets: Optional[list[int]], expected: str
):
    circuit = QCircuit(
        [X(0), BasisMeasure(targets, c_targets=c_targets, shots=32)], nb_qubits=3
    )
    original = deepcopy(circuit)
    raw = mc.QCSResults()
    measurement = circuit.measurements[0]
    assert isinstance(measurement, BasisMeasure) and measurement.c_targets is not None
    bits = [0] * (max(measurement.c_targets) + 1)
    for target, bit in zip(targets, measurement.c_targets):
        bits[bit] = int(target == 0)
    raw.cstates = [mc.BitString(bits) for _ in range(32)]
    raw.fidelities = [1.0]
    cloud.get_result.return_value = raw
    result = run(circuit, MIMIQDevice.CLOUD_STATEVECTOR)
    assert [(s.bin_str, s.count) for s in result.samples] == [(expected, 32)]
    assert sum(result.counts) == 32
    assert circuit == original


def test_dynamic_basis_change_spans_the_circuit():
    circuit = QCircuit(
        [H(2), H(0), BasisMeasure([2, 0], shots=16, basis=HadamardBasis())], nb_qubits=3
    )
    original = deepcopy(circuit)
    expected = np.zeros(8, dtype=np.complex128)
    expected[[0, 2]] = 1 / np.sqrt(2)
    np.testing.assert_allclose(translated_matrix(circuit)[:, 0], expected, atol=1e-12)
    assert circuit == original


def test_symbolic_binding_and_unbound_rejection(cloud: Mock):
    theta = symbols('theta')
    circuit = QCircuit([Ry(theta, 2), BasisMeasure(shots=8)], nb_qubits=3)
    raw = mc.QCSResults()
    raw.cstates = [mc.BitString([0, 0, 1]) for _ in range(8)]
    raw.fidelities = [1.0]
    cloud.get_result.return_value = raw
    result = run(
        circuit, MIMIQDevice.CLOUD_STATEVECTOR, values={theta: cast(Complex, np.pi)}
    )
    assert result.counts[1] == 8
    assert not cloud.execute.call_args.args[0].is_symbolic()
    assert theta in circuit.variables()
    with pytest.raises(ValueError, match='Bind all circuit parameters'):
        run(circuit, MIMIQDevice.CLOUD_STATEVECTOR)


def test_native_translation_and_zero_shots():
    circuit = QCircuit([Rxx(0.3, 2, 0), BasisMeasure(shots=0)], nb_qubits=4)
    translated = mpqp_to_mimiq(circuit, skip_measurements=True)
    assert isinstance(next(iter(translated)).operation, mc.GateRXX)
    assert translated.num_qubits() == 4
    assert not any(isinstance(i.operation, mc.Measure) for i in translated)
    assert isinstance(circuit.to_other_language(Language.MIMIQ), mc.Circuit)


def test_empty_circuit_with_declared_qubits():
    np.testing.assert_array_equal(
        translated_matrix(QCircuit(3))[:, 0], np.eye(1, 8, 0)[0]
    )
    with pytest.raises(ValueError, match='at least one qubit'):
        mpqp_to_mimiq(QCircuit(0))


@pytest.mark.parametrize('device', list(MIMIQDevice))
def test_capabilities_and_unsupported_observables(device: MIMIQDevice):
    assert device.is_remote()
    assert device.supports_samples()
    assert not device.supports_state_vector()
    circuit = QCircuit([ExpectationMeasure(Observable(np.diag([1, -1])))])
    with pytest.raises(DeviceJobIncompatibleError):
        run(circuit, device)


@pytest.mark.parametrize(
    'option,error',
    [
        ({'seed': True}, TypeError),
        ({'seed': -1}, ValueError),
        ({'bonddim': 0}, ValueError),
        ({'entdim': 65}, ValueError),
        ({'timelimit': 0}, ValueError),
    ],
)
def test_parameter_validation(option: dict[str, Any], error: type[Exception]):
    with pytest.raises(error):
        MimiqParams(**option)


def test_no_noise_is_silently_dropped(cloud: Mock):
    from mpqp import Depolarizing

    circuit = QCircuit([H(0), Depolarizing(0.1), BasisMeasure(shots=4)])
    with pytest.raises(DeviceJobIncompatibleError):
        run(circuit, MIMIQDevice.CLOUD_STATEVECTOR)
    with pytest.raises(NotImplementedError, match='noise translation'):
        mpqp_to_mimiq(circuit)


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> Mock:
    connection = create_autospec(mc.MimiqConnection, instance=True)
    connection.execute.return_value = 'mimiq-job-id'
    connection.requestInfo.return_value = {'_id': 'mimiq-job-id', 'status': 'NEW'}
    monkeypatch.setattr(mimiq, 'get_mimiq_connection', lambda: connection)
    return connection


def cloud_circuit() -> QCircuit:
    return QCircuit(
        [X(0), CNOT(0, 2), BasisMeasure([2, 1, 0], c_targets=[4, 0, 2], shots=8)],
        nb_qubits=4,
    )


def raw_samples() -> Any:
    raw = mc.QCSResults()
    # q0=q2=1 and q1=0, measured into c2=c4=1 and c0=0.
    raw.cstates = [mc.BitString([0, 0, 1, 0, 1]) for _ in range(8)]
    raw.fidelities = [1.0]
    return raw


def test_cloud_submission_status_and_result(cloud: Mock):
    circuit = cloud_circuit()
    cloud.get_result.return_value = raw_samples()
    job_id, job = submit(
        circuit, MIMIQDevice.CLOUD_MPS, provider_params=MimiqParams(seed=0, bonddim=32)
    )
    assert job_id == job.id == 'mimiq-job-id'
    assert job.status == JobStatus.QUEUED
    options = cloud.execute.call_args.kwargs
    assert (
        options['algorithm'] == 'mps'
        and options['seed'] == 0
        and options['bonddim'] == 32
    )
    cloud.requestInfo.return_value['status'] = 'RUNNING'
    assert job.status == JobStatus.RUNNING
    result = get_remote_result(job)
    assert [(s.bin_str, s.count) for s in result.samples] == [('101', 8)]
    assert job.status == JobStatus.DONE


def test_cloud_retrieval_after_restart(cloud: Mock):
    circuit = cloud_circuit()
    cloud.get_inputs.return_value = ([mpqp_to_mimiq(circuit)], {'nsamples': 8})
    cloud.get_result.return_value = raw_samples()
    result = get_remote_result('mimiq-job-id', MIMIQDevice.CLOUD_STATEVECTOR)
    assert [(s.bin_str, s.count) for s in result.samples] == [('101', 8)]
    assert result.job.id == 'mimiq-job-id'


def test_blocking_cloud_run(cloud: Mock):
    circuit = cloud_circuit()
    cloud.get_result.return_value = raw_samples()
    result = run(circuit, MIMIQDevice.CLOUD_STATEVECTOR)
    assert result.job.id == 'mimiq-job-id'
    assert cloud.execute.call_args.kwargs['algorithm'] == 'statevector'
    assert 'bonddim' not in cloud.execute.call_args.kwargs


@pytest.mark.parametrize(
    'status,expected',
    [
        ('NEW', JobStatus.QUEUED),
        ('RUNNING', JobStatus.RUNNING),
        ('DONE', JobStatus.DONE),
        ('ERROR', JobStatus.ERROR),
        ('CANCELED', JobStatus.CANCELLED),
    ],
)
def test_cloud_status_mapping(cloud: Mock, status: str, expected: JobStatus):
    cloud.requestInfo.return_value['status'] = status
    assert mimiq.get_mimiq_job_status('job-id') == expected


def test_unknown_cloud_status_is_rejected(cloud: Mock):
    cloud.requestInfo.return_value['status'] = 'UNKNOWN'
    with pytest.raises(RemoteExecutionError, match='Unrecognized'):
        mimiq.get_mimiq_job_status('job-id')


def test_unsupported_jobs_fail_before_cloud_submission(cloud: Mock):
    with pytest.raises(DeviceJobIncompatibleError):
        submit(QCircuit([X(0)]), MIMIQDevice.CLOUD_STATEVECTOR)
    circuit = QCircuit([BasisMeasure(shots=65537)], nb_qubits=1)
    with pytest.raises(ValueError, match='65536'):
        submit(circuit, MIMIQDevice.CLOUD_MPS)
    cloud.execute.assert_not_called()


@pytest.mark.parametrize('runner', [run, submit])
def test_other_provider_parameters_are_rejected(
    cloud: Mock, runner: Callable[..., Any]
):
    with pytest.raises(TypeError, match='MimiqParams'):
        runner(cloud_circuit(), MIMIQDevice.CLOUD_MPS, provider_params=QiskitParams())
    cloud.execute.assert_not_called()


def test_account_setup_is_never_implicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from mpqp.execution.connection import mimiq_connection

    mimiq_connection.get_mimiq_connection.cache_clear()
    monkeypatch.setattr(
        mimiq_connection, '_token_path', lambda: tmp_path / 'missing-token.json'
    )
    with pytest.raises(RuntimeError, match='No MIMIQ account configured'):
        mimiq_connection.get_mimiq_connection()


def test_truncated_results_warn(cloud: Mock):
    circuit = cloud_circuit()
    raw = raw_samples()
    raw.fidelities = [0.9]
    cloud.get_result.return_value = raw
    with pytest.warns(RuntimeWarning, match='truncation'):
        run(circuit, MIMIQDevice.CLOUD_MPS)


@pytest.mark.parametrize(
    'status,expected', [('ERROR', JobStatus.ERROR), ('CANCELED', JobStatus.CANCELLED)]
)
def test_remote_failure_preserves_status(cloud: Mock, status: str, expected: JobStatus):
    _, job = submit(cloud_circuit(), MIMIQDevice.CLOUD_MPS)
    cloud.requestInfo.return_value['status'] = status
    cloud.get_result.side_effect = RuntimeError('Remote job failed or was cancelled.')
    with pytest.raises(RuntimeError):
        get_remote_result(job)
    assert job.status == expected


def test_session_can_retry_result_download(cloud: Mock):
    _, job = submit(cloud_circuit(), MIMIQDevice.CLOUD_MPS)
    cloud.requestInfo.return_value['status'] = 'DONE'
    cloud.get_result.side_effect = [
        ConnectionError('Connection interrupted.'),
        raw_samples(),
    ]
    with pytest.raises(ConnectionError):
        get_remote_result(job)
    assert get_remote_result(job).counts[5] == 8


def test_mpqp_import_does_not_load_optional_mimiq_sdks():
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'mimiqcircuits'}:
        raise RuntimeError('Optional SDK imported while loading MPQP: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from mpqp import MIMIQDevice, QCircuit
assert not MIMIQDevice.CLOUD_STATEVECTOR.supports_state_vector()
"""
    subprocess.run(
        [sys.executable, '-c', script], check=True, capture_output=True, timeout=30
    )


def test_saved_token_connection_is_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from mpqp.execution.connection import mimiq_connection

    token_path = tmp_path / 'token.json'
    token_path.write_text('{}', encoding='utf-8')
    factory = Mock()
    monkeypatch.setattr(mc, 'MimiqConnection', factory)
    monkeypatch.setattr(mimiq_connection, '_token_path', lambda: token_path)
    mimiq_connection.get_mimiq_connection.cache_clear()
    try:
        assert (
            mimiq_connection.get_mimiq_connection()
            is mimiq_connection.get_mimiq_connection()
        )
        factory.return_value.loadtoken.assert_called_once_with(str(token_path))
        factory.return_value.connect.assert_not_called()
    finally:
        mimiq_connection.get_mimiq_connection.cache_clear()


def test_incomplete_cloud_samples_are_rejected(cloud: Mock):
    raw = raw_samples()
    raw.cstates.pop()
    cloud.get_result.return_value = raw
    with pytest.raises(RuntimeError, match='different number of samples'):
        run(cloud_circuit(), MIMIQDevice.CLOUD_STATEVECTOR)


def test_barrier_preserves_quantum_state():
    from mpqp import Barrier

    circuit = QCircuit([H(0), Barrier(), CNOT(0, 1)])
    np.testing.assert_allclose(
        translated_matrix(circuit)[:, 0],
        np.array([1, 0, 0, 1]) / np.sqrt(2),
        atol=1e-12,
    )


def test_shared_account_configuration_without_sdk(monkeypatch: pytest.MonkeyPatch):
    from mpqp.execution.connection import mimiq_connection

    connection = Mock()
    monkeypatch.setattr(mimiq_connection, 'get_env_variable', lambda _: 'True')
    monkeypatch.setattr(mimiq_connection, 'find_spec', lambda _: None)
    monkeypatch.setattr(mimiq_connection, 'get_mimiq_connection', connection)
    assert mimiq_connection.get_all_job_ids() == []
    connection.assert_not_called()


def test_job_listing_uses_sdk_request_dictionaries(monkeypatch: pytest.MonkeyPatch):
    from mpqp.execution.connection import mimiq_connection

    connection = create_autospec(mc.MimiqConnection, instance=True)
    connection.requests.return_value = [
        {'_id': 'first-job', 'status': 'DONE'},
        {'_id': 'second-job', 'status': 'RUNNING'},
    ]
    monkeypatch.setattr(mimiq_connection, 'get_env_variable', lambda _: 'True')
    monkeypatch.setattr(mimiq_connection, 'get_mimiq_connection', lambda: connection)
    assert mimiq_connection.get_all_job_ids() == ['first-job', 'second-job']


@pytest.mark.parametrize('controlled', [False, True])
def test_cloud_recovery_preserves_custom_gate_matrix(cloud: Mock, controlled: bool):
    gate = CustomGate(np.array([[0, 1j], [1j, 0]]), [2])
    operation = CustomControlledGate([0], gate) if controlled else gate
    circuit = QCircuit([H(0), operation, BasisMeasure(shots=8)], nb_qubits=3)
    cloud.get_inputs.return_value = ([mpqp_to_mimiq(circuit)], {'nsamples': 8})
    raw = mc.QCSResults()
    raw.cstates = [mc.BitString([0, 0, 0]) for _ in range(8)]
    raw.fidelities = [1.0]
    cloud.get_result.return_value = raw
    result = get_remote_result('custom-job', MIMIQDevice.CLOUD_STATEVECTOR)
    assert isinstance(result.job.circuit, QCircuit)
    recovered = next(
        i for i in reversed(result.job.circuit.instructions) if isinstance(i, Gate)
    )
    assert recovered.targets == [2]
    if controlled:
        assert isinstance(recovered, CustomControlledGate)
        assert recovered.controls == [0]
        recovered = recovered.non_controlled_gate
    assert isinstance(recovered, CustomGate)
    np.testing.assert_array_equal(
        recovered.to_canonical_matrix(), np.array([[0, 1j], [1j, 0]])
    )


def test_mimiq_exposes_only_cloud_devices():
    assert {device.value for device in MIMIQDevice} == {
        'cloud-statevector',
        'cloud-mps',
    }
