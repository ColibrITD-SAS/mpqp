"""Exercise remote sampling bindings locally, without credentials or submission."""

from types import SimpleNamespace

import numpy as np
import pytest
from sympy import Symbol

from mpqp import BasisMeasure, BindingMode, CircuitBinding, IBMDevice, QCircuit
from mpqp.execution.job import ExecutionMode, Job, JobType
from mpqp.execution.providers import ibm
from mpqp.execution.runner import run
from mpqp.gates import Ry, X

pytestmark = pytest.mark.provider("qiskit")


@pytest.mark.parametrize("mode", list(ExecutionMode))
@pytest.mark.parametrize("binding_mode", list(BindingMode))
def test_remote_sample_binding(monkeypatch, mode, binding_mode):
    from qiskit.primitives import StatevectorSampler
    import qiskit_ibm_runtime
    import mpqp.execution.connection.ibm_connection as connection

    theta = Symbol("theta")
    circuit = QCircuit([X(0), Ry(theta, 2)])
    measure = BasisMeasure([2, 0], shots=32)
    binding = CircuitBinding(
        circuit,
        values=[{theta: 0.0}, {"theta": np.pi}],
        measurements=[measure, measure] if binding_mode == BindingMode.ZIP else measure,
        mode=binding_mode,
    )
    backend = SimpleNamespace(name="ibm_fez")
    session = object()
    submitted = []
    translations = []

    def translate(circuit, device):
        from mpqp.core.languages import Language

        translations.append(circuit)
        return circuit.to_other_language(Language.QISKIT)

    class LocalSampler:
        def __init__(self, mode):
            assert mode is (
                session if mode_expected == ExecutionMode.SESSION else backend
            )

        def run(self, pubs, shots):
            submitted.append((pubs, shots))
            result = StatevectorSampler(seed=123).run(pubs, shots=shots).result()
            return SimpleNamespace(job_id=lambda: "local-only", result=lambda: result)

    mode_expected = mode
    monkeypatch.setattr(QCircuit, "to_other_device", translate)
    monkeypatch.setattr(ibm, "get_backend", lambda *args: backend)
    monkeypatch.setattr(connection, "get_or_create_ibm_session", lambda *args: session)
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", LocalSampler)
    result = run(binding, IBMDevice.IBM_FEZ, mode=mode)
    assert len(submitted) == 1
    assert len(translations) == 1
    assert submitted[0][1] == 32
    assert len(result.results) == 2
    assert [r.counts for r in result.results] == [
        [0, 32, 0, 0],
        [0, 0, 0, 32],
    ]
    assert all(r.shots == 32 for r in result.results)
    assert all(s.nb_qubits == 2 for r in result.results for s in r.samples)
    assert circuit.variables() == {theta}
    assert circuit.measurements == []
    assert circuit.transpiled_circuit == {}


def test_sample_binding_requires_all_parameters():
    theta = Symbol("theta")
    binding = CircuitBinding(
        QCircuit([Ry(theta, 0)]), measurements=BasisMeasure([0], shots=32)
    )
    with pytest.raises(ValueError, match="Missing IBM sample parameters"):
        ibm._prepare_ibm_binding(Job(JobType.SAMPLE, binding, IBMDevice.AER_SIMULATOR))


def test_remote_binding_rejects_statevector():
    binding = CircuitBinding(QCircuit([X(0)]))
    with pytest.raises(NotImplementedError, match="observable and sample"):
        ibm._prepare_ibm_binding(Job(JobType.STATE_VECTOR, binding, IBMDevice.IBM_FEZ))


def test_sample_binding_mixed_registers_and_numeric_circuits():
    from qiskit.primitives import StatevectorSampler

    binding = CircuitBinding(
        [QCircuit([X(0)]), QCircuit([X(2)])],
        measurements=[BasisMeasure([0], shots=16), BasisMeasure([0, 1, 2], shots=16)],
        mode=BindingMode.ZIP,
    )
    prepared = ibm._prepare_ibm_binding(
        Job(JobType.SAMPLE, binding, IBMDevice.AER_SIMULATOR)
    )
    raw = StatevectorSampler(seed=123).run(prepared.pubs).result()
    results = ibm._extract_ibm_binding_results(raw, prepared, IBMDevice.AER_SIMULATOR)
    assert [r.counts for r in results.results] == [
        [0, 16],
        [0, 16, 0, 0, 0, 0, 0, 0],
    ]


def test_sample_binding_rejects_zero_shots():
    from mpqp.tools.errors import DeviceJobIncompatibleError

    binding = CircuitBinding(QCircuit([X(0)]), measurements=BasisMeasure([0], shots=0))
    with pytest.raises(DeviceJobIncompatibleError, match="positive shot count"):
        ibm._prepare_ibm_binding(Job(JobType.SAMPLE, binding, IBMDevice.IBM_FEZ))
