from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import PubResult
from qiskit.primitives.containers import DataBin
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2, SamplerV2

from mpqp import (
    BatchResult,
    CircuitBinding,
    IBMDevice,
    Job,
    JobType,
    QCircuit,
    Result,
    run,
)
from mpqp.execution._execution_context import ExecutionContext
from mpqp.execution.job import JobStatus
from mpqp.execution.providers import ibm as ibm_provider
from mpqp.execution.providers.ibm import extract_result
from mpqp.execution.providers.providers_params import QiskitParams
from mpqp.gates import *
from mpqp.measures import *
from mpqp.tools.errors import IBMRemoteExecutionError


class _FakeQctrlRun:
    def __init__(self, result: Any, status: str = "DONE"):
        self.job_id = "qctrl-job-id"
        self._result = result
        self._status = status

    def result(self):
        return self._result

    def status(self):
        return self._status


class _FakeQctrlFunction:
    provider = "q-ctrl"

    def __init__(self, run: _FakeQctrlRun):
        self._run = run
        self.run_kwargs: dict[str, Any] | None = None

    def run(self, **kwargs: Any):
        self.run_kwargs = kwargs
        return self._run


class _FakeRemoteDevice:
    value = "ibm_fake"

    def is_simulator(self):
        return False


class _FakeBinding:
    shots = 256

    def __init__(self, pubs_with_context: list[tuple[Any, list[Any]]]):
        self._pubs_with_context = pubs_with_context


def test_qiskit_params_copy_options():
    assert QiskitParams().get_options() == {}

    options: dict[str, Any] = {
        "environment": {"job_tags": ["experiment"], "private": True}
    }
    params = QiskitParams(options=options)
    options["environment"]["job_tags"].append("mutated")

    assert params.options == {
        "environment": {"job_tags": ["experiment"], "private": True}
    }

    returned = params.get_options()
    returned["environment"]["job_tags"].append("sdk-mutation")
    assert params.options == {
        "environment": {"job_tags": ["experiment"], "private": True}
    }


def test_adapt_qctrl_options():
    function: Any = _FakeQctrlFunction(_FakeQctrlRun(object()))

    assert QiskitParams(
        qctrl_function=function,
        options={
            "environment": {
                "job_tags": ["experiment"],
                "private": False,
                "log_level": "INFO",
            },
            "session_id": "session-id",
        },
    ).get_options() == {
        "job_tags": ["experiment"],
        "private": False,
        "log_level": "INFO",
        "session_id": "session-id",
    }

    with pytest.raises(ValueError, match="Conflicting option 'job_tags'"):
        QiskitParams(
            qctrl_function=function,
            options={
                "job_tags": ["top-level"],
                "environment": {"job_tags": ["environment"]},
            },
        ).get_options()


def test_extract_estimator_pubs_result_uses_execution_context():
    job = Job(
        JobType.OBSERVABLE,
        QCircuit([X(0), ExpectationMeasure(Observable(pZ), shots=1000)]),
        IBMDevice.AER_SIMULATOR,
    )
    pub_result = PubResult(
        DataBin(
            evs=np.array([-1.0]),
            stds=np.array([0.1]),
            shape=(1,),
        ),
        metadata={"shots": 1000},
    )

    extract_pubs = ibm_provider._extract_estimator_pubs_result  # pyright: ignore
    result = extract_pubs([pub_result], job, [[ExecutionContext(job, 0)]])

    assert isinstance(result, Result)
    assert result.expectation_values == pytest.approx(-1.0)
    assert result.error == pytest.approx(0.1)


def test_qctrl_observable_uses_function_result(monkeypatch: pytest.MonkeyPatch):
    contexts = [object()]
    circuit = _FakeBinding([("estimator-pub", contexts)])
    qctrl_result = object()
    expected = object()
    function: Any = _FakeQctrlFunction(_FakeQctrlRun(qctrl_result))
    params = QiskitParams(
        qctrl_function=function,
        options={"environment": {"job_tags": ["experiment"]}},
    )
    job: Any = type(
        "FakeJob",
        (),
        {
            "circuit": circuit,
            "device": _FakeRemoteDevice(),
            "job_type": JobType.OBSERVABLE,
            "status": JobStatus.INIT,
            "id": None,
        },
    )()

    monkeypatch.setattr(ibm_provider, "check_job_compatibility", lambda _job: None)
    monkeypatch.setattr(
        "mpqp.translation.qiskit.circuitbinding_to_pubs",
        lambda binding, device, *, transpile: binding._pubs_with_context,
    )

    def extract(result: Any, root_job: Any, pubs_contexts: Any):
        assert result is qctrl_result
        assert root_job is job
        assert pubs_contexts == [contexts]
        return expected

    monkeypatch.setattr(ibm_provider, "_extract_estimator_pubs_result", extract)

    assert ibm_provider.run_qctrl_function(job, params) is expected
    assert function.run_kwargs == {
        "primitive": "estimator",
        "pubs": ["estimator-pub"],
        "backend_name": "ibm_fake",
        "options": {"default_shots": 256, "job_tags": ["experiment"]},
    }
    assert job.id == "qctrl-job-id"
    assert job.status == JobStatus.DONE


def test_qctrl_error_status_is_not_reported_as_done(
    monkeypatch: pytest.MonkeyPatch,
):
    function: Any = _FakeQctrlFunction(_FakeQctrlRun("provider error", status="ERROR"))
    params = QiskitParams(qctrl_function=function)
    circuit = type(
        "FakeCircuit", (), {"to_other_language": lambda self, language: object()}
    )()
    measure = type("FakeMeasure", (), {"shots": 32})()
    job: Any = type(
        "FakeJob",
        (),
        {
            "circuit": circuit,
            "measure": measure,
            "device": _FakeRemoteDevice(),
            "job_type": JobType.SAMPLE,
            "status": JobStatus.INIT,
            "id": None,
        },
    )()

    monkeypatch.setattr(ibm_provider, "check_job_compatibility", lambda _job: None)

    with pytest.raises(IBMRemoteExecutionError, match="failed"):
        ibm_provider.run_qctrl_function(job, params)

    assert job.status == JobStatus.ERROR


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("execution", ["sampler", "estimator", "binding"])
def test_qctrl_submits_logical_circuits(
    monkeypatch: pytest.MonkeyPatch, execution: str
):
    class SubmissionCaptured(Exception):
        pass

    def capture_submission(**kwargs: Any):
        pub = kwargs["pubs"][0]
        circuit = pub[0]
        assert circuit.num_qubits == 2
        assert circuit.layout is None
        state = Statevector.from_instruction(circuit.remove_final_measurements(False))
        if execution == "sampler":
            assert circuit.count_ops()["measure"] == 2
            assert state.probabilities_dict() == {"01": 1.0}
        else:
            observable = pub[1][0][0]
            assert observable.num_qubits == 2
            assert state.expectation_value(observable) == pytest.approx(-1.0)
        raise SubmissionCaptured

    def unexpected_backend(*args: Any, **kwargs: Any):
        pytest.fail("Q-CTRL submission must not fetch a hardware backend.")

    monkeypatch.setattr(
        "mpqp.execution.connection.ibm_connection.get_backend", unexpected_backend
    )
    function: Any = _FakeQctrlFunction(_FakeQctrlRun(object()))
    monkeypatch.setattr(function, "run", capture_submission)
    params = QiskitParams(qctrl_function=function)
    measurement = (
        BasisMeasure([0, 1], shots=100)
        if execution == "sampler"
        else ExpectationMeasure(Observable(pZ @ pI), shots=100)
    )
    circuit = QCircuit([X(0), measurement], nb_qubits=2)
    if execution == "binding":
        # A cached hardware observable must not replace its logical counterpart.
        assert isinstance(measurement, ExpectationMeasure)
        monkeypatch.setattr(
            measurement.observables[0], "pre_transpiled", SparsePauliOp("III")
        )
        execution_input = CircuitBinding(circuit)
    else:
        execution_input = circuit

    with pytest.raises(SubmissionCaptured):
        run(execution_input, IBMDevice.IBM_FEZ, provider_params=params)


@pytest.mark.provider("qiskit")
def test_extract_result_accepts_estimator_primitive_result():
    """Remote jobs (``RuntimeJobV2.result()``) return a ``PrimitiveResult``,
    with or without an MPQP job attached.
    This test produces locally a PrimitiveResult and pass to extract result
    to test that the extraction is happening correctly."""

    qc = QuantumCircuit(1)
    qc.x(0)
    backend = AerSimulator()
    primitive_result = (
        EstimatorV2(mode=backend)
        .run([(transpile(qc, backend), [SparsePauliOp("Z")])])
        .result()
    )
    job = Job(
        JobType.OBSERVABLE,
        QCircuit([X(0), ExpectationMeasure(Observable(pZ), shots=1000)]),
        IBMDevice.AER_SIMULATOR,
    )

    for attached_job in (job, None):  # None: path of get_result_from_ibm_job_id
        result = extract_result(primitive_result, attached_job, IBMDevice.AER_SIMULATOR)
        assert isinstance(result, Result)
        assert result.expectation_values == pytest.approx(-1.0)


@pytest.mark.provider("qiskit")
def test_extract_result_accepts_sampler_primitive_result():

    qc = QuantumCircuit(1, 1)  # classical register named "c", as produced by MPQP
    qc.x(0)
    qc.measure(0, 0)
    backend = AerSimulator()
    primitive_result = (
        SamplerV2(mode=backend).run([transpile(qc, backend)], shots=100).result()
    )
    job = Job(
        JobType.SAMPLE,
        QCircuit([X(0), BasisMeasure([0], shots=100)]),
        IBMDevice.AER_SIMULATOR,
    )
    result = extract_result(primitive_result, job, IBMDevice.AER_SIMULATOR)
    assert isinstance(result, Result)
    assert result.counts == [0, 100]


@pytest.mark.provider("qiskit")
def test_extract_result_accepts_1d_pub_result_with_multiple_measurements():
    job = Job(
        JobType.OBSERVABLE,
        QCircuit(
            [
                X(0),
                ExpectationMeasure(Observable(pZ), shots=1000),
                ExpectationMeasure(Observable(pZ), shots=1000),
            ]
        ),
        IBMDevice.AER_SIMULATOR,
    )
    pub_result = PubResult(
        DataBin(
            evs=np.array([-1.0, -1.0]),
            stds=np.array([0.1, 0.2]),
            shape=(2,),
        ),
        metadata={"shots": 1000},
    )

    result = extract_result(pub_result, job, IBMDevice.AER_SIMULATOR)

    assert isinstance(result, BatchResult)
    assert [item.error for item in result.results] == [0.1, 0.2]
