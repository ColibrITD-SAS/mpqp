from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import (
    PrimitiveResult,
    PubResult,
    SamplerPubResult,
    StatevectorEstimator,
)
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2, SamplerV2
from sympy import Symbol

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
    qiskit_circuit = QuantumCircuit(1)
    observable = SparsePauliOp("Z")
    circuit = _FakeBinding([((qiskit_circuit, [[observable]]), contexts)])
    qctrl_result = PrimitiveResult([])
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

    def extract(result: Any, root_job: Any, pubs_contexts: Any, *, fallback_shots: int):
        assert result is qctrl_result
        assert root_job is job
        assert pubs_contexts == [contexts]
        assert fallback_shots == 256
        return expected

    monkeypatch.setattr(ibm_provider, "_extract_estimator_pubs_result", extract)

    assert ibm_provider.run_qctrl_function(job, params) is expected
    assert function.run_kwargs == {
        "primitive": "estimator",
        "pubs": [(qiskit_circuit, [observable])],
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


def test_qctrl_rejects_sampling_binding_before_submission():
    function: Any = _FakeQctrlFunction(_FakeQctrlRun(object()))
    params = QiskitParams(qctrl_function=function)
    binding = CircuitBinding(QCircuit([X(0)]), measurements=BasisMeasure([0], shots=32))
    job = Job(JobType.SAMPLE, binding, IBMDevice.IBM_FEZ)

    with pytest.raises(NotImplementedError, match="CircuitBinding sampling jobs"):
        ibm_provider.run_qctrl_function(job, params)

    assert job.id is None
    assert function.run_kwargs is None


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("job_type", [JobType.SAMPLE, JobType.OBSERVABLE])
@pytest.mark.parametrize("payload", ["unexpected response", {"message": "unexpected"}])
def test_qctrl_unsupported_result_preserves_done_status(
    monkeypatch: pytest.MonkeyPatch, job_type: JobType, payload: Any
):
    function: Any = _FakeQctrlFunction(_FakeQctrlRun(payload))
    params = QiskitParams(qctrl_function=function)
    measure = (
        BasisMeasure([0], shots=100)
        if job_type == JobType.SAMPLE
        else ExpectationMeasure(Observable(pZ), shots=100)
    )
    job = Job(job_type, QCircuit([X(0), measure]), IBMDevice.IBM_FEZ)

    def unexpected_backend(*args: Any, **kwargs: Any):
        pytest.fail("This test must not access a remote backend.")

    monkeypatch.setattr(
        "mpqp.execution.connection.ibm_connection.get_backend", unexpected_backend
    )
    with pytest.raises(
        TypeError, match="unsupported result type.*expected PrimitiveResult"
    ):
        ibm_provider.run_ibm(job, params)

    assert job.status == JobStatus.DONE
    assert job.id == "qctrl-job-id"


@pytest.mark.provider("qiskit")
def test_qctrl_sampler_accepts_primitive_result(monkeypatch: pytest.MonkeyPatch):
    payload = PrimitiveResult(
        [SamplerPubResult(DataBin(c=BitArray.from_counts({"1": 100})))]
    )
    function: Any = _FakeQctrlFunction(_FakeQctrlRun(payload))
    params = QiskitParams(qctrl_function=function)

    def unexpected_backend(*args: Any, **kwargs: Any):
        pytest.fail("This test must not access a remote backend.")

    monkeypatch.setattr(
        "mpqp.execution.connection.ibm_connection.get_backend", unexpected_backend
    )
    result = run(
        QCircuit([X(0), BasisMeasure([0], shots=100)]),
        IBMDevice.IBM_FEZ,
        provider_params=params,
    )

    assert isinstance(result, Result)
    assert result.counts == [0, 100]
    assert result.job.status == JobStatus.DONE


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
            observable = pub[1][0]
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
@pytest.mark.parametrize(
    "execution", ["multiple_observables", "sweep", "multi_observable_sweep", "ragged"]
)
def test_qctrl_preserves_parameter_observable_pairs(
    monkeypatch: pytest.MonkeyPatch, execution: str
):
    def execute(**kwargs: Any):
        result = StatevectorEstimator().run(kwargs["pubs"]).result()
        for index in range(len(result)):
            # Isolate PUB pairing from Q-CTRL's separate metadata contract.
            result[index].metadata["shots"] = 100
        return _FakeQctrlRun(result)

    def unexpected_backend(*args: Any, **kwargs: Any):
        pytest.fail("This test must not access a remote backend.")

    monkeypatch.setattr(
        "mpqp.execution.connection.ibm_connection.get_backend", unexpected_backend
    )
    function: Any = _FakeQctrlFunction(_FakeQctrlRun(object()))
    monkeypatch.setattr(function, "run", execute)
    params = QiskitParams(qctrl_function=function)
    z = Observable(pZ, label="z")
    x = Observable(pX, label="x")
    theta = Symbol("theta")
    execution_input: QCircuit | CircuitBinding
    if execution == "multiple_observables":
        execution_input = QCircuit([X(0), ExpectationMeasure([z, x], shots=100)])
        expected = [{"z": -1.0, "x": 0.0}]
    elif execution == "ragged":
        execution_input = CircuitBinding(
            QCircuit([X(0)]),
            measurements=[
                ExpectationMeasure(z, shots=100),
                ExpectationMeasure([z, x], shots=100),
            ],
        )
        expected = [-1.0, {"z": -1.0, "x": 0.0}]
    else:
        observables = [z, x] if execution == "multi_observable_sweep" else [z]
        execution_input = CircuitBinding(
            QCircuit([Ry(theta, 0), ExpectationMeasure(observables, shots=100)]),
            values=[{theta: 0.0}, {theta: np.pi}],
        )
        expected = (
            [{"z": 1.0, "x": 0.0}, {"z": -1.0, "x": 0.0}]
            if execution == "multi_observable_sweep"
            else [1.0, -1.0]
        )

    result = run(execution_input, IBMDevice.IBM_FEZ, provider_params=params)
    results = result.results if isinstance(result, BatchResult) else [result]
    assert len(results) == len(expected)
    for item, expected_value in zip(results, expected):
        assert item.expectation_values == pytest.approx(expected_value)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("reported_shots", [None, 73])
@pytest.mark.parametrize("binding", [False, True])
def test_qctrl_estimator_shot_metadata(
    monkeypatch: pytest.MonkeyPatch, reported_shots: int | None, binding: bool
):
    metadata: dict[str, int | None] = {"precision": None}
    if reported_shots is not None:
        metadata["shots"] = reported_shots
    pub_result = PubResult(
        DataBin(evs=np.array([-1.0]), stds=np.array([0.1]), shape=(1,)),
        metadata=metadata.copy(),
    )
    function: Any = _FakeQctrlFunction(_FakeQctrlRun(PrimitiveResult([pub_result])))
    params = QiskitParams(qctrl_function=function)
    measure = ExpectationMeasure(Observable(pZ), shots=100)
    execution_input = (
        CircuitBinding(QCircuit([X(0)]), measurements=measure)
        if binding
        else QCircuit([X(0), measure])
    )

    def unexpected_backend(*args: Any, **kwargs: Any):
        pytest.fail("This test must not access a remote backend.")

    monkeypatch.setattr(
        "mpqp.execution.connection.ibm_connection.get_backend", unexpected_backend
    )
    result = run(execution_input, IBMDevice.IBM_FEZ, provider_params=params)
    item = result.results[0] if isinstance(result, BatchResult) else result

    assert item.expectation_values == pytest.approx(-1.0)
    assert item.shots == (reported_shots if reported_shots is not None else 100)
    assert function.run_kwargs["options"]["default_shots"] == 100
    assert pub_result.metadata == metadata


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
