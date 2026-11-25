"""Add ``--long`` to the CLI arguments to run the remote tests (disabled by
default because they require a configured Quantinuum Nexus account and project).
"""

import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from pytket.backends.backendresult import BackendResult
from pytket.circuit import Circuit as tket_Circuit
from sympy import Symbol

from mpqp import (
    CNOT,
    BasisMeasure,
    BindingMode,
    CircuitBinding,
    ExpectationMeasure,
    H,
    IBMDevice,
    Observable,
    QCircuit,
    QUANTINUUMDevice,
    Ry,
    X,
    get_remote_result,
    pX,
    pZ,
    run,
    submit,
)
from mpqp.core._types import BindingParameters
from mpqp.core.instruction.measurement.basis import HadamardBasis
from mpqp.core.instruction.measurement.pauli_string import CommutingTypes
from mpqp.execution.job import Job, JobStatus, JobType
from mpqp.execution.providers import quantinuum as quantinuum_provider
from mpqp.execution.providers.providers_params import QuantinuumParams
from mpqp.execution.providers.quantinuum import check_job_compatibility
from mpqp.execution.result import BatchResult, Result
from mpqp.tools.errors import DeviceJobIncompatibleError
from mpqp.tools.maths import matrix_eq

pytestmark = pytest.mark.provider("quantinuum")


def _mock_nexus_binding_submission(
    job: Job,
    _circuits: list[tket_Circuit],
    *_args: object,
    **_kwargs: object,
) -> SimpleNamespace:
    job.id = "nexus-binding-id"
    return SimpleNamespace(id=job.id)


def _sample_backend_result(outcome: tuple[int, ...], shots: int) -> BackendResult:
    result = MagicMock()
    result.get_counts.return_value = Counter({outcome: shots})
    return result


def _state_vector_backend_result(
    state_vector: npt.NDArray[np.complex128],
) -> BackendResult:
    result = MagicMock()
    result.get_state.return_value = state_vector
    return result


@pytest.mark.parametrize(
    "device",
    [
        QUANTINUUMDevice.TKET_AER_SIMULATOR,
        QUANTINUUMDevice.TKET_QULACS_SIMULATOR,
    ],
)
def test_local_tket_sample(device: QUANTINUUMDevice):
    result = run(
        QCircuit([X(0), BasisMeasure([0], shots=100)]),
        device,
    )

    assert result.counts == [0, 100]
    assert result.probabilities.tolist() == [0, 1]
    assert result.shots == 100


@pytest.mark.parametrize(
    "device",
    [
        QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
        QUANTINUUMDevice.TKET_QULACS_SIMULATOR,
    ],
)
def test_local_tket_state_vector(device: QUANTINUUMDevice):
    result = run(
        QCircuit([H(0), CNOT(0, 1)]),
        device,
    )

    expected = np.array([2**-0.5, 0, 0, 2**-0.5])
    assert matrix_eq(result.amplitudes, expected)
    assert matrix_eq(result.probabilities, np.abs(expected) ** 2)
    assert result.shots == 0


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (
            Job(JobType.STATE_VECTOR, QCircuit(1), IBMDevice.AER_SIMULATOR),
            "must correspond to a `QUANTINUUMDevice`",
        ),
        (
            Job(
                JobType.SAMPLE,
                QCircuit([ExpectationMeasure(Observable(pZ), shots=100)]),
                QUANTINUUMDevice.TKET_AER_SIMULATOR,
            ),
            "SAMPLE job requires",
        ),
        (
            Job(
                JobType.SAMPLE,
                QCircuit([BasisMeasure([0], shots=100)]),
                QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
            ),
            "does not support `SAMPLE` jobs",
        ),
        (
            Job(
                JobType.STATE_VECTOR,
                QCircuit(1),
                QUANTINUUMDevice.TKET_AER_SIMULATOR,
            ),
            "does not support `STATE_VECTOR` jobs",
        ),
        (
            Job(
                JobType.OBSERVABLE,
                QCircuit([ExpectationMeasure(Observable(pX), shots=0)]),
                QUANTINUUMDevice.H1_EMULATOR,
            ),
            "does not handle exact observable jobs",
        ),
        (
            Job(
                JobType.OBSERVABLE,
                QCircuit([ExpectationMeasure(Observable(pX), shots=100)]),
                QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
            ),
            "does not support sampled observable jobs",
        ),
        (
            Job(
                JobType.OBSERVABLE,
                QCircuit(
                    [
                        ExpectationMeasure(
                            Observable(pX),
                            shots=100,
                            commuting_type=CommutingTypes.FULL,
                        )
                    ]
                ),
                QUANTINUUMDevice.H1_EMULATOR,
            ),
            "qubit-wise commuting",
        ),
    ],
)
def test_job_compatibility(job: Job, message: str):
    with pytest.raises(DeviceJobIncompatibleError, match=message):
        check_job_compatibility(job)


@pytest.mark.parametrize(
    "device",
    [
        QUANTINUUMDevice.TKET_AER_SIMULATOR,
        QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
        QUANTINUUMDevice.TKET_QULACS_SIMULATOR,
    ],
)
def test_local_tket_exact_observables(device: QUANTINUUMDevice):
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                [
                    Observable(pX, label="X"),
                    Observable(np.diag([1, -1]), label="Z"),
                ],
                shots=0,
            ),
        ]
    )

    result = run(circuit, device)

    assert isinstance(result.expectation_values, dict)
    assert np.isclose(result.expectation_values["X"], 1)
    assert np.isclose(result.expectation_values["Z"], 0)
    assert result.error == {"X": 0, "Z": 0}
    assert result.shots == 0


@pytest.mark.parametrize(
    "device",
    [
        QUANTINUUMDevice.TKET_AER_SIMULATOR,
        QUANTINUUMDevice.TKET_QULACS_SIMULATOR,
    ],
)
def test_local_tket_sampled_observables(device: QUANTINUUMDevice):
    shots = 1000
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                Observable(pX + pZ),
                shots=shots,
            ),
        ]
    )

    result = run(circuit, device)

    assert isinstance(result.expectation_values, float)
    assert isinstance(result.error, float)
    assert np.isclose(result.expectation_values, 1, atol=0.15)

    sampled_z_expectation = result.expectation_values
    expected_variance = (1 - sampled_z_expectation**2) / shots

    assert np.isclose(result.error, expected_variance)
    assert result.shots == shots


def test_local_tket_sample_binding_preserves_unroll_order():
    repeated_circuit = QCircuit([X(0)])
    nested_binding = CircuitBinding([repeated_circuit, QCircuit(1)])
    binding = CircuitBinding(
        [nested_binding, repeated_circuit],
        measurements=BasisMeasure(shots=10),
    )

    batch = run(binding, QUANTINUUMDevice.TKET_AER_SIMULATOR)

    assert isinstance(batch, BatchResult)
    assert len(batch.results) == 3
    assert [result.counts for result in batch.results] == [
        [0, 10],
        [10, 0],
        [0, 10],
    ]


def test_local_tket_binding_compiles_symbolic_circuit_once():
    theta = Symbol("theta")
    binding = CircuitBinding(
        QCircuit([Ry(theta, 0)]),
        values=[{theta: 0.0}, {theta: np.pi}],
        measurements=BasisMeasure(shots=10),
    )
    backend_results = [
        _sample_backend_result((0,), 10),
        _sample_backend_result((1,), 10),
    ]

    symbolic_circuits: list[tket_Circuit] = []

    def compile_circuits(
        circuits: list[tket_Circuit],
        optimisation_level: int,
        allow_symbolic: bool,
    ) -> list[tket_Circuit]:
        assert optimisation_level == 0
        symbolic_circuits.extend(circuits)
        assert allow_symbolic
        assert len(circuits) == 1
        assert {str(symbol) for symbol in circuits[0].free_symbols()} == {"theta"}
        return circuits

    def run_bound_circuits(
        circuits: list[tket_Circuit], n_shots: int | list[int] | None
    ) -> list[BackendResult]:
        assert len(circuits) == 2
        assert all(circuit.free_symbols() == set() for circuit in circuits)
        assert n_shots == [10, 10]
        return backend_results

    backend = MagicMock()
    backend.get_compiled_circuits.side_effect = compile_circuits
    backend.run_circuits.side_effect = run_bound_circuits

    with patch.object(
        quantinuum_provider,
        "_get_local_tket_backend",
        return_value=backend,
    ):
        batch = run(binding, QUANTINUUMDevice.TKET_AER_SIMULATOR)

    assert backend.get_compiled_circuits.call_count == 1
    assert {str(symbol) for symbol in symbolic_circuits[0].free_symbols()} == {"theta"}
    assert [result.counts for result in batch.results] == [[10, 0], [0, 10]]


def test_local_tket_binding_handles_compilation_error():
    binding = CircuitBinding(
        QCircuit([X(0)]),
        measurements=BasisMeasure(shots=10),
    )
    job = Job(
        binding.job_type,
        binding,
        QUANTINUUMDevice.TKET_AER_SIMULATOR,
    )
    backend = MagicMock()
    backend.get_compiled_circuits.side_effect = RuntimeError("compilation failed")

    with (
        patch.object(
            quantinuum_provider,
            "_get_local_tket_backend",
            return_value=backend,
        ),
        pytest.raises(RuntimeError, match="compilation failed"),
    ):
        quantinuum_provider.run_quantinuum(job)

    assert job.status == JobStatus.ERROR
    assert job.status_message == "compilation failed"


def test_local_tket_binding_with_different_circuit_sizes():
    binding = CircuitBinding(
        [
            QCircuit([X(0)]),
            QCircuit([H(0), CNOT(0, 1)]),
        ]
    )

    batch = run(
        binding,
        QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
    )

    assert isinstance(batch, BatchResult)
    assert len(batch.results) == 2
    assert matrix_eq(batch.results[0].amplitudes, np.array([0, 1]))
    assert matrix_eq(
        batch.results[1].amplitudes,
        np.array([2**-0.5, 0, 0, 2**-0.5]),
    )
    assert [result.job.circuit.nb_qubits for result in batch.results] == [1, 2]


@pytest.mark.parametrize(
    ("mode", "expected_values"),
    [
        (BindingMode.ZIP, [1.0, 0.0]),
        (BindingMode.PRODUCT, [1.0, 0.0, -1.0, 0.0]),
    ],
)
def test_local_tket_observable_binding(mode: BindingMode, expected_values: list[float]):
    theta = Symbol("theta")
    binding = CircuitBinding(
        QCircuit([Ry(theta, 0)]),
        values=[{theta: 0.0}, {theta: np.pi}],
        measurements=[
            ExpectationMeasure(Observable(pZ)),
            ExpectationMeasure(Observable(pX)),
        ],
        mode=mode,
    )

    batch = run(binding, QUANTINUUMDevice.TKET_AER_SIMULATOR)

    assert isinstance(batch, BatchResult)
    assert [result.expectation_values for result in batch.results] == pytest.approx(
        expected_values
    )


def test_local_tket_binding_preserves_custom_basis_and_input_measurement():
    measurement = BasisMeasure(basis=HadamardBasis(), shots=10)
    binding = CircuitBinding(QCircuit([H(0)]), measurements=measurement)

    batch = run(binding, QUANTINUUMDevice.TKET_AER_SIMULATOR)

    assert batch.results[0].counts == [10, 0]
    assert measurement.targets == []
    assert batch.results[0].job.measure is not None
    assert batch.results[0].job.measure.targets == [0]


def test_nexus_sample_binding_submits_all_circuits_together():
    binding = CircuitBinding(
        [QCircuit([X(0)]), QCircuit(1)],
        measurements=BasisMeasure(shots=10),
    )
    backend_results = [
        _sample_backend_result((1,), 10),
        _sample_backend_result((0,), 10),
    ]

    with (
        patch.object(
            quantinuum_provider,
            "submit_tket_circuits_to_nexus",
            side_effect=_mock_nexus_binding_submission,
        ) as submit_batch,
        patch.object(
            quantinuum_provider,
            "fetch_nexus_results",
            return_value=backend_results,
        ) as fetch_results,
    ):
        batch = run(binding, QUANTINUUMDevice.NEXUS_AER_SIMULATOR)

    assert submit_batch.call_count == 1
    _, submitted_circuits, submitted_shot_counts = submit_batch.call_args.args
    assert len(submitted_circuits) == 2
    assert submitted_shot_counts == [10, 10]
    assert submit_batch.call_args.kwargs["parameter_bindings"] == [
        (0, None),
        (1, None),
    ]
    assert fetch_results.call_count == 1
    assert [result.counts for result in batch.results] == [
        [0, 10],
        [10, 0],
    ]
    assert [result.job.id for result in batch.results] == [
        "nexus-binding-id",
        "nexus-binding-id",
    ]


def test_nexus_parameterized_binding_compiles_once_and_binds_each_execution():
    from mpqp.translation.tket import mpqp_to_tket

    theta = Symbol("theta")
    binding = CircuitBinding(
        QCircuit([Ry(theta, 0)]),
        values=[{theta: 0.0}, {theta: np.pi}],
        measurements=BasisMeasure(shots=10),
    )
    symbolic_circuit = mpqp_to_tket(QCircuit([Ry(theta, 0), BasisMeasure(shots=10)]))

    compiled_circuit_ref = MagicMock()
    compiled_circuit_ref.download_circuit.return_value = symbolic_circuit
    compilation_result_ref = MagicMock()
    compilation_result_ref.job_item_integer_id = 0
    compilation_result_ref.get_output.return_value = compiled_circuit_ref

    symbolic_ref = MagicMock()
    bound_refs = [MagicMock(), MagicMock()]
    execute_job_ref = SimpleNamespace(id="nexus-binding-id")

    with (
        patch.object(
            quantinuum_provider,
            "get_quantinuum_config",
            return_value=MagicMock(),
        ),
        patch(
            "qnexus.circuits.upload",
            side_effect=[symbolic_ref, *bound_refs],
        ) as upload,
        patch("qnexus.start_compile_job", return_value=MagicMock()) as compile_job,
        patch("qnexus.jobs.wait_for"),
        patch("qnexus.jobs.results", return_value=[compilation_result_ref]),
        patch(
            "qnexus.start_execute_job",
            return_value=execute_job_ref,
        ) as execute_job,
    ):
        job_id, job = submit(
            binding,
            QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
        )

    assert job_id == "nexus-binding-id"
    assert job.id == job_id
    assert upload.call_count == 3
    assert compile_job.call_args.kwargs["programs"] == [symbolic_ref]
    assert compiled_circuit_ref.download_circuit.call_count == 1
    uploaded_circuits = [call.kwargs["circuit"] for call in upload.call_args_list]
    assert {str(symbol) for symbol in uploaded_circuits[0].free_symbols()} == {"theta"}
    assert all(circuit.free_symbols() == set() for circuit in uploaded_circuits[1:])
    assert [
        float(circuit.get_commands()[0].op.params[0])
        for circuit in uploaded_circuits[1:]
    ] == pytest.approx([0.0, 1.0])
    assert execute_job.call_args.kwargs["programs"] == bound_refs
    assert execute_job.call_args.kwargs["n_shots"] == [10, 10]
    assert execute_job.call_args.kwargs["description"] == "mpqp:circuit-binding"


@pytest.mark.parametrize(
    ("parameter_bindings", "expected_error", "message"),
    [
        ([(1, {"theta": 0.0})], ValueError, "circuit index 1 does not exist"),
        ([(0, {})], ValueError, "Missing values for circuit parameters.*theta"),
        ([(0, {"theta": 1j})], TypeError, "must be real numbers.*theta"),
    ],
)
def test_nexus_validates_parameters_before_submission(
    parameter_bindings: list[tuple[int, BindingParameters | None]],
    expected_error: type[Exception],
    message: str,
):
    theta = Symbol("theta")
    tket_circuit = tket_Circuit(1).Ry(theta, 0)
    job = Job(
        JobType.SAMPLE,
        QCircuit([Ry(theta, 0), BasisMeasure(shots=10)]),
        QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
    )

    with (
        patch.object(quantinuum_provider, "get_quantinuum_config") as get_config,
        patch("qnexus.circuits.upload") as upload,
        patch("qnexus.start_compile_job") as compile_job,
        pytest.raises(expected_error, match=message),
    ):
        quantinuum_provider.submit_tket_circuits_to_nexus(
            job,
            [tket_circuit],
            10,
            name="binding",
            parameter_bindings=parameter_bindings,
        )

    get_config.assert_not_called()
    upload.assert_not_called()
    compile_job.assert_not_called()


@pytest.mark.parametrize(
    ("circuits", "parameter_bindings", "nexus_job_type"),
    [
        ([tket_Circuit(1)] * 301, None, "compilation"),
        ([tket_Circuit(1)], [(0, None)] * 301, "execution"),
    ],
)
def test_nexus_submission_rejects_more_than_300_programs(
    circuits: list[tket_Circuit],
    parameter_bindings: list[tuple[int, BindingParameters | None]] | None,
    nexus_job_type: str,
):
    job = Job(
        JobType.SAMPLE,
        QCircuit([X(0), BasisMeasure(shots=10)]),
        QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
    )

    with (
        patch.object(quantinuum_provider, "get_quantinuum_config") as get_config,
        patch("qnexus.circuits.upload") as upload,
        pytest.raises(
            ValueError,
            match=f"{nexus_job_type} job contains 301",
        ),
    ):
        quantinuum_provider.submit_tket_circuits_to_nexus(
            job,
            circuits,
            10,
            name="binding",
            parameter_bindings=parameter_bindings,
        )

    get_config.assert_not_called()
    upload.assert_not_called()


def test_nexus_binding_retrieval_uses_submitted_executions():
    binding = CircuitBinding(
        [QCircuit([X(0)]), QCircuit(1)],
        measurements=BasisMeasure(shots=10),
    )
    backend_results = [
        _sample_backend_result((1,), 10),
        _sample_backend_result((0,), 10),
    ]

    nexus_job_ref = SimpleNamespace(
        annotations=SimpleNamespace(description="mpqp:circuit-binding")
    )
    with patch.object(
        quantinuum_provider,
        "submit_tket_circuits_to_nexus",
        side_effect=_mock_nexus_binding_submission,
    ):
        job_id, job = submit(binding, QUANTINUUMDevice.NEXUS_AER_SIMULATOR)

    assert job_id == "nexus-binding-id"
    binding.circuits.append(QCircuit(1))
    with (
        patch("qnexus.jobs.get", return_value=nexus_job_ref),
        patch.object(
            quantinuum_provider,
            "fetch_nexus_results",
            return_value=backend_results,
        ),
    ):
        batch = get_remote_result(job)

    assert isinstance(batch, BatchResult)
    assert [result.counts for result in batch.results] == [[0, 10], [10, 0]]


def test_nexus_binding_retrieval_requires_original_job():
    nexus_job_ref = SimpleNamespace(
        annotations=SimpleNamespace(description="mpqp:circuit-binding")
    )

    with (
        patch("qnexus.jobs.get", return_value=nexus_job_ref),
        pytest.raises(ValueError, match="requires the original MPQP `Job`"),
    ):
        get_remote_result(
            "nexus-binding-id",
            QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
        )


def test_nexus_result_items_are_sorted_before_download():
    result_refs = [
        SimpleNamespace(
            job_item_integer_id=job_item_number,
            download_result=MagicMock(return_value=result),
        )
        for job_item_number, result in [
            (2, "two"),
            (None, "first result without id"),
            (0, "zero"),
            (1, "one"),
            (None, "second result without id"),
        ]
    ]

    with (
        patch("qnexus.jobs.wait_for"),
        patch("qnexus.jobs.results", return_value=result_refs),
    ):
        results = quantinuum_provider.fetch_nexus_results(MagicMock(id="job-id"))

    assert results == [
        "zero",
        "one",
        "two",
        "first result without id",
        "second result without id",
    ]


def test_nexus_binding_with_different_circuit_sizes():
    binding = CircuitBinding(
        [
            QCircuit([X(0)]),
            QCircuit([H(0), CNOT(0, 1)]),
        ]
    )
    backend_results = [
        _state_vector_backend_result(np.array([0, 1])),
        _state_vector_backend_result(np.array([2**-0.5, 0, 0, 2**-0.5])),
    ]

    with (
        patch.object(
            quantinuum_provider,
            "submit_tket_circuits_to_nexus",
            side_effect=_mock_nexus_binding_submission,
        ),
        patch.object(
            quantinuum_provider,
            "fetch_nexus_results",
            return_value=backend_results,
        ),
    ):
        batch = run(
            binding,
            QUANTINUUMDevice.NEXUS_AER_STATEVECTOR_SIMULATOR,
        )

    assert [result.job.circuit.nb_qubits for result in batch.results] == [1, 2]
    assert matrix_eq(batch.results[0].amplitudes, np.array([0, 1]))
    assert matrix_eq(
        batch.results[1].amplitudes,
        np.array([2**-0.5, 0, 0, 2**-0.5]),
    )


def test_nexus_observable_binding_maps_group_results():
    theta = Symbol("theta")
    binding = CircuitBinding(
        QCircuit([Ry(theta, 0)]),
        values=[{theta: 0.0}, {theta: np.pi}],
        measurements=ExpectationMeasure(
            [
                Observable(pZ, label="Z"),
                Observable(pX, label="X"),
            ],
            shots=10,
            optimize_measurement=True,
        ),
    )
    backend_results: list[BackendResult] = []

    def submit_observable_binding(
        job: Job,
        circuits: list[tket_Circuit],
        n_shots: int | list[int] | list[None],
        name: str,
        description: str = "",
        provider_params: QuantinuumParams | None = None,
        parameter_bindings: list[tuple[int, BindingParameters | None]] | None = None,
    ) -> SimpleNamespace:
        assert n_shots == [10, 10, 10, 10]
        assert parameter_bindings is not None
        for circuit_index, values in parameter_bindings:
            assert values is not None
            operations = [
                command.op.type.name
                for command in circuits[circuit_index].get_commands()
            ]
            if "H" in operations:
                counts = Counter({(0,): 5, (1,): 5})
            else:
                parameter_values = {str(key): value for key, value in values.items()}
                theta_value = parameter_values["theta"]
                assert theta_value in (0.0, np.pi)
                state = (0,) if theta_value == 0 else (1,)
                counts = Counter({state: 10})
            backend_result = MagicMock()
            backend_result.get_counts.return_value = counts
            backend_results.append(backend_result)
        return _mock_nexus_binding_submission(job, circuits)

    def fetch_results(_execute_job_ref: object) -> list[BackendResult]:
        return backend_results

    with (
        patch.object(
            quantinuum_provider,
            "submit_tket_circuits_to_nexus",
            side_effect=submit_observable_binding,
        ) as submit_batch,
        patch.object(
            quantinuum_provider,
            "fetch_nexus_results",
            side_effect=fetch_results,
        ),
        patch.object(
            QCircuit,
            "subs",
            side_effect=AssertionError(
                "Quantinuum bindings must bind compiled TKET circuits."
            ),
        ),
        pytest.warns(UserWarning, match="one sample circuit per Pauli group"),
    ):
        batch = run(binding, QUANTINUUMDevice.NEXUS_AER_SIMULATOR)

    assert submit_batch.call_count == 1
    _, submitted_circuits, submitted_shot_counts = submit_batch.call_args.args
    assert len(submitted_circuits) == 2
    assert all(len(circuit.free_symbols()) != 0 for circuit in submitted_circuits)
    assert submitted_shot_counts == [10, 10, 10, 10]
    assert submit_batch.call_args.kwargs["parameter_bindings"] == [
        (0, {theta: 0.0}),
        (1, {theta: 0.0}),
        (0, {theta: np.pi}),
        (1, {theta: np.pi}),
    ]
    assert [result.expectation_values for result in batch.results] == [
        {"Z": 1, "X": 0},
        {"Z": -1, "X": 0},
    ]


@pytest.mark.parametrize(
    ("measurement", "message"),
    [
        (
            ExpectationMeasure(Observable(pZ), shots=0),
            "does not handle exact observable jobs",
        ),
        (
            ExpectationMeasure(
                Observable(pZ),
                shots=10,
                optimize_measurement=False,
            ),
            "require `optimize_measurement=True`",
        ),
    ],
)
def test_nexus_binding_rejects_unsupported_observable_measurements(
    measurement: ExpectationMeasure,
    message: str,
):
    binding = CircuitBinding(
        QCircuit(1),
        measurements=measurement,
    )

    with (
        patch.object(binding, "to_other_device") as translate,
        pytest.raises(
            DeviceJobIncompatibleError,
            match=message,
        ),
    ):
        run(binding, QUANTINUUMDevice.NEXUS_AER_SIMULATOR)

    translate.assert_not_called()


@pytest.mark.skipif(
    "--long" not in sys.argv,
    reason="requires a configured Quantinuum Nexus account and project",
)
def test_remote_quantinuum_jobs():
    sample_result = run(
        QCircuit([X(0), BasisMeasure([0], shots=100)]),
        QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
    )
    assert isinstance(sample_result.job.id, str)
    assert sample_result.job.id
    assert sample_result.counts == [0, 100]

    state_vector_result = run(
        QCircuit([H(0), CNOT(0, 1)]),
        QUANTINUUMDevice.NEXUS_AER_STATEVECTOR_SIMULATOR,
    )
    assert matrix_eq(
        state_vector_result.amplitudes,
        np.array([2**-0.5, 0, 0, 2**-0.5]),
    )

    sampled_circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(Observable(pX + 2 * pZ), shots=100),
        ]
    )
    job_id, sampled_job = submit(
        sampled_circuit,
        QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
    )
    assert sampled_job.id == job_id

    sampled_result = get_remote_result(sampled_job)
    assert isinstance(sampled_result, Result)
    assert sampled_result.job.id == job_id
    assert isinstance(sampled_result.expectation_values, float)
    assert np.isclose(sampled_result.expectation_values, 1, atol=0.3)

    theta = Symbol("theta")
    binding = CircuitBinding(
        QCircuit([Ry(theta, 0)]),
        values=[{theta: 0.0}, {theta: np.pi}],
        measurements=BasisMeasure(shots=10),
    )
    binding_job_id, binding_job = submit(
        binding,
        QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
    )

    binding_result = get_remote_result(binding_job)

    assert isinstance(binding_result, BatchResult)
    assert binding_job.id == binding_job_id
    assert [result.job.id for result in binding_result.results] == [
        binding_job_id,
        binding_job_id,
    ]
    assert [result.counts for result in binding_result.results] == [
        [10, 0],
        [0, 10],
    ]
