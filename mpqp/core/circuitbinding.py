from enum import Enum, auto
from typing import TYPE_CHECKING, Any, overload
from typing import Optional, TypeVar, Union
from numbers import Complex
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.tools.generics import OneOrMany
from sympy import Expr
from mpqp.noise.noise_model import NoiseModel
from mpqp.execution.devices import AWSDevice, AvailableDevice, IBMDevice

if TYPE_CHECKING:
    from mpqp.execution.job import Job
    from qiskit_aer import AerSimulator
    from qiskit.primitives.containers import EstimatorPubLike
    from braket.program_sets import ProgramSet


BindingParameters = dict[Union["Expr", str], Union[Complex, float]]
BindingExecution = tuple[QCircuit, Optional[BindingParameters], Optional[Measure]]
BindingOptions = tuple[Optional[BindingParameters], Optional[Measure]]
_AxisElement = TypeVar("_AxisElement")


class BindingMode(Enum):
    """Control how a :class:`CircuitBinding` combines its execution axes.

    ``PRODUCT`` builds the Cartesian product of circuits, parameter sets and
    measurements. ``ZIP`` associates elements positionally and broadcasts an
    axis containing a single element to the length of the longest axis.
    """

    PRODUCT = auto()
    ZIP = auto()


class CircuitBinding:
    """Describe a batch of related circuit executions lazily.

    A binding groups one or more circuits with optional parameter sets and
    measurements. It can also contain nested bindings. The execution axes are
    resolved only when the binding is translated for a provider or passed to
    :func:`~mpqp.execution.runner.run`.

    Args:
        circuits: Circuit, nested binding, or sequence of either to execute.
        values: Parameter mapping or sequence of mappings to bind to the
            symbolic circuit parameters.
        measurements: Measurement or sequence of measurements to apply. A
            circuit that already contains measurements must not also receive
            this argument.
        mode: Rule used to combine circuits, values and measurements.
        noises: Noise models shared by every circuit in the binding.
        shots: Number of shots shared by every execution. When omitted, it is
            inferred from the measurements when possible.

    Raises:
        ValueError: If the execution axes are incompatible, circuits have
            incompatible job types, or measurements, noise and shots are
            specified inconsistently.

    Note:
        In :attr:`BindingMode.ZIP` mode, every non-singleton axis must have the
        same length. Singleton axes are broadcast when the binding is unrolled.
    """

    def __init__(
        self,
        circuits: OneOrMany["QCircuit | CircuitBinding"],
        values: Optional[OneOrMany[dict[str, float] | dict[Expr, float]]] = None,
        measurements: Optional[OneOrMany[Measure]] = None,
        mode: BindingMode = BindingMode.PRODUCT,
        noises: Optional[list[NoiseModel]] = None,
        shots: Optional[int] = None,
    ) -> None:
        from mpqp.execution.runner import adjust_measure
        from mpqp.execution.job import JobType
        from typing import TYPE_CHECKING, Sequence
        from mpqp.core.instruction.measurement import (
            Measure,
            ExpectationMeasure,
            BasisMeasure,
        )

        if mode == BindingMode.ZIP and measurements is not None and values is not None:
            m = measurements if isinstance(measurements, Sequence) else [measurements]
            v = values if isinstance(values, Sequence) else [values]
            if len(m) != len(v):
                raise ValueError(
                    f"In ZIP mode, the number of measurements {len(m)} must match the number of parameter sets {len(v)}."
                )

        if not isinstance(circuits, Sequence):
            circuits = [circuits]
        else:
            circuits = list(circuits)

        normalized_circuits: list[QCircuit | CircuitBinding] = []

        for circuit in circuits:
            if isinstance(circuit, QCircuit) and circuit.measurements:
                if measurements is not None:
                    raise ValueError(
                        "your circuit already contains measurements, "
                        "you cannot have multiple measurements"
                    )

                circuit_measurements = circuit.measurements
                circuit_without_measurements = circuit.without_measurements(
                    deep_copy=False
                )

                circuit_without_measurements.transpiled_circuit = None

                normalized_circuits.append(
                    CircuitBinding(
                        circuits=circuit_without_measurements,
                        measurements=circuit_measurements,
                        shots=shots,
                    )
                )
            else:
                normalized_circuits.append(circuit)
        circuits = normalized_circuits

        if isinstance(values, Sequence):
            parameters = list(values)
        elif values is not None:
            parameters = [values]
        else:
            parameters = None

        self.transpiled_noise_model = None
        self.noises = noises
        self.shots = shots
        self._translated_circuits = None
        self._translated_observables = None
        self._translated_variables = None
        self.is_noisy = noises is not None and len(noises) > 0
        """ is_noisy is True if any of the circuits in the binding has noise instructions, False otherwise. """
        self.measurements = None

        self.job_type = circuits[0].job_type
        self.nb_qubits = circuits[0].nb_qubits
        for circ in circuits:
            if circ.job_type != self.job_type:
                if (
                    circ.job_type == JobType.STATE_VECTOR
                    or self.job_type == JobType.STATE_VECTOR
                ):
                    self.job_type = circ.job_type
                else:
                    raise ValueError(
                        "All circuits in CircuitBinding must have the same job type."
                    )
            if circ.is_noisy:
                if isinstance(circ, CircuitBinding) and self.noises is None:
                    self.noises = circ.noises
                    self.is_noisy = True
                else:
                    raise ValueError(
                        "All circuits in CircuitBinding must have the same noise. please use the noises parameter."
                    )
            if isinstance(circ, CircuitBinding):
                if self.shots is None:
                    self.shots = circ.shots
                elif circ.shots is not None and circ.shots != self.shots:
                    raise ValueError(
                        "All circuits in CircuitBinding must have the same number of shots."
                    )

            self.nb_qubits = max(self.nb_qubits, circ.nb_qubits)

        if measurements is not None:
            for c in circuits:
                if isinstance(c, QCircuit):
                    if c.job_type != JobType.STATE_VECTOR:
                        raise ValueError(
                            "your circuit already contains measurements, you cannot have multiple measurements"
                        )
            measurements = (
                [measurements]
                if isinstance(measurements, Measure)
                else list(measurements)
            )
            self.job_type = (
                JobType.OBSERVABLE
                if isinstance(measurements[0], ExpectationMeasure)
                else JobType.SAMPLE
            )

            shots_ = None
            for index, measure in enumerate(measurements):
                if self.shots is not None:
                    # TODO: this is a check for default shots but it is hardcode
                    if self.shots != measure.shots:
                        if (
                            isinstance(measure, ExpectationMeasure)
                            and measure.shots != 0
                        ):
                            raise ValueError(
                                "shots is already specified in CircuitBinding, you cannot specify it again in the measures"
                            )
                        elif (
                            isinstance(measure, BasisMeasure) and measure.shots != 1024
                        ):
                            raise ValueError(
                                "shots is already specified in CircuitBinding, you cannot specify it again in the measures"
                            )
                        else:
                            measure.shots = self.shots
                else:
                    if shots_ is None:
                        shots_ = measure.shots
                    elif measure.shots != shots_:
                        raise ValueError(
                            "All measurements in CircuitBinding must have the same number of shots."
                        )
                if isinstance(measure, ExpectationMeasure):
                    measurements[index] = adjust_measure(measure, self.nb_qubits)
                    if self.job_type != JobType.OBSERVABLE:
                        raise ValueError(
                            "All measurements in CircuitBinding must be of the same type."
                        )
                elif isinstance(measure, BasisMeasure):
                    if self.job_type != JobType.SAMPLE:
                        raise ValueError(
                            "All measurements in CircuitBinding must be of the same type."
                        )
            if shots_ is not None:
                self.shots = shots_

        if (
            self.job_type != JobType.STATE_VECTOR
            and measurements is None
            and self.shots is None
        ):
            shots = -1
            for circuit in circuits:
                if isinstance(circuit, 'CircuitBinding'):
                    for c in circuit.circuits:
                        if TYPE_CHECKING:
                            assert c.measurements is not None
                        m = c.measurements[0]
                        if m.shots != shots:
                            if shots == -1:
                                shots = m.shots
                            else:
                                raise ValueError(
                                    "All measurements in CircuitBinding must have the same number of shots"
                                )
                else:
                    if TYPE_CHECKING:
                        assert circuit.measurements
                    m = circuit.measurements[0]
                    if m.shots != shots:
                        if shots == -1:
                            shots = m.shots
                        else:
                            raise ValueError(
                                "All measurements in CircuitBinding must have the same number of shots"
                            )

        self.circuits: list["QCircuit | CircuitBinding"] = circuits
        self.value = parameters
        self.measurements = (
            self.measurements if self.measurements is not None else measurements
        )
        self.mode = mode

    def transpiled_circuits(
        self,
        device: AvailableDevice,
        skip_pre_measure: bool = False,
        backend_sim: Optional["AerSimulator"] = None,
    ) -> None:
        """Transpile every circuit in the binding for a target device.

        Transpiled provider circuits are cached on the corresponding
        :class:`QCircuit` objects. Nested bindings are processed recursively.
        For noisy IBM executions, the generated Qiskit noise model is cached on
        this binding.

        Args:
            device: Device for which the circuits are transpiled.
            skip_pre_measure: Whether provider translation should omit
                pre-measurement operations.
            backend_sim: Optional Qiskit Aer backend used during IBM
                transpilation.
        """

        from mpqp.execution.providers.ibm import generate_qiskit_noise_model
        from mpqp import QCircuit

        for i, c in enumerate(self.circuits):
            if isinstance(c, QCircuit):
                if self.is_noisy:
                    if c.transpiled_circuit is None:
                        original_noises = c.noises
                        c.noises = self.noises if self.noises is not None else []
                        nm, modified_circuit = generate_qiskit_noise_model(c)
                        c.noises = original_noises
                        if self.transpiled_noise_model is None:
                            self.transpiled_noise_model = nm

                        modified_circuit.transpiled_circuit = (
                            modified_circuit.to_other_device(
                                device, skip_pre_measure, backend_sim
                            )
                        )
                        self.circuits[i] = (
                            modified_circuit  # TODO: check if we need to update the circuit in the list or if we can just modify it in place
                        )
                else:
                    c.transpiled_circuit = c.to_other_device(
                        device, skip_pre_measure, backend_sim
                    )
            else:
                c.transpiled_circuits(device, skip_pre_measure, backend_sim)

    def unroll(
        self,
    ) -> list[BindingExecution]:
        """Resolve the lazy binding graph into individual executions.

        Returns:
            A flat, ordered list of ``(circuit, values, measurement)`` tuples.
            Values and measurements can be ``None`` when the corresponding
            execution axis is absent.

        Raises:
            ValueError: If a non-singleton axis cannot be broadcast in
                :attr:`BindingMode.ZIP` mode.
        """
        import itertools
        from typing import cast

        raw_parent_values = (
            self.value
            if isinstance(self.value, list)
            else ([self.value] if self.value is not None else [None])
        )
        parent_values: list[Optional[BindingParameters]] = [
            (
                cast(BindingParameters, parameter_set)
                if parameter_set is not None
                else None
            )
            for parameter_set in raw_parent_values
        ]
        parent_measurements: list[Measure | None] = (
            self.measurements
            if isinstance(self.measurements, list)
            else ([self.measurements] if self.measurements is not None else [None])
        )  # pyright: ignore[reportAssignmentType]

        def merge_values(
            child_parameters: Optional[BindingParameters],
            parent_parameters: Optional[BindingParameters],
        ) -> Optional[BindingParameters]:
            """Merge child and parent values, giving precedence to the parent."""
            if child_parameters is None and parent_parameters is None:
                return None
            combined_parameters = (
                dict(child_parameters) if child_parameters is not None else {}
            )
            if parent_parameters is not None:
                combined_parameters.update(parent_parameters)
            return combined_parameters

        def bind_same_parameters(
            child_parameters: Optional[BindingParameters],
            parent_parameters: Optional[BindingParameters],
        ) -> bool:
            """Return whether child and parent bind at least one common name."""
            if child_parameters is None or parent_parameters is None:
                return False
            child_names = {str(key) for key in child_parameters}
            parent_names = {str(key) for key in parent_parameters}
            return bool(child_names & parent_names)

        def combine_zipped_options(
            child_parameters: Optional[BindingParameters],
            child_measurement: Optional[Measure],
            parent_parameters: Optional[BindingParameters],
            parent_measurement: Optional[Measure],
        ) -> list[BindingOptions]:
            """Combine one child option with one parent option in ZIP mode."""
            parameter_collision = bind_same_parameters(
                child_parameters, parent_parameters
            )
            measurement_collision = (
                child_measurement is not None and parent_measurement is not None
            )
            if not parameter_collision and not measurement_collision:
                return [
                    (
                        merge_values(child_parameters, parent_parameters),
                        (
                            parent_measurement
                            if parent_measurement is not None
                            else child_measurement
                        ),
                    )
                ]

            if parameter_collision:
                if TYPE_CHECKING:
                    assert parent_parameters is not None
                    assert child_parameters is not None
                parent_execution_parameters = parent_parameters.copy()
                child_execution_parameters = child_parameters.copy()
            else:
                parent_execution_parameters = child_execution_parameters = merge_values(
                    child_parameters, parent_parameters
                )

            return [
                (
                    parent_execution_parameters,
                    (
                        parent_measurement
                        if parent_measurement is not None
                        else child_measurement
                    ),
                ),
                (
                    child_execution_parameters,
                    (
                        child_measurement
                        if child_measurement is not None
                        else parent_measurement
                    ),
                ),
            ]

        def parameter_set_key(
            parameter_set: Optional[BindingParameters],
        ) -> Optional[tuple[tuple[str, str]]]:
            """Build a stable, hashable key for a parameter mapping."""
            if parameter_set is None:
                return None
            return tuple(
                sorted((str(key), repr(value)) for key, value in parameter_set.items())
            )  # pyright: ignore[reportReturnType]

        def execution_key(
            execution: BindingExecution,
        ) -> tuple[int, Optional[tuple[tuple[str, str]]], int]:
            """Build an identity key used to remove duplicate executions."""
            circuit, parameters, measurement = execution
            return id(circuit), parameter_set_key(parameters), id(measurement)

        def product_values(
            child_parameters: Optional[BindingParameters],
        ) -> list[Optional[BindingParameters]]:
            """Resolve and deduplicate parameter sets for PRODUCT mode."""
            unique_values: dict[
                Optional[tuple[tuple[str, str]]], Optional[BindingParameters]
            ] = {}
            keep_child_parameters = False

            for parent_parameters in parent_values:
                if child_parameters is None:
                    parameter_set = (
                        parent_parameters.copy()
                        if parent_parameters is not None
                        else None
                    )
                elif parent_parameters is None:
                    keep_child_parameters = True
                    continue
                elif bind_same_parameters(child_parameters, parent_parameters):
                    parameter_set = parent_parameters.copy()
                    keep_child_parameters = True
                else:
                    parameter_set = merge_values(child_parameters, parent_parameters)

                unique_values.setdefault(
                    parameter_set_key(parameter_set), parameter_set
                )

            if keep_child_parameters:
                if TYPE_CHECKING:
                    assert child_parameters is not None
                child_parameters = child_parameters.copy()
                unique_values.setdefault(
                    parameter_set_key(child_parameters), child_parameters
                )

            return list(unique_values.values())

        def product_measurements(
            child_measurement: Optional[Measure],
        ) -> list[Optional[Measure]]:
            """Resolve and deduplicate measurements for PRODUCT mode."""
            unique_measurements: dict[int, Optional[Measure]] = {}
            keep_child_measurement = False

            for parent_measurement in parent_measurements:
                if child_measurement is None:
                    unique_measurements.setdefault(
                        id(parent_measurement), parent_measurement
                    )
                elif parent_measurement is None:
                    keep_child_measurement = True
                else:
                    unique_measurements.setdefault(
                        id(parent_measurement), parent_measurement
                    )
                    keep_child_measurement = True

            if keep_child_measurement:
                unique_measurements.setdefault(id(child_measurement), child_measurement)

            return list(unique_measurements.values())

        def expand_branch(
            branch: QCircuit | CircuitBinding,
        ) -> list[BindingExecution]:
            """Return the executions represented by a circuit or nested binding."""
            if isinstance(branch, CircuitBinding):
                return branch.unroll()
            return [(branch, None, None)]

        executions: list[BindingExecution] = []

        if self.mode == BindingMode.ZIP:
            zip_length = max(
                len(self.circuits),
                len(parent_values),
                len(parent_measurements),
            )

            def broadcast_axis(
                axis: list[_AxisElement], target_length: int
            ) -> list[_AxisElement]:
                """Broadcast a singleton ZIP axis or validate its length."""
                if len(axis) == 1:
                    return [axis[0]] * target_length
                if len(axis) != target_length:
                    raise ValueError(
                        f"In ZIP mode, lists must be length 1 or match the maximum list length ({target_length})."
                    )
                return axis

            branches = broadcast_axis(self.circuits, zip_length)
            values = broadcast_axis(parent_values, zip_length)
            measurements = broadcast_axis(parent_measurements, zip_length)

            for branch, parent_parameters, parent_measurement in zip(
                branches, values, measurements
            ):
                assert branch is not None
                for circuit, child_parameters, child_measurement in expand_branch(
                    branch
                ):
                    for parameters, measurement in combine_zipped_options(
                        child_parameters,
                        child_measurement,
                        parent_parameters,
                        parent_measurement,
                    ):
                        executions.append((circuit, parameters, measurement))

        else:
            for branch in self.circuits:
                unique_branch_executions: dict[
                    tuple[int, Optional[tuple[tuple[str, str]]], int],
                    BindingExecution,
                ] = {}
                for circuit, child_parameters, child_measurement in expand_branch(
                    branch
                ):
                    for parameters, measurement in itertools.product(
                        product_values(child_parameters),
                        product_measurements(child_measurement),
                    ):
                        execution = (circuit, parameters, measurement)
                        unique_branch_executions.setdefault(
                            execution_key(execution), execution
                        )
                executions.extend(unique_branch_executions.values())

        return executions

    def __repr__(self):
        """Return an unambiguous representation of the binding."""
        return f"CircuitBinding(circuits={repr(self.circuits)}, values={repr(self.value)}, measurements={repr(self.measurements)}, mode={repr(self.mode)}, noises={repr(self.noises)}, shots={repr(self.shots)})"

    @overload
    def to_other_device(
        self, device: AWSDevice
    ) -> tuple["ProgramSet", list[tuple[Any]]]: ...
    @overload
    def to_other_device(
        self, device: IBMDevice
    ) -> list[tuple["EstimatorPubLike", list["Job"]]]: ...
    def to_other_device(
        self, device: AvailableDevice
    ) -> "CircuitBinding | tuple[ProgramSet, list[tuple[Any]]] | list[tuple[EstimatorPubLike, list[Job]]]":
        """Translate a binding for the selected provider.

        Args:
            device: Provider device targeted by the translation.

        Returns:
            For AWS devices, a Braket ``ProgramSet`` and the MPQP context used
            to reconstruct results. For IBM devices, a list pairing each
            Qiskit estimator PUB with its ordered context jobs.

        Raises:
            ValueError: If IBM translation is requested for a binding that
                does not contain expectation measurements.
            NotImplementedError: If the device provider is unsupported.

        Note:
            For IBM, PRODUCT/ZIP combinations are resolved before executions
            are grouped by circuit. PUB fields are circuit, observables and
            parameter values; absent trailing fields are omitted.
        """
        from mpqp.execution.job import Job
        from mpqp.execution.devices import (
            IBMDevice,
            AWSDevice,
        )
        from mpqp.execution.providers.ibm import JobType
        from copy import deepcopy
        from mpqp.core.instruction.measurement import ExpectationMeasure
        from mpqp.core.languages import Language

        if isinstance(device, IBMDevice):
            if self.job_type != JobType.OBSERVABLE:
                raise ValueError(
                    "to_other_device is only supported for circuits with expectation measurements."
                )

            unrolled = self.unroll()
            if any(c.transpiled_circuit is None for c, _, _ in unrolled):
                self.transpiled_circuits(device=device)

            grouped = {}
            for c, values, measure in unrolled:
                grouped.setdefault(id(c), []).append((c, values, measure))

            pubs_with_context = []
            for executions in grouped.values():
                q_c = executions[0][0].transpiled_circuit
                if TYPE_CHECKING:
                    from qiskit.circuit import QuantumCircuit

                    assert isinstance(q_c, QuantumCircuit)
                parameter_names = [p.name for p in q_c.parameters]
                params = []
                q_obs = []
                contexts = []
                for c, values, measure in executions:
                    normalized_values = {str(k): v for k, v in (values or {}).items()}
                    missing = [
                        name
                        for name in parameter_names
                        if name not in normalized_values
                    ]
                    if missing:
                        raise ValueError(
                            f"Missing values for circuit parameters: {missing}"
                        )
                    params.append([normalized_values[name] for name in parameter_names])
                    if not isinstance(measure, ExpectationMeasure):
                        raise ValueError(
                            "Observable bindings require expectation measures."
                        )
                    q_obs.append(
                        [
                            (
                                obs.pre_transpiled
                                if obs.pre_transpiled is not None
                                else obs.to_other_language(Language.QISKIT)
                            )
                            for obs in measure.observables
                        ]
                    )
                    context = c.without_measurements(deep_copy=False)
                    context.add(deepcopy(measure))
                    contexts.append(Job(self.job_type, context, device, values=values))

                if all(len(p) == 0 for p in params):
                    params = None
                if all(len(o) == 0 for o in q_obs):
                    q_obs = None
                if q_obs and params:
                    pub = (q_c, q_obs, params)
                elif q_obs:
                    pub = (q_c, q_obs)
                elif params:
                    pub = (q_c, None, params)
                else:
                    pub = (q_c,)
                pubs_with_context.append((pub, contexts))

            return pubs_with_context
        elif isinstance(device, AWSDevice):
            from mpqp.translation.braket import circuitbinding_to_programset

            return circuitbinding_to_programset(self, device)
        else:
            raise NotImplementedError(
                f"Translation to {device} is not supported yet on CircuitBindings"
            )
