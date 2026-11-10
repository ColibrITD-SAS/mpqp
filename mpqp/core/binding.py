"""Package related quantum computations for optimized provider execution.

A :class:`CircuitBinding` combines circuits, parameter values and measurements
into a description of related runs. Providers can package these runs into one
submission and, where supported, share compatible hardware executions. For
example, Pauli observables with compatible measurement bases can be estimated
from shared samples of the same circuit, reducing QPU time and cost. Execution
paths without native support fall back to individual runs.

:class:`BindingMode` specifies how the three execution axes combine:

* :attr:`BindingMode.PRODUCT` explores combinations of circuits, parameter sets
  and measurements, for example a parameter sweep evaluated against several
  observables.
* :attr:`BindingMode.ZIP` associates corresponding entries, for example pairing
  each parameter set with a particular measurement.

Bindings can be nested to supply parameters and measurements at different
levels. :meth:`CircuitBinding.unroll` exposes the resolved execution sequence
without running it. Pass the binding to :func:`~mpqp.execution.runner.run` to
execute it and obtain a :class:`~mpqp.execution.result.BatchResult`. Its results
follow unroll order even when a provider groups executions internally. The
number of logical results need not equal the number of provider submissions or
hardware executions.

Both public classes are available through ``from mpqp import CircuitBinding,
BindingMode``.
"""

from __future__ import annotations

from copy import copy
from enum import Enum, auto
from numbers import Complex
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union, overload

from sympy import Expr

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.noise.noise_model import NoiseModel
from mpqp.tools.generics import OneOrMany

if TYPE_CHECKING:
    from braket.program_sets import ProgramSet
    from qiskit.primitives.containers import EstimatorPubLike
    from qiskit_aer import AerSimulator

    from mpqp.execution._execution_context import ExecutionContext
    from mpqp.execution.devices import AvailableDevice, AWSDevice, IBMDevice


BindingParameters = dict[Union[Expr, str], Union[Complex, float]]
"""Parameter values keyed by symbolic expressions or parameter names."""
BindingExecution = tuple[QCircuit, Optional[BindingParameters], Optional[Measure]]
"""One logical execution: circuit, optional parameters and optional measurement."""
BindingOptions = tuple[Optional[BindingParameters], Optional[Measure]]
"""Optional parameters and measurement combined while resolving a binding."""
_AxisElement = TypeVar("_AxisElement")


class BindingMode(Enum):
    """Control how a :class:`CircuitBinding` combines its execution axes.

    The axes are circuits, parameter mappings and measurements. An omitted
    parameter or measurement axis contributes a single unspecified entry.
    Nested bindings resolve their own axes before being combined with the
    enclosing binding's options.
    """

    PRODUCT = auto()
    """Builds the Cartesian product of circuits, parameter sets and measurements."""
    ZIP = auto()
    """Associates elements positionally and broadcasts an axis containing a
    single element to the length of the longest axis."""


class CircuitBinding:
    """Group related circuit executions for provider optimizations.

    A binding groups one or more circuits with optional parameter sets and
    measurements. It can also contain nested bindings. The execution axes are
    resolved only when the binding is translated for a provider or passed to
    :func:`~mpqp.execution.runner.run`.

    Use :attr:`BindingMode.PRODUCT` for a parameter or measurement sweep and
    :attr:`BindingMode.ZIP` to pair corresponding inputs. Measurements already
    present in a circuit are extracted into an inner binding automatically.

    Args:
        circuits: Circuit, nested binding, or sequence of either to execute.
        values: Parameter mapping or sequence of mappings to bind to the
            symbolic circuit parameters. Keys may be symbolic expressions or
            parameter names. Omission leaves parameters to an enclosing binding
            or requires the circuit to have no unbound parameters at execution.
        measurements: Measurement or sequence of measurements to apply. A
            circuit that already contains measurements must not also receive
            this argument. Each supplied measurement is shallow-copied before
            shot overrides are applied. Nested mutable objects, including
            observables and bases, may remain shared.
        mode: Rule used to combine circuits, values and measurements.
        noises: Noise models shared by every circuit in the binding.
            Due to provider limitations, all circuits must share the same noise model.
        shots: Number of shots shared by every execution. When omitted, it is
            inferred from the measurements when possible. Zero shots requests
            a state vector for basis measurements and exact expectation values
            for expectation measurements.

    Raises:
        ValueError: If the execution axes are incompatible, circuits have
            incompatible job types, or measurements, noise and shots are
            specified inconsistently.

    Examples:
        Evaluate two parameter sets against two observables. PRODUCT creates
        four logical executions; ZIP pairs the entries to create two:

        >>> from sympy import Symbol
        >>> from mpqp import (BindingMode, CircuitBinding, ExpectationMeasure,
        ...                   Observable, QCircuit, Ry, pX, pZ)
        >>> theta = Symbol("theta")
        >>> circuit = QCircuit([Ry(theta, 0)])
        >>> values = [{theta: 0.0}, {theta: 1.0}]
        >>> measures = [ExpectationMeasure(Observable(pZ)),
        ...             ExpectationMeasure(Observable(pX))]
        >>> product = CircuitBinding(circuit, values=values, measurements=measures)
        >>> len(product.unroll())
        4
        >>> paired = CircuitBinding(circuit, values=values,
        ...                         measurements=measures, mode=BindingMode.ZIP)
        >>> len(paired.unroll())
        2

        A single measurement is reused across the parameter axis in ZIP mode:

        >>> shared_measurement = CircuitBinding(circuit, values=values,
        ...     measurements=measures[0], mode=BindingMode.ZIP)
        >>> len(shared_measurement.unroll())
        2

        Nest a parameter sweep inside a binding that supplies the measurement:

        >>> sweep = CircuitBinding(circuit, values=values)
        >>> measured = CircuitBinding(sweep, measurements=measures[0])
        >>> len(measured.unroll())
        2

        Execute a sweep on a local simulator. The analytic expectation of Z
        after Ry(theta) is cos(theta):

        >>> import math
        >>> from mpqp import IBMDevice, run
        >>> sweep = CircuitBinding(circuit,
        ...     values=[{theta: 0.0}, {theta: math.pi}],
        ...     measurements=ExpectationMeasure(Observable(pZ)))
        >>> result = run(sweep, IBMDevice.AER_SIMULATOR)  # doctest: +QISKIT
        >>> [round(float(r.expectation_values), 6)
        ...  for r in result.results]  # doctest: +QISKIT
        [1.0, -1.0]

    Note:
        In :attr:`BindingMode.ZIP` mode, every non-singleton axis must have the
        same length. Singleton axes are broadcast when the binding is unrolled.
        This applies equally to circuits, parameter sets and measurements.
        An omitted parameter or measurement axis counts as one unspecified
        entry. A nested binding counts as one entry in the circuit axis.

        Measurements in a binding must have the same type and effective shot
        count. Binding-level ``shots`` can replace a measurement's default count
        (1024 for basis measurements, zero for expectation measurements);
        conflicting non-default counts are rejected. Without an explicit
        measurement, a numeric circuit requests a state vector.

        Circuits are retained by reference. A shallow measurement copy protects
        its shot count but does not provide an independent copy of nested arrays,
        target lists or basis objects. Finish preparing inputs before execution.

        Results returned by :func:`~mpqp.execution.runner.run` follow
        :meth:`unroll` order, independently of provider execution grouping.
    """

    def __init__(
        self,
        circuits: OneOrMany[QCircuit | CircuitBinding],
        values: Optional[OneOrMany[dict[str, float] | dict[Expr, float]]] = None,
        measurements: Optional[OneOrMany[Measure]] = None,
        mode: BindingMode = BindingMode.PRODUCT,
        noises: Optional[list[NoiseModel]] = None,
        shots: Optional[int] = None,
    ) -> None:
        from typing import TYPE_CHECKING, Sequence

        from mpqp.core.instruction.measurement import (
            BasisMeasure,
            ExpectationMeasure,
            Measure,
        )
        from mpqp.execution.job import JobType

        if circuits == []:
            raise ValueError(
                "At least one Circuit/CircuitBinding must be provided to a CircuitBinding."
            )

        if mode == BindingMode.ZIP:
            lengths = [
                len(axis) if isinstance(axis, Sequence) else 1
                for axis in (circuits, values, measurements)
            ]
            zip_length = max(lengths)
            if any(length not in (1, zip_length) for length in lengths):
                raise ValueError(
                    "In ZIP mode, each axis must have length 1 or match the "
                    f"maximum axis length ({zip_length}); got lengths "
                    f"{tuple(lengths)} for circuits, values and measurements."
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
            measurements = [copy(measure) for measure in measurements]
            is_expectation = isinstance(measurements[0], ExpectationMeasure)

            shots_ = None
            for measure in measurements:
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

                    if not is_expectation:
                        raise ValueError(
                            "All measurements in CircuitBinding must be of the same type."
                        )
                elif isinstance(measure, BasisMeasure):
                    if is_expectation:
                        raise ValueError(
                            "All measurements in CircuitBinding must be of the same type."
                        )
            if shots_ is not None:
                self.shots = shots_
            self.job_type = (
                JobType.OBSERVABLE
                if is_expectation
                else JobType.STATE_VECTOR if self.shots == 0 else JobType.SAMPLE
            )

        if (
            self.job_type != JobType.STATE_VECTOR
            and measurements is None
            and self.shots is None
        ):
            shots = -1
            for circuit in circuits:
                if isinstance(circuit, CircuitBinding):
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

        self.circuits: list[QCircuit | CircuitBinding] = circuits
        self.value = parameters
        self.measurements = (
            self.measurements if self.measurements is not None else measurements
        )
        self.mode = mode

    def unroll(
        self,
    ) -> list[BindingExecution]:
        """Resolve the lazy binding graph into individual executions.

        Expectation measurements are embedded into each resolved circuit's
        register, not the largest register in the binding. Input measurements
        are not modified when embedding or reordering their targets.

        Returns:
            A flat, ordered list of ``(circuit, values, measurement)`` tuples.
            Values and measurements can be ``None`` when the corresponding
            execution axis is absent.

        Raises:
            ValueError: If a non-singleton axis cannot be broadcast in
                :attr:`BindingMode.ZIP` mode, or expectation measurement targets
                fall outside a resolved circuit's register.

        Note:
            For a flat PRODUCT binding, circuit order is outermost, followed by
            parameter order and then measurement order. ZIP follows the paired
            axis order. Returned circuits and unchanged measurements are
            references to binding objects. Measurements requiring embedding
            are copied and reused for matching register sizes within this call.

            Nested bindings combine disjoint parameter mappings. If parent and
            child mappings contain overlapping parameter names, they contribute
            alternative parameter sets rather than overwriting one another.
            Measurements supplied at both levels also contribute alternatives.
            PRODUCT removes duplicate parameter mappings and repeated identical
            measurement objects within each circuit branch. These rules can
            make nested execution counts differ from a simple product of all
            input lengths.
        """
        import itertools
        from copy import deepcopy
        from typing import cast

        from mpqp.core.instruction.measurement import ExpectationMeasure
        from mpqp.execution.runner import adjust_measure
        from mpqp.tools.binding_utils import (
            broadcast_axis,
            expand_branch,
            combine_zipped_options,
            product_values,
            product_measurements,
            execution_key,
        )

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

        executions: list[BindingExecution] = []

        if self.mode == BindingMode.ZIP:
            zip_length = max(
                len(self.circuits),
                len(parent_values),
                len(parent_measurements),
            )

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
                        product_values(child_parameters, parent_values),
                        product_measurements(child_measurement, parent_measurements),
                    ):
                        execution = (circuit, parameters, measurement)
                        unique_branch_executions.setdefault(
                            execution_key(execution), execution
                        )
                executions.extend(unique_branch_executions.values())

        # Resolve ZIP/PRODUCT and nested bindings before choosing a register size.
        # Reuse embeddings within this unroll to preserve measurement identity.
        adjusted: dict[tuple[int, int], ExpectationMeasure] = {}
        resolved: list[BindingExecution] = []
        for circuit, parameters, measurement in executions:
            if isinstance(measurement, ExpectationMeasure):
                if any(
                    target < 0 or target >= circuit.nb_qubits
                    for target in measurement.targets
                ):
                    raise ValueError(
                        f"Measurement targets {measurement.targets} are outside "
                        f"the resolved circuit's {circuit.nb_qubits}-qubit register."
                    )
                if measurement.targets != list(range(circuit.nb_qubits)):
                    key = (id(measurement), circuit.nb_qubits)
                    if key not in adjusted:
                        adjusted[key] = adjust_measure(deepcopy(measurement), circuit)
                    measurement = adjusted[key]
            resolved.append((circuit, parameters, measurement))
        return resolved

    def __repr__(self):
        """Return an unambiguous representation of the binding."""
        return f"CircuitBinding(circuits={repr(self.circuits)}, values={repr(self.value)}, measurements={repr(self.measurements)}, mode={repr(self.mode)}, noises={repr(self.noises)}, shots={repr(self.shots)})"

    @overload
    def to_other_device(
        self, device: AWSDevice, backend_sim: Optional[AerSimulator] = None
    ) -> tuple[ProgramSet, list[Any]]: ...
    @overload
    def to_other_device(
        self, device: IBMDevice, backend_sim: Optional[AerSimulator] = None
    ) -> list[tuple[EstimatorPubLike, list[ExecutionContext]]]: ...
    @overload
    def to_other_device(
        self, device: AvailableDevice, backend_sim: Optional[AerSimulator] = None
    ) -> (
        list[tuple[EstimatorPubLike, list[ExecutionContext]]]
        | tuple[ProgramSet, list[Any]]
    ): ...
    def to_other_device(
        self, device: AvailableDevice, backend_sim: Optional[AerSimulator] = None
    ) -> (
        list[tuple[EstimatorPubLike, list[ExecutionContext]]]
        | tuple[ProgramSet, list[Any]]
    ):
        """Translate a binding for the selected provider.

        Args:
            device: Provider device targeted by the translation.
            backend_sim: Optional Aer backend. Currently unused by this method.

        Returns:
            For AWS devices, a Braket ``ProgramSet`` and the MPQP context used
            to reconstruct results. For IBM devices, a list pairing each
            Qiskit estimator PUB with execution contexts containing each job
            and its unroll index before provider grouping.

        Raises:
            ValueError: If IBM translation is requested for a binding that
                does not contain expectation measurements.
            NotImplementedError: If the device provider is unsupported.

        Note:
            For IBM, PRODUCT/ZIP combinations are resolved before executions
            are grouped by circuit. PUB fields are circuit, observables and
            parameter values; absent trailing fields are omitted.
            This method prepares provider input and execution context; it does
            not submit jobs. Prefer :func:`~mpqp.execution.runner.run` when
            execution results are needed.
        """
        from mpqp.execution.devices import (
            AWSDevice,
            IBMDevice,
        )

        if isinstance(device, IBMDevice):
            from mpqp.translation.qiskit import circuitbinding_to_pubs

            return circuitbinding_to_pubs(self, device)

        elif isinstance(device, AWSDevice):
            from mpqp.translation.braket import circuitbinding_to_programset

            return circuitbinding_to_programset(self, device)
        else:
            raise NotImplementedError(
                f"Translation to {device} is not supported yet on CircuitBindings"
            )
