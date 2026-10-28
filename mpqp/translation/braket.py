from __future__ import annotations
from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
)
from mpqp.environment.var_cache import (
    InstalledProviders,
)

if InstalledProviders.BRAKET in _INSTALLED_MPQP_PROVIDERS:
    from typing import (
        TYPE_CHECKING,
        Any,
        Mapping,
        Optional,
    )
    from warnings import warn

    from braket.program_sets import CircuitBinding as BraketBinding

    from dataclasses import dataclass

    from mpqp.core.binding import BindingParameters
    from mpqp.core.circuit import QCircuit
    from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
    from mpqp.core.instruction.measurement.measure import Measure

    if TYPE_CHECKING:
        from braket.program_sets import ProgramSet

        from mpqp.core.binding import CircuitBinding
        from mpqp.execution.devices import AWSDevice
        from braket.circuits import Circuit as BraketCircuit
        from mpqp.core.instruction.measurement.pauli_string import PauliStringMonomial
        import numpy.typing as npt
        import numpy as np
        from mpqp import Gate

    @dataclass(frozen=True)
    class BraketPauliGroupingContext:
        """Metadata for one logical part of a Braket program-set entry.

        A grouped entry produces probabilities rather than a Braket expectation
        value. ``eigenvalues`` maps every Pauli monomial measured by the entry to
        the diagonal used to recover its expectation from those probabilities.
        For a non-grouped entry, ``observable_index`` identifies the provider
        expectation that belongs to the original MPQP measurement and
        ``executable_span`` records the number of Braket executables occupied by
        that observable.
        """

        """MPQP circuit to be executed"""
        circuit: QCircuit

        """Optional measurement with which the circuit will be executed."""
        measurement: Optional[Measure]

        """Parameters with which the circuit will be executed."""
        values: Optional[BindingParameters]

        """Index of the result in the original binding's unrolled executions."""
        result_index: int

        """True if the entry is part of a grouped Pauli measurement, False otherwise."""
        optimized: bool

        """Index of the Pauli group in the measurement's grouping, if optimized."""
        group_index: Optional[int] = None

        """If not optimized, stored the index of the observable in the measurement's observables."""
        observable_index: Optional[int] = None

        """If optimized, store the Pauli monomials to be executed."""
        monomials: tuple[PauliStringMonomial, ...] = ()

        """If optimized, store the eigenvalues of the Pauli monomials used to compute the expectation value."""
        eigenvalues: Optional[Mapping[str, npt.NDArray[np.float64]]] = None

        """If not optimized, store the number of Braket executables occupied by the observable."""
        executable_span: Optional[int] = None

    def braket_to_mpqp(qcircuit: "BraketCircuit") -> "QCircuit":

        from braket.circuits.serialization import IRType
        from braket.ir.openqasm.program_v1 import Program
        from braket.circuits import Circuit as BraketCircuit
        from mpqp.core.languages import Language
        from mpqp.translation.qasm.open_qasm_2_and_3 import open_qasm_3_to_2
        from mpqp.translation.qasm.qasm_to_braket import braket_noise_to_mpqp
        from mpqp.translation.qasm.qasm_to_mpqp import qasm2_parse

        assert isinstance(qcircuit, BraketCircuit)
        remove_measure = True
        for instr in qcircuit.instructions:
            if instr.operator.name == "Measure":
                remove_measure = False
                break

        qasm3_code = qcircuit.to_ir(IRType.OPENQASM)

        if TYPE_CHECKING:
            assert isinstance(qasm3_code, Program)
        noises, qasm3_code = braket_noise_to_mpqp(qasm3_code.source)

        qasm2_code = open_qasm_3_to_2(
            qasm3_code,
            language=Language.BRAKET,
            remove_measure=remove_measure,
        )

        qc = qasm2_parse(qasm2_code)
        if len(noises) != 0:
            qc.add(noises)
        return qc

    def get_braket_gate_set() -> set[type[Gate]]:
        """Return gates directly representable by Braket."""
        from mpqp.gates import CNOT, PRX, Rx, Rxx, Ry, Ryy, Rz, Rzz

        return {
            Rx,
            Ry,
            Rz,
            PRX,
            Rxx,
            Ryy,
            Rzz,
            CNOT,
        }

    def mpqp_to_braket(
        circuit: QCircuit,
        skip_pre_measure: bool = False,
        skip_measurements: bool = False,
    ) -> BraketCircuit:
        """Translate a MPQP circuit to a Braket equivalent.

        Note:
            If the circuit contains ComposedGate type instructions you need to include the gate in the authorize_gates set for it to avoid decomposition, otherwise, the translation will always
            try to use the gate's decomposition (see example).
        Args:
            circuit: The original MPQP circuit to be translated.
            skip_pre_measure: If set at True will translate the circuit without its pre-measurement circuit (see QCircuit.to_other_language for more information).
            skip_measurements: If set at True will translate the circuit without any measurement.

        Examples:
            >>> circuit = QCircuit([H(0), CNOT(0, 1), BasisMeasure()])
            >>> cirq_with = mpqp_to_braket(circuit)
            >>> cirq_without = mpqp_to_braket(circuit, skip_measurements=True)
            >>> print(cirq_with)  # doctest: +NORMALIZE_WHITESPACE
            T  : │  0  │  1  │  2  │
                  ┌───┐       ┌───┐
            q0 : ─┤ H ├───●───┤ M ├─
                  └───┘   │   └───┘
                        ┌─┴─┐ ┌───┐
            q1 : ───────┤ X ├─┤ M ├─
                        └───┘ └───┘
            T  : │  0  │  1  │  2  │
            >>> print(cirq_without)  # doctest: +NORMALIZE_WHITESPACE
            T  : │  0  │  1  │
                  ┌───┐
            q0 : ─┤ H ├───●───
                  └───┘   │
                        ┌─┴─┐
            q1 : ───────┤ X ├─
                        └───┘
            T  : │  0  │  1  │
        """
        from mpqp.core.circuit import QCircuit
        from mpqp.core.instruction import (
            Barrier,
            BasisMeasure,
            Breakpoint,
            ControlledGate,
            Measure,
        )
        from mpqp.core.instruction.gates.custom_controlled_gate import (
            CustomControlledGate,
        )
        from mpqp.core.instruction.gates.custom_gate import CustomGate
        from mpqp.core.instruction.gates.gate import Gate
        from mpqp.core.instruction.gates.native_gates import CRk
        from mpqp.core.languages import Language
        from mpqp.execution.providers.aws import apply_noise_to_braket_circuit
        from mpqp.core.instruction.gates.gate_decomposition import resolve_gate
        from braket.circuits import Circuit as BraketCircuit

        if len(circuit.noises) != 0:
            if any(isinstance(instr, CRk) for instr in circuit.instructions):
                raise NotImplementedError(
                    "Cannot simulate noisy circuit with CRk gate due to "
                    "an error on AWS Braket side."
                )

        braket_circuit = BraketCircuit()

        # If the number of qubits are defined by the user, we ensure that every qubits are used.
        # Otherwise the circuit can remain non continuous.
        if circuit._user_nb_qubits is not None:  # pyright: ignore[reportPrivateUsage]
            used_qubits = set().union(
                *(
                    inst.connections()
                    for inst in circuit.instructions
                    if isinstance(inst, Gate)
                )
            )
            if len(used_qubits) != circuit.nb_qubits:
                from copy import deepcopy

                from mpqp.gates import Id

                circuit = QCircuit(
                    [
                        Id(qubit)
                        for qubit in range(circuit.nb_qubits)
                        if qubit not in used_qubits
                    ],
                    nb_qubits=circuit.nb_qubits,
                ) + deepcopy(circuit)

        for instruction in circuit.instructions + circuit.measurements:
            targets = [target for target in instruction.targets]
            if isinstance(instruction, (Barrier, Breakpoint)):
                continue
            if isinstance(instruction, Measure):
                if not skip_pre_measure:
                    for pre_measure in instruction.pre_measure:
                        bracket_pre_measure = pre_measure.to_other_language(
                            Language.BRAKET
                        )
                        braket_circuit.add(bracket_pre_measure, targets)
                if not skip_measurements:
                    if isinstance(instruction, BasisMeasure) and instruction.shots != 0:
                        braket_circuit.measure(targets)
                continue
            if isinstance(instruction, Gate):
                instructions = resolve_gate(
                    instruction,
                    get_braket_gate_set(),
                )
            else:
                instructions = (instruction,)

            for instruction in instructions:
                braket_instr = instruction.to_other_language(Language.BRAKET)
                try:
                    targets = [target for target in instruction.targets]
                    if isinstance(instruction, CustomControlledGate):
                        if isinstance(instruction.non_controlled_gate, CustomGate):
                            targets = [
                                control for control in instruction.controls
                            ] + targets
                            targets.sort()
                    elif isinstance(instruction, ControlledGate):
                        targets = [
                            control for control in instruction.controls
                        ] + targets
                    braket_circuit.add_instruction(braket_instr, target=targets)
                except Exception as e:
                    raise ValueError(
                        f"{type(braket_instr)}{braket_instr} cannot be added to the braket circuit: {e}"
                    )
        if len(circuit.noises) != 0:
            braket_circuit = apply_noise_to_braket_circuit(
                braket_circuit,
                circuit.noises,
                circuit.nb_qubits,
            )
        return braket_circuit

    def _group_measurement_circuit(
        measurement: ExpectationMeasure,
        group: list[PauliStringMonomial],
        nb_qubits: int,
    ) -> BraketCircuit:
        """Helper function destined to CircuitBinding to ProgramSet translation in the case of Manual Pauli grouping.
        It builds the basis rotation and computational-basis measurement circuit to be stored later in the BraketPauliGroupingContext.
        """

        from mpqp.core.instruction.measurement import BasisMeasure
        from mpqp.core.languages import Language
        from mpqp.tools.pauli_grouping import find_qubitwise_rotations
        from mpqp.core.instruction.measurement.pauli_string import CommutingTypes

        if measurement.commuting_type != CommutingTypes.QUBITWISE:
            raise NotImplementedError(
                "Braket Pauli grouping currently supports only qubit-wise "
                "commuting groups."
            )

        pre_measure = QCircuit(
            find_qubitwise_rotations(group, measurement.targets)
            + [BasisMeasure(targets=measurement.targets, shots=measurement.shots)],
            nb_qubits=nb_qubits,
        ).to_other_language(Language.BRAKET)
        return pre_measure

    def _braket_values(values: BindingParameters) -> dict[str, float]:
        """Helper function destined to CircuitBinding to ProgramSet translation.
        Convert MPQP parameter values to a Braket input mapping.

        Args:
            values: Numeric values keyed by MPQP strings or symbolic
                expressions.

        Returns:
            A Braket-compatible mapping with string keys and real
            floating-point values.

        Raises:
            TypeError: If a value is not real, since Braket circuit inputs
                cannot represent complex values.
        """
        from numbers import Real

        converted: dict[str, float] = {}
        for key, value in values.items():
            if not isinstance(value, Real):
                raise TypeError(
                    "Braket circuit parameters must be real numbers, "
                    f"but parameter {key} has value {value}."
                )
            converted[str(key)] = float(value)
        return converted

    def _cb_to_programset_pauli_grouping(
        binding: CircuitBinding,
        device: AWSDevice,
    ) -> tuple[ProgramSet, list[list[BraketPauliGroupingContext]]]:
        """Translate an MPQP binding to Braket and group compatible Pauli terms.

        Pauli grouping is applied independently to every
        :class:`~mpqp.core.instruction.measurement.expectation_value.ExpectationMeasure`
        whose ``optimize_measurement`` attribute is true. Compatible terms share a
        basis-rotation circuit, so one sampled execution supplies all their
        expectation values. Parameter rows using that same circuit and group are
        packed into one Braket ``CircuitBinding``.

        Measurements with ``optimize_measurement=False`` retain Braket's native
        observable translation. If no measurement requests grouping, this function
        delegates to the existing translator unchanged.

        Args:
            binding: Binding containing circuits, parameter rows and measurements.
            device: AWS device targeted by the translation.

        Returns:
            A Braket program set and one ordered context list for each program-set
            entry. Grouped contexts contain the monomials and eigenvalue diagonals
            needed to reconstruct the original expectation values from sampled
            probabilities.

        Raises:
            ValueError: If grouping is requested for a zero-shot binding or an
                execution has no expectation measurement.
            NotImplementedError: If full-commutation grouping is requested.
            TypeError: If a circuit parameter is not real.

        Note:
            This function only prepares provider input. The current AWS result
            adapter does not yet reconstruct grouped probability contexts.
        """

        from braket.circuits.observables import Sum
        from braket.program_sets import CircuitBinding as BraketBinding
        from braket.program_sets import ProgramSet
        from braket.circuits import Circuit as BraketCircuit
        from collections import OrderedDict

        from mpqp.core.languages import Language
        from mpqp.tools.pauli_grouping import pauli_monomial_eigenvalues

        executions = binding.unroll()
        optimized = [
            measurement
            for _, _, measurement in executions
            if isinstance(measurement, ExpectationMeasure)
            and measurement.optimize_measurement
        ]
        if not optimized:
            return _grouped_cb_to_programset(binding, device)
        if binding.shots == 0:
            raise ValueError("Braket ProgramSets are not available with zero shots.")

        translated_circuits: dict[int, BraketCircuit] = {}
        grouped_entries: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
        entries: list[Any] = []
        entry_contexts: list[list[BraketPauliGroupingContext]] = []

        for result_index, (circuit, values, measurement) in enumerate(executions):
            if not isinstance(measurement, ExpectationMeasure):
                raise ValueError(
                    "Pauli-grouped Braket bindings require ExpectationMeasure "
                    "executions."
                )

            circuit_id = id(circuit)
            provider_circuit = translated_circuits.get(circuit_id)
            if provider_circuit is None:
                from braket.circuits import Circuit as BraketCircuit

                if isinstance(circuit.transpiled_circuit, BraketCircuit):
                    from mpqp.execution.providers.aws import (
                        apply_noise_to_braket_circuit,
                    )

                    provider_circuit = apply_noise_to_braket_circuit(
                        circuit.transpiled_circuit,
                        noises=binding.noises or [],
                        nb_qubits=circuit.nb_qubits,
                    )
                else:
                    provider_circuit = (
                        circuit + QCircuit(binding.noises or [])
                    ).to_other_device(device)
                translated_circuits[circuit_id] = provider_circuit

            if measurement.optimize_measurement:
                grouping = measurement.get_pauli_grouping()
                for group_index, group in enumerate(grouping):
                    parameter_schema = (
                        tuple(sorted(str(parameter) for parameter in values))
                        if values is not None
                        else None
                    )
                    key = (
                        circuit_id,
                        id(measurement),
                        group_index,
                        parameter_schema,
                    )
                    entry = grouped_entries.get(key)
                    if entry is None:
                        entry = {
                            "circuit": provider_circuit
                            + _group_measurement_circuit(
                                measurement, group, circuit.nb_qubits
                            ),
                            "input_sets": [],
                            "contexts": [],
                        }
                        grouped_entries[key] = entry
                    if values is not None:
                        entry[
                            "input_sets"
                        ].append(  # pyright: ignore[reportAttributeAccessIssue]
                            _braket_values(values)
                        )
                    entry[
                        "contexts"
                    ].append(  # pyright: ignore[reportAttributeAccessIssue]
                        BraketPauliGroupingContext(
                            circuit=circuit,
                            measurement=measurement,
                            values=values,
                            result_index=result_index,
                            optimized=True,
                            group_index=group_index,
                            monomials=tuple(group),
                            eigenvalues={
                                monomial.name: pauli_monomial_eigenvalues(monomial)
                                for monomial in group
                            },
                        )
                    )
                continue

            for observable_index, observable in enumerate(measurement.observables):
                provider_observable = observable.pauli_string.to_other_language(
                    Language.BRAKET
                )
                span = (
                    len(provider_observable)
                    if isinstance(provider_observable, Sum)
                    else 1
                )
                entries.append(
                    BraketBinding(
                        provider_circuit,
                        input_sets=(
                            [_braket_values(values)] if values is not None else None
                        ),
                        observables=(
                            provider_observable
                            if isinstance(provider_observable, Sum)
                            else [provider_observable]
                        ),
                    )
                )
                entry_contexts.append(
                    [
                        BraketPauliGroupingContext(
                            circuit=circuit,
                            measurement=measurement,
                            values=values,
                            result_index=result_index,
                            optimized=False,
                            observable_index=observable_index,
                            executable_span=span,
                        )
                    ]
                )

        for entry in grouped_entries.values():
            contexts: list[BraketPauliGroupingContext] = entry["contexts"]
            input_sets: list[dict[str, float]] = entry["input_sets"]
            if input_sets:
                entries.append(BraketBinding(entry["circuit"], input_sets=input_sets))
                entry_contexts.append(contexts)
            else:
                # A bare circuit has no binding axis that can retain multiplicity.
                # Repeated logical executions therefore need distinct entries.
                for context in contexts:
                    entries.append(entry["circuit"])
                    entry_contexts.append([context])

        return ProgramSet(entries, binding.shots), entry_contexts

    def _grouped_cb_to_programset(
        binding: "CircuitBinding", device: "AWSDevice"
    ) -> "tuple[ProgramSet, list[list[BraketPauliGroupingContext]]]":
        """Translate a binding while sharing one Braket program per circuit.

        The MPQP binding is expanded once, then executions are grouped by
        circuit and by their observable layout. Parameter sets whose
        observable layouts match are represented by one Braket
        :class:`CircuitBinding`. ZIP layouts that are not Cartesian remain in
        separate entries so Braket cannot introduce extra executions.

        Args:
            binding: Binding containing the circuits, values and measurements.
            device: AWS device targeted by the program set.

        Returns:
            The grouped program set and one ordered context list per program
            entry. Each context stores the executable span, MPQP result index
            and, for expectation measurements, observable index needed to
            reconstruct the original results.
        """

        from collections import OrderedDict

        from braket.circuits.observables import Sum
        from braket.program_sets import ProgramSet
        from braket.circuits import Observable as BraketObservable
        from braket.circuits import Circuit as BraketCircuit
        from mpqp.core import Language, QCircuit
        from mpqp.core.instruction import BasisMeasure

        if binding.shots == 0:
            raise ValueError("Braket's ProgramSets are not available with shots at 0.")

        def parameter_key(
            values: BindingParameters | None,
        ) -> tuple[tuple[str, str], ...] | None:
            """Build a stable identity key for an MPQP parameter mapping.

            Args:
                values: Parameter mapping attached to one unrolled execution,
                    or ``None`` for a non-parametric execution.

            Returns:
                A sorted tuple based on parameter names and value
                representations. ``None`` is preserved as its own key.

            Note:
                ``repr`` is used only for grouping. The original values are
                retained unchanged in the result reconstruction context.
            """
            if values is None:
                return None
            return tuple(
                sorted((str(key), repr(value)) for key, value in values.items())
            )

        # A translated measurement is immutable for the duration of this
        # translation. Caching it here avoids repeating matrix-to-Pauli and
        # provider conversions for every parameter set.
        observable_cache: dict[
            int, list[tuple[ExpectationMeasure, BraketObservable, int, int]]
        ] = {}

        def observable_translation_info(
            measure: ExpectationMeasure,
        ) -> list[tuple[ExpectationMeasure, BraketObservable, int, int]]:
            """Translate the observables of one expectation measurement.

            Each returned item contains the original measurement, translated
            Braket observable, executable span and observable index required
            to reconstruct the MPQP result. All items from the same
            :class:`ExpectationMeasure` are aggregated into one MPQP
            :class:`Result`. A Braket ``Sum`` occupies several physical
            executables but still produces one expectation value.

            Args:
                measure: Expectation measurement to translate.

            Returns:
                Ordered observable translation information suitable for a
                Braket ``CircuitBinding``. Repeated calls for the same
                measurement instance reuse the cached translation.

            Warns:
                UserWarning: If an observable matrix must first be converted
                    to a Pauli string.
            """
            cached = observable_cache.get(id(measure))
            if cached is not None:
                return cached
            if any(observable.is_matrix_set() for observable in measure.observables):
                warn(
                    "To translate observables from CircuitBindings to braket we "
                    "need to translate the matrix to pauli string. This process "
                    "might impact performances on big matrices."
                )
            translation_info: list[
                tuple[ExpectationMeasure, BraketObservable, int, int]
            ] = []
            for observable_index, observable in enumerate(measure.observables):
                translated = observable.pauli_string.to_other_language(Language.BRAKET)
                translation_info.append(
                    (
                        measure,
                        translated,
                        len(translated) if isinstance(translated, Sum) else 1,
                        observable_index,
                    )
                )
            observable_cache[id(measure)] = translation_info
            return translation_info

        # First and only pass over the MPQP executions. Observable executions
        # share the bare circuit; sample executions also include the measurement
        # in the provider-circuit identity.
        circuit_groups: "OrderedDict[Any, Any]" = OrderedDict()
        translated_circuits: dict[int, BraketCircuit] = {}

        result_index = 0
        for circuit, values, measure in binding.unroll():
            circuit_id = id(circuit)
            translated_circuit = translated_circuits.get(circuit_id)
            if translated_circuit is None:
                if not isinstance(circuit.transpiled_circuit, BraketCircuit):
                    translated_circuit = (
                        circuit + QCircuit(binding.noises)
                    ).to_other_device(device=device)
                else:
                    from mpqp.execution.providers.aws import (
                        apply_noise_to_braket_circuit,
                    )

                    translated_circuit = apply_noise_to_braket_circuit(
                        circuit.transpiled_circuit,
                        noises=binding.noises if binding.noises is not None else [],
                        nb_qubits=circuit.nb_qubits,
                    )
                if TYPE_CHECKING:
                    assert isinstance(translated_circuit, BraketCircuit)
                translated_circuits[circuit_id] = translated_circuit

            measurement_id = id(measure) if isinstance(measure, BasisMeasure) else None
            group_key: tuple[int, int | None] = (circuit_id, measurement_id)
            group_data = circuit_groups.get(group_key)
            if group_data is None:
                provider_circuit = translated_circuit
                if isinstance(measure, BasisMeasure):
                    provider_circuit = provider_circuit + QCircuit(
                        [measure], nb_qubits=circuit.nb_qubits
                    ).to_other_language(Language.BRAKET)
                group_data = (circuit, provider_circuit, OrderedDict())
                circuit_groups[group_key] = group_data

            parameters: "OrderedDict[Any, Any]" = group_data[
                2
            ]  # Parameter groups indexed by their values.
            values_key = parameter_key(values)
            parameter_data: (
                tuple[
                    BindingParameters | None,
                    list[
                        tuple[
                            Measure | None,
                            BraketObservable | None,
                            int,
                            int,
                            int | None,
                        ]
                    ],
                ]
                | None
            ) = parameters.get(values_key)
            if parameter_data is None:
                parameter_data = (values, [])
                parameters[values_key] = parameter_data
            translation_info: list[
                tuple[
                    Measure | None,
                    BraketObservable | None,
                    int,
                    int,
                    int | None,
                ]
            ] = parameter_data[
                1
            ]  # Observable translation information.
            if isinstance(measure, ExpectationMeasure):
                observable_info = observable_translation_info(measure)
                translation_info.extend(
                    (
                        info[0],  # Original ExpectationMeasure.
                        info[1],  # Translated Braket observable.
                        info[2],  # Number of Braket executables.
                        result_index,
                        info[3],  # Observable index in the measurement.
                    )
                    for info in observable_info
                )
                result_index += 1
            else:
                translation_info.append((measure, None, 1, result_index, None))
                result_index += 1

        entries: list[BraketCircuit | BraketBinding] = []
        # Contexts retain the MPQP execution identity while provider entries
        # are reorganized for Braket.
        entry_contexts: list[list[BraketPauliGroupingContext]] = []

        for group_data in circuit_groups.values():
            original_circuit: QCircuit = group_data[0]  # Original MPQP circuit.
            provider_circuit: BraketCircuit = group_data[
                1
            ]  # Translated Braket circuit.
            parameter_groups_by_values: "OrderedDict[Any, Any]" = group_data[
                2
            ]  # Parameter groups indexed by their values.
            # Parameter rows can share a Braket CircuitBinding exactly when
            # they have the same ordered observable axis. This is the common
            # PRODUCT case; ZIP pairs naturally fall into distinct layouts.
            layouts: "OrderedDict[Any, list[Any]]" = OrderedDict()
            for parameter_group in parameter_groups_by_values.values():
                parameter_values: BindingParameters | None = parameter_group[
                    0
                ]  # MPQP parameter mapping.
                translation_info: list[
                    tuple[
                        Measure | None,
                        BraketObservable | None,
                        int,
                        int,
                        int | None,
                    ]
                ] = parameter_group[
                    1
                ]  # Observable translation information.
                # Record the measurement, provider observable and executable
                # span of every item to identify compatible Braket layouts.
                layout_key: tuple[tuple[int, int, int], ...] = tuple(
                    (
                        id(info[0]),  # Original measurement identity.
                        id(info[1]),  # Translated observable identity.
                        info[2],  # Number of Braket executables.
                    )
                    for info in translation_info
                )
                schema: tuple[str, ...] = tuple(
                    sorted(str(key) for key in (parameter_values or {}))
                )
                layouts.setdefault((schema, layout_key), []).append(parameter_group)

            for stored_parameter_groups in layouts.values():
                # Restore the precise shape hidden by the intentionally loose
                # layout container typing.
                parameter_groups: list[
                    tuple[
                        BindingParameters | None,
                        list[
                            tuple[
                                Measure | None,
                                BraketObservable | None,
                                int,
                                int,
                                int | None,
                            ]
                        ],
                    ]
                ] = stored_parameter_groups
                translation_info: list[
                    tuple[
                        Measure | None,
                        BraketObservable | None,
                        int,
                        int,
                        int | None,
                    ]
                ] = parameter_groups[0][1]

                # Braket accepts either a list of simple observables or one
                # Sum. A list containing Sum objects is deliberately rejected
                # by Braket, so each Sum forms its own entry.
                partitions: list[list[int]] = []
                simple_partition: list[int] = []
                for info_index, info in enumerate(translation_info):
                    if isinstance(info[1], Sum):  # Translated Braket observable.
                        if simple_partition:
                            partitions.append(simple_partition)
                            simple_partition = []
                        partitions.append([info_index])
                    else:
                        simple_partition.append(info_index)
                if simple_partition or not partitions:
                    partitions.append(simple_partition)

                for partition in partitions:
                    partition_has_observables = any(
                        translation_info[info_index][1] is not None
                        for info_index in partition
                    )
                    input_sets: list[dict[str, float]] = []
                    for parameter_group in parameter_groups:
                        parameter_values = parameter_group[0]  # MPQP parameter mapping.
                        if parameter_values is not None:
                            converted_values = _braket_values(parameter_values)
                            # Without an observable axis, each item in the
                            # partition is a distinct execution. Keep repeated
                            # parameter rows so Braket performs every run,
                            # including byte-for-byte identical ones.
                            repeat_count = (
                                1 if partition_has_observables else len(partition)
                            )
                            input_sets.extend(
                                converted_values.copy() for _ in range(repeat_count)
                            )
                    provider_observable_list: list[BraketObservable] = []
                    for info_index in partition:
                        provider_observable = translation_info[info_index][
                            1
                        ]  # Translated Braket observable.
                        if provider_observable is not None:
                            provider_observable_list.append(provider_observable)
                    provider_observables: list[BraketObservable] | Sum | None = (
                        provider_observable_list
                    )
                    if len(partition) == 1:
                        single_provider_observable = translation_info[partition[0]][
                            1
                        ]  # Translated Braket observable.
                        if isinstance(single_provider_observable, Sum):
                            provider_observables = single_provider_observable
                    if not provider_observable_list:
                        provider_observables = None

                    contexts: list[BraketPauliGroupingContext] = []
                    for parameter_group in parameter_groups:
                        parameter_values = parameter_group[0]  # MPQP parameter mapping.
                        parameter_translation_info = parameter_group[
                            1
                        ]  # Observable translation information.
                        for info_index in partition:
                            info = parameter_translation_info[info_index]
                            contexts.append(
                                BraketPauliGroupingContext(
                                    circuit=original_circuit,
                                    measurement=info[0],
                                    values=parameter_values,
                                    result_index=info[3],
                                    optimized=False,
                                    observable_index=info[4],
                                    executable_span=info[2],
                                )
                            )

                    if input_sets or provider_observables is not None:
                        entries.append(
                            BraketBinding(
                                provider_circuit,
                                input_sets=input_sets or None,
                                observables=provider_observables,
                            )
                        )
                        entry_contexts.append(contexts)
                    else:
                        # A bare circuit has no Braket binding axis on which to
                        # encode multiplicity. Add it once per logical
                        # occurrence instead of collapsing identical runs.
                        for context in contexts:
                            entries.append(provider_circuit)
                            entry_contexts.append([context])

        return ProgramSet(entries, binding.shots), entry_contexts

    def circuitbinding_to_programset(
        binding: "CircuitBinding", device: "AWSDevice"
    ) -> "tuple[ProgramSet, list[list[BraketPauliGroupingContext]]]":
        """Convert an MPQP circuit binding to an AWS Braket ``ProgramSet``.

        Args:
            binding: Binding containing the circuits, values and measurements
                to translate.
            device: AWS device targeted by the program set.

        Returns:
            The translated program set together with ordered contexts
            used by the AWS result adapter.
        """
        if binding.optimize_measurement:
            return _cb_to_programset_pauli_grouping(binding, device)
        return _grouped_cb_to_programset(binding, device)
