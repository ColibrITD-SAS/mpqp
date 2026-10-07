from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
)
from warnings import warn

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.gate_decomposition import resolve_gate
from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
)
from mpqp.environment.var_cache import (
    InstalledProviders,
)

if InstalledProviders.BRAKET in _INSTALLED_MPQP_PROVIDERS:
    from braket.circuits import Circuit as braket_Circuit
    from braket.circuits import Observable as BraketObservable
    from braket.program_sets import CircuitBinding as BraketBinding

    from mpqp.core.circuit import QCircuit
    from mpqp.core.circuitbinding import BindingParameters
    from mpqp.core.instruction.measurement.expectation_value import ExpectationMeasure
    from mpqp.core.instruction.measurement.measure import Measure

    if TYPE_CHECKING:
        from braket.program_sets import ProgramSet

        from mpqp.core.circuitbinding import CircuitBinding
        from mpqp.execution.devices import AvailableDevice

    def braket_to_mpqp(qcircuit: "braket_Circuit") -> "QCircuit":

        from braket.circuits.serialization import IRType
        from braket.ir.openqasm.program_v1 import Program

        from mpqp.core.languages import Language
        from mpqp.translation.qasm.open_qasm_2_and_3 import open_qasm_3_to_2
        from mpqp.translation.qasm.qasm_to_braket import braket_noise_to_mpqp
        from mpqp.translation.qasm.qasm_to_mpqp import qasm2_parse

        assert isinstance(qcircuit, braket_Circuit)
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
    ) -> braket_Circuit:
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

        if len(circuit.noises) != 0:
            if any(isinstance(instr, CRk) for instr in circuit.instructions):
                raise NotImplementedError(
                    "Cannot simulate noisy circuit with CRk gate due to "
                    "an error on AWS Braket side."
                )

        braket_circuit = braket_Circuit()

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

    @overload
    def _cb_to_programset_pauli_grouping(
        binding: "CircuitBinding", device: "AvailableDevice", depth: Literal[0]
    ) -> "tuple[ProgramSet, list[tuple[Any]]]": ...
    @overload
    def _cb_to_programset_pauli_grouping(
        binding: "CircuitBinding", device: "AvailableDevice", depth: Literal[1, 2]
    ) -> "CircuitBinding": ...
    @overload
    def _cb_to_programset_pauli_grouping(
        binding: "CircuitBinding", device: "AvailableDevice", depth: Literal[0, 1, 2]
    ) -> "CircuitBinding | tuple[ProgramSet, list[tuple[Any]]]": ...
    def _cb_to_programset_pauli_grouping(
        binding: "CircuitBinding",
        device: "AvailableDevice",
        depth: Literal[0, 1, 2] = 0,
    ) -> "tuple[ProgramSet, list[tuple[Any]]] | CircuitBinding":
        """Translate a binding to a Braket program set using Pauli grouping.

        Nested bindings are translated recursively and retain their translated
        circuits, observables and variables until the root binding assembles
        the final program set.

        Args:
            binding: Circuit binding to translate.
            device: AWS device targeted by the translated circuits.
            depth: Nesting depth of ``binding``. Zero denotes the root binding.

        Returns:
            At the root, the Braket ``ProgramSet`` and the context required to
            rebuild MPQP results. At a nested depth, the binding populated with
            its translated provider data.

        Raises:
            ValueError: If both a nested binding and its parent define values
                or observables for the same execution axis.
        """
        from braket.program_sets import ProgramSet
        from braket.circuits import Circuit as braket_Circuit
        from mpqp.core import Language, QCircuit
        from mpqp.core.circuitbinding import CircuitBinding, BindingMode

        # translate inner circuits to braket and CB's elements to Braket
        translated: list[CircuitBinding | braket_Circuit] = []
        for c in binding.circuits:
            if isinstance(c, QCircuit):
                translated.append(c.to_other_language(Language.BRAKET))
            else:
                translation = _cb_to_programset_pauli_grouping(c, device, depth + 1)  # type: ignore

                if isinstance(translation, list):
                    translated.extend(translation)
                else:
                    translated.append(translation)

        # translate var
        var = []
        if binding.value:
            from copy import deepcopy

            var = deepcopy(binding.value)
            converted = []
            for variables in var:
                current = {}
                for key in variables.keys():
                    val = variables[key]  # pyright: ignore[reportArgumentType]
                    current.update({str(key): [val]})
                converted.append(current)
            var = converted

        from braket.program_sets import CircuitBinding as BraketBinding

        obs = []
        if binding.measurements:

            for m in binding.measurements:
                from mpqp.core.instruction import ExpectationMeasure

                if isinstance(m, ExpectationMeasure):
                    if any([o.is_matrix_set() for o in m.observables]):
                        # This is because of braket's programSet limitations
                        warn(
                            "To translate observables from CircuitBindings to braket we need to translate the matrix to pauli string. This process might impact performances on big matrices."
                        )
                    from mpqp.tools.pauli_grouping import (
                        find_qubitwise_rotations,
                        pauli_monomial_eigenvalues,
                    )

                    grouping = m.get_pauli_grouping()
                    transpiled_pre_measures = [
                        QCircuit(find_qubitwise_rotations(group)).to_other_language(
                            Language.BRAKET
                        )
                        for group in grouping
                    ]
                    eigenvalues = [
                        {
                            monom.name: pauli_monomial_eigenvalues(monom)
                            for monom in group
                        }
                        for group in grouping
                    ]
                    obs.append(
                        (
                            m.observables,
                            eigenvalues,
                            transpiled_pre_measures,
                            grouping,
                        )
                    )
                else:
                    obs.append(m.to_other_language(Language.BRAKET))

        if depth != 0:  # If the circuitBinding is embedded store the translated data.
            binding._translated_circuits = translated  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
            binding._translated_observables = (  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
                obs
            )
            binding._translated_variables = (  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
                var
            )
            return binding

        executable_list = []
        context = []  # this list holds information to sort the results afterwards
        # This list helps differentiate exp_values later because braket creates 1 job per pauli MONOMIALS so we will need to group them afterwards.

        # i is used only if the binding mode is zip
        if len(translated) == 1 and (
            not isinstance(translated[0], CircuitBinding)
            or len(translated[0].circuits) == 1
        ):
            # if i == -1 then we're in the case of a single circuit in the binding.
            # Otherwise it'll iterate of the translated circuit (and bindings) and apply the values and obs accordingly
            i = -1
        else:
            i = 0
        if binding.mode == BindingMode.ZIP:
            if not var and not obs:
                executables = [(None, None)] * (1 if i == -1 else len(translated))
            else:
                executables = list(
                    zip(
                        var or [None] * len(obs),
                        obs or [None] * len(var),
                    )
                )
        else:
            from itertools import product

            executables = list(product(var or [None], obs or [None]))

        for values, observable in executables:
            if binding.mode == BindingMode.PRODUCT:
                for t in translated:
                    if isinstance(t, CircuitBinding):
                        if (
                            t._translated_observables  # pyright: ignore[reportPrivateUsage]
                            and observable
                        ):
                            raise ValueError(
                                "Cannot declare an observable both inside a CircuitBinding and outside"
                            )
                        if (
                            t._translated_variables  # pyright: ignore[reportPrivateUsage]
                            and values
                        ):
                            raise ValueError(
                                "Cannot declare variables both inside a CircuitBinding and outside"
                            )
                        from itertools import product

                        inside_executables = list(
                            product(
                                t._translated_observables  # pyright: ignore[reportPrivateUsage]
                                or [observable],
                                t._translated_variables  # pyright: ignore[reportPrivateUsage]
                                or [values],
                            )
                        )

                        for inside_observable, inside_val in inside_executables:
                            for circuitindex, c in enumerate(
                                t._translated_circuits  # pyright: ignore[reportPrivateUsage,reportArgumentType]
                            ):
                                if TYPE_CHECKING:
                                    assert isinstance(c, braket_Circuit)
                                mpqp_circuit = t.circuits[circuitindex]
                                (
                                    observables,
                                    eigenvalues,
                                    transpiled_pre_measures,
                                    grouping,
                                ) = inside_observable  # pyright: ignore[reportGeneralTypeIssues]
                                for index in range(len(grouping)):
                                    executable_list.append(
                                        BraketBinding(
                                            c + transpiled_pre_measures[index],
                                            input_sets=inside_val,
                                        )
                                        if inside_val
                                        else c + transpiled_pre_measures[index]
                                    )

                                    context.append(
                                        (
                                            mpqp_circuit,
                                            observables,
                                            inside_val,
                                            eigenvalues[index],
                                            grouping[index],
                                        )
                                    )

                    else:
                        if observable:
                            (
                                observables,
                                eigenvalues,
                                transpiled_pre_measures,
                                grouping,
                            ) = observable  # pyright: ignore[reportGeneralTypeIssues]

                            mpqp_circuit = binding.circuits[translated.index(t)]
                            for index in range(len(grouping)):
                                if values:
                                    executable_list.append(
                                        BraketBinding(
                                            t + transpiled_pre_measures[index],
                                            input_sets=values,
                                        )
                                    )
                                else:
                                    executable_list.append(
                                        t + transpiled_pre_measures[index]
                                    )
                                context.append(
                                    (
                                        mpqp_circuit,
                                        observables,
                                        values,
                                        eigenvalues[index],
                                        grouping[index],
                                    )
                                )
                        else:
                            mpqp_circuit = binding.circuits[translated.index(t)]
                            if values:
                                executable_list.append(
                                    BraketBinding(
                                        t,
                                        input_sets=values,
                                    )
                                )
                            else:
                                executable_list.append(t)
                            context.append(
                                (
                                    mpqp_circuit,
                                    values,
                                )
                            )
            else:
                if isinstance(
                    translated[i], CircuitBinding
                ):  # ZIP to Binding ==> distribute upper binding to embedded binding
                    if (
                        translated[
                            i
                        ]._translated_observables  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
                        and observable
                    ):
                        raise ValueError(
                            "Cannot declare observables both inside a CircuitBinding and outside"
                        )

                    if (
                        translated[
                            i
                        ]._translated_variables  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
                        and values
                    ):
                        raise ValueError(
                            "Cannot declare variables both inside a CircuitBinding and outside"
                        )

                    from itertools import product

                    inside_executables = list(
                        product(
                            translated[
                                i
                            ]._translated_observables  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
                            or [observable],
                            translated[
                                i
                            ]._translated_variables  # pyright: ignore[reportAttributeAccessIssue,reportPrivateUsage]
                            or [values],
                        )
                    )

                    if i == -1:
                        binding = translated[i]  # pyright: ignore[reportAssignmentType]
                        if TYPE_CHECKING:
                            assert isinstance(binding, CircuitBinding)
                            assert (
                                binding._translated_circuits  # pyright: ignore[reportPrivateUsage]
                            )
                        for circuitindex, c in enumerate(
                            binding._translated_circuits  # pyright: ignore[reportPrivateUsage]
                        ):
                            mpqp_circuit = binding.circuits[circuitindex]

                            for inside_observable, inside_val in inside_executables:
                                if TYPE_CHECKING:
                                    assert isinstance(c, braket_Circuit)
                                (
                                    observables,
                                    eigenvalues,
                                    transpiled_pre_measures,
                                    grouping,
                                ) = inside_observable  # pyright: ignore[reportGeneralTypeIssues]
                                for index in range(len(grouping)):
                                    executable_list.append(
                                        BraketBinding(
                                            translated[
                                                i
                                            ]._translated_circuits[  # pyright: ignore
                                                0
                                            ],
                                            input_sets=inside_val,
                                        )
                                    )
                                    context.append(
                                        (
                                            mpqp_circuit,
                                            observables,
                                            inside_val,
                                            eigenvalues[index],
                                            grouping[index],
                                        )
                                    )
                            else:
                                for inside_observable, inside_val in inside_executables:
                                    if TYPE_CHECKING:
                                        assert isinstance(c, braket_Circuit)
                                    mpqp_circuit = binding.circuits[translated.index(c)]
                                    (
                                        observables,
                                        eigenvalues,
                                        transpiled_pre_measures,
                                        grouping,
                                    ) = inside_observable  # pyright: ignore[reportGeneralTypeIssues]
                                    for index in range(len(grouping)):
                                        executable_list.append(
                                            BraketBinding(
                                                c + transpiled_pre_measures[index],
                                                input_sets=inside_val,
                                            )
                                        )

                                        context.append(
                                            (
                                                mpqp_circuit,
                                                observables,
                                                inside_val,
                                                eigenvalues[index],
                                                grouping[index],
                                            )
                                        )
                else:
                    if i == -1:
                        executable_list.append(
                            BraketBinding(
                                translated[0],  # pyright: ignore
                                input_sets=values,
                                observables=observable,
                            )
                        )
                    else:
                        executable_list.append(
                            BraketBinding(
                                translated[i],  # pyright: ignore
                                input_sets=values,
                                observables=observable,
                            )
                        )
                        i += 1

        from braket.program_sets import ProgramSet

        ps = ProgramSet(executable_list, binding.shots)
        return ps, context

    def _grouped_cb_to_programset(
        binding: "CircuitBinding", device: "AvailableDevice"
    ) -> "tuple[ProgramSet, list[list[tuple[QCircuit, Measure | None, BindingParameters | None, int, int, int | None]]]]":
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
            entry. Each context tuple stores the executable span, MPQP result
            index and, for expectation measurements, observable index needed
            to reconstruct the original results.
        """

        # Note: the following function was made with help of chatGPT.
        # It combined elements from the previous version of this function and the qiskit's pubs translation.
        # Due to Braket splitting the execution of Sums (multiple monomials observable) we need to keep a LOT of information
        # to be able to group up the executions into single Jobs then Results.
        # This means that this function creates and handle crazy big datatypes to be able to group up bindings so that we have the best possible result.
        # It uses a LOT of ints to track different infos (observable's spans, result indexes or various ids of measurements and parameters).
        # I did my best to document the whole process but the code is still fairly confusing.
        from numbers import Real
        from collections import OrderedDict

        from braket.circuits.observables import Sum
        from braket.program_sets import ProgramSet

        from mpqp.core import Language, QCircuit
        from mpqp.core.instruction import BasisMeasure

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

        def braket_values(values: BindingParameters) -> dict[str, float]:
            """Convert MPQP parameter values to a Braket input mapping.

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
            converted: dict[str, float] = {}
            for key, value in values.items():
                if not isinstance(value, Real):
                    raise TypeError(
                        "Braket circuit parameters must be real numbers, "
                        f"but parameter {key!s} has value {value!r}."
                    )
                converted[str(key)] = float(value)
            return converted

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
        translated_circuits: dict[int, braket_Circuit] = {}
        result_index = 0
        for circuit, values, measure in binding.unroll():
            circuit_id = id(circuit)
            translated_circuit = translated_circuits.get(circuit_id)
            if translated_circuit is None:
                if not isinstance(circuit.transpiled_circuit, braket_Circuit):
                    circuit.transpiled_circuit = circuit.to_other_device(device=device)
                translated_circuit = circuit.transpiled_circuit
                if TYPE_CHECKING:
                    assert isinstance(translated_circuit, braket_Circuit)
                translated_circuits[circuit_id] = translated_circuit

            measurement_id = id(measure) if isinstance(measure, BasisMeasure) else None
            group_key: tuple[int, int | None] = (circuit_id, measurement_id)
            group_data = circuit_groups.get(group_key)
            if group_data is None:
                provider_circuit = translated_circuit
                if isinstance(measure, BasisMeasure):
                    provider_circuit = provider_circuit + QCircuit(
                        [measure]
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

        entries: list[braket_Circuit | BraketBinding] = []
        # This list holds the necessary contexts to build the according jobs and Results afterwards.
        # in order it has: the circuit, measure, parameters, number of single execution the job is needed if multiple monomials (ex: Obs(X-Z) = 2 because braket doesn't group here),
        # result index and finally observable index if needed.
        entry_contexts: list[
            list[
                tuple[
                    QCircuit,
                    Measure | None,
                    BindingParameters | None,
                    int,
                    int,
                    int | None,
                ]
            ]
        ] = []

        for group_data in circuit_groups.values():
            original_circuit: QCircuit = group_data[0]  # Original MPQP circuit.
            provider_circuit: braket_Circuit = group_data[
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
                    input_sets: list[dict[str, float]] = []
                    for parameter_group in parameter_groups:
                        parameter_values = parameter_group[0]  # MPQP parameter mapping.
                        if parameter_values is not None:
                            input_sets.append(braket_values(parameter_values))
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

                    if input_sets or provider_observables is not None:
                        entries.append(
                            BraketBinding(
                                provider_circuit,
                                input_sets=input_sets or None,
                                observables=provider_observables,
                            )
                        )
                    else:
                        entries.append(provider_circuit)

                    contexts: list[
                        tuple[
                            QCircuit,
                            Measure | None,
                            BindingParameters | None,
                            int,
                            int,
                            int | None,
                        ]
                    ] = []
                    for parameter_group in parameter_groups:
                        parameter_values = parameter_group[0]  # MPQP parameter mapping.
                        parameter_translation_info = parameter_group[
                            1
                        ]  # Observable translation information.
                        for info_index in partition:
                            info = parameter_translation_info[info_index]
                            contexts.append(
                                (
                                    original_circuit,
                                    info[0],  # Original measurement.
                                    parameter_values,
                                    info[2],  # Number of Braket executables.
                                    info[3],  # MPQP result index.
                                    info[4],  # Observable index in the measurement.
                                )
                            )
                    entry_contexts.append(contexts)

        return ProgramSet(entries, binding.shots), entry_contexts

    def circuitbinding_to_programset(
        binding: "CircuitBinding", device: "AvailableDevice"
    ) -> "tuple[ProgramSet, list[Any]]":
        """Convert an MPQP circuit binding to an AWS Braket ``ProgramSet``.

        Args:
            binding: Binding containing the circuits, values and measurements
                to translate.
            device: AWS device targeted by the program set.

        Returns:
            The translated program set together with ordered context tuples
            used by the AWS result adapter.
        """

        # Will be used when pauli grouping is implemented
        """

        from mpqp.core.instruction import ExpectationMeasure
        if binding.measurements:
            if (
                isinstance(binding.measurements[0], ExpectationMeasure)
                and binding.measurements[0].optimize_measurement
            ):
                return _cb_to_programset_pauli_grouping(binding, device, True)"""
        return _grouped_cb_to_programset(binding, device)
