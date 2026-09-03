from typing import TYPE_CHECKING, Any, Literal, overload
from warnings import warn

from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
    InstalledProviders,
)
from mpqp.execution.job import Job

if InstalledProviders.BRAKET in _INSTALLED_MPQP_PROVIDERS:
    if TYPE_CHECKING:
        from braket.circuits import Circuit as braket_Circuit
        from braket.program_sets import ProgramSet
        from mpqp.core.instruction.gates.native_gates import NativeGate
        from mpqp.core.circuit import QCircuit, CircuitBinding
        from mpqp.execution.devices import AvailableDevice

    def braket_to_mpqp(qcircuit: "braket_Circuit") -> "QCircuit":

        from braket.circuits.serialization import IRType
        from braket.ir.openqasm.program_v1 import Program

        from mpqp.translation.qasm.open_qasm_2_and_3 import open_qasm_3_to_2
        from mpqp.translation.qasm.qasm_to_braket import braket_noise_to_mpqp
        from mpqp.translation.qasm.qasm_to_mpqp import qasm2_parse
        from braket.circuits import Circuit as braket_Circuit
        from mpqp.core.languages import Language

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

    def mpqp_to_braket(
        circuit: "QCircuit",
        skip_pre_measure: bool = False,
        skip_measurements: bool = False,
        authorized_gates: set[type["NativeGate"]] | None = None,
    ) -> "braket_Circuit":
        """Translate a MPQP circuit to a Braket equivalent.

        Note:
            If the circuit contains ComposedGate type instructions you need to include the gate in the authorize_gates set for it to avoid decomposition, otherwise, the translation will always
            try to use the gate's decomposition (see example).
        Args:
            circuit: The original MPQP circuit to be translated.
            skip_pre_measure: If set at True will translate the circuit without its pre-measurement circuit (see QCircuit.to_other_language for more information).
            skip_measurements: If set at True will translate the circuit without any measurement.
            authorized_gates: The set of gates allowed on the circuit, if the circuit contains any other gates it raises a ValueError.

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
        from mpqp.execution.providers.aws import apply_noise_to_braket_circuit
        from mpqp.core.instruction.gates.custom_controlled_gate import (
            CustomControlledGate,
        )
        from mpqp.core.instruction.gates.custom_gate import CustomGate
        from mpqp.core.instruction.gates.gate import Gate
        from mpqp.core.instruction.gates.native_gates import CRk
        from mpqp.core.languages import Language
        from mpqp.core.instruction import (
            Measure,
            Breakpoint,
            Barrier,
            ControlledGate,
            BasisMeasure,
        )
        from mpqp.core.circuit import QCircuit

        if authorized_gates is None:
            authorized_gates = set()
        if len(circuit.noises) != 0:
            if any(isinstance(instr, CRk) for instr in circuit.instructions):
                raise NotImplementedError(
                    "Cannot simulate noisy circuit with CRk gate due to "
                    "an error on AWS Braket side."
                )
        from braket.circuits import Circuit as BracketCircuit

        braket_circuit = BracketCircuit()

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
                from mpqp.gates import Id
                from copy import deepcopy

                circuit = QCircuit(
                    [
                        Id(qubit)
                        for qubit in range(circuit.nb_qubits)
                        if qubit not in used_qubits
                    ],
                    nb_qubits=circuit.nb_qubits,
                ) + deepcopy(circuit)

        for instruction in circuit.instructions:
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
                from mpqp.translation.utils import verify_convert_instructions

                instr = verify_convert_instructions(instruction, authorized_gates)
            else:
                instr = [instruction]

            for instruction in instr:
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
        binding: "CircuitBinding", device: "AvailableDevice"
    ) -> "tuple[ProgramSet, list[tuple[Any]]]": ...
    @overload
    def _cb_to_programset_pauli_grouping(
        binding: "CircuitBinding", device: "AvailableDevice", depth: Literal[1, 2]
    ) -> "CircuitBinding": ...
    def _cb_to_programset_pauli_grouping(
        binding: "CircuitBinding", device: "AvailableDevice", programSet: bool = True
    ) -> "tuple[ProgramSet, list[tuple[Any]]] | CircuitBinding":
        from braket.program_sets import ProgramSet
        from braket.circuits import Circuit as braket_Circuit
        from mpqp.core import Language, QCircuit
        from mpqp.core.circuit import CircuitBinding, BindingMode

        # translate inner circuits to braket and CB's elements to Braket
        translated: list[CircuitBinding | braket_Circuit] = []
        for c in binding.circuits:
            if isinstance(c, QCircuit):
                translated.append(c.to_other_language(Language.BRAKET))
            else:
                translation = _cb_to_programset_pauli_grouping(
                    c, device, programSet=False
                )

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
                    if any([o.is_matrix() for o in m.observables]):
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

        if (
            not programSet
        ):  # If the circuitBinding is embedded store the translated data.
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

    @overload
    def _cb_to_programset(
        binding: "CircuitBinding", device: "AvailableDevice", depth: Literal[0]
    ) -> "tuple[ProgramSet, list[tuple[Any]]]": ...
    @overload
    def _cb_to_programset(
        binding: "CircuitBinding", device: "AvailableDevice"
    ) -> "tuple[ProgramSet, list[tuple[Any]]]": ...
    @overload
    def _cb_to_programset(
        binding: "CircuitBinding",
        device: "AvailableDevice",
        depth: Literal[0, 1, 2],
    ) -> "CircuitBinding": ...
    def _cb_to_programset(
        binding: "CircuitBinding",
        device: "AvailableDevice",
        depth: Literal[0, 1, 2] = 0,
    ) -> "tuple[ProgramSet, list[tuple[Any]]] | CircuitBinding":
        from braket.program_sets import ProgramSet
        from braket.circuits import Circuit as braket_Circuit

        from mpqp.core import Language
        from mpqp.core.circuit import CircuitBinding, QCircuit

        # translate inner circuits to braket and CB's elements to Braket
        translated: list[CircuitBinding | braket_Circuit] = []
        for c in binding.circuits:
            if isinstance(c, QCircuit):
                translated.append(c.to_other_language(Language.BRAKET))
            else:
                print(c)
                translation = _cb_to_programset(c, device, depth=depth + 1)  # type: ignore

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
        from mpqp.core.instruction import ExpectationMeasure

        obs = []
        if binding.measurements:

            for m in binding.measurements:
                if isinstance(m, ExpectationMeasure):
                    if any([o.is_matrix() for o in m.observables]):
                        # This is because of braket's programSet limitations
                        warn(
                            "To translate observables from CircuitBindings to braket we need to translate the matrix to pauli string. This process might impact performances on big matrices."
                        )
                    for o in m.observables:
                        from braket.circuits.observables import Sum

                        translation = o.pauli_string.to_other_language(Language.BRAKET)
                        if not isinstance(translation, Sum):
                            translation = [translation]
                        if len(m.observables) != 1:
                            obs.append((ExpectationMeasure(o), translation))
                        else:
                            obs.append((m, translation))
                else:
                    translation = braket_Circuit()
                    translation.measure(m.targets)
                    if depth == 2:
                        return translated[0] + translation  # type: ignore
                    obs.append((m, translation))
        if depth != 0:  # If the circuitBinding is embedded store the translated data.
            binding._translated_circuits = (  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
                translated
            )
            binding._translated_observables = (  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
                obs
            )
            binding._translated_variables = (  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
                var
            )
            return binding

        result = []
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
        from mpqp.core.circuit import BindingMode

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

        from braket.circuits.observables import Sum

        for values, observable in executables:
            if binding.mode == BindingMode.PRODUCT:
                for i, t in enumerate(translated):
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
                        if TYPE_CHECKING:
                            assert isinstance(
                                t._translated_circuits,  # pyright: ignore[reportPrivateUsage]
                                list,
                            )
                            assert all(
                                [
                                    isinstance(c, braket_Circuit)
                                    for c in t._translated_circuits  # pyright: ignore[reportGeneralTypeIssues,reportPrivateUsage]
                                ]
                            )

                        for inside_observable, inside_val in inside_executables:
                            if inside_observable:
                                if TYPE_CHECKING:
                                    assert isinstance(inside_observable, tuple)
                                from mpqp.execution.job import JobType

                                mpqp_obs, braket_obs = (
                                    inside_observable  # pyright: ignore[reportGeneralTypeIssues]
                                )
                                for index, c in enumerate(t._translated_circuits):  # type: ignore

                                    if binding.job_type == JobType.SAMPLE:
                                        result.append(
                                            BraketBinding(
                                                c + braket_obs,
                                                input_sets=inside_val,
                                            )
                                        )
                                    else:
                                        result.append(
                                            BraketBinding(
                                                c,
                                                input_sets=inside_val,
                                                observables=braket_obs,
                                            )
                                        )
                                    if isinstance(t.circuits[index], CircuitBinding):
                                        # t means depth 1 if CB in depth 1 means special nested in depth 2
                                        context.append(
                                            (t.circuits[index].circuits[0], mpqp_obs, inside_val)  # type: ignore
                                        )
                                    else:
                                        context.append(
                                            (t.circuits[index], mpqp_obs, inside_val)
                                        )
                            else:
                                for index, c in enumerate(t._translated_circuits):  # type: ignore
                                    result.append(
                                        BraketBinding(
                                            c,
                                            input_sets=inside_val,
                                        )
                                        if inside_val
                                        else c
                                    )
                                    if isinstance(t.circuits[index], CircuitBinding):
                                        # t means depth 1 if CB in depth 1 means special nested in depth 2
                                        context.append(
                                            (t.circuits[index].circuits[0], t.circuits[index].measurements[0], inside_val)  # type: ignore
                                        )
                                    else:
                                        context.append((t.circuits[index], inside_val))
                    else:
                        if observable:
                            if TYPE_CHECKING:
                                assert isinstance(observable, tuple)
                            mpqp_obs, braket_obs = (
                                observable  # pyright: ignore[reportGeneralTypeIssues]
                            )
                            result.append(
                                BraketBinding(
                                    t,
                                    input_sets=values,
                                    observables=braket_obs,
                                )
                            )
                            context.extend([(binding.circuits[i], mpqp_obs, values)])
                        else:
                            result.append(
                                BraketBinding(
                                    t,
                                    input_sets=values,
                                )
                                if values
                                else t
                            )
                            context.append((binding.circuits[i], values))
            else:
                if isinstance(translated[i], CircuitBinding):
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

                    cb: CircuitBinding = translated[
                        i
                    ]  # pyright: ignore[reportAssignmentType]
                    for index, c in enumerate(translated[i]._translated_circuits):  # type: ignore
                        if TYPE_CHECKING:
                            assert isinstance(c, braket_Circuit)
                        for inside_observable, inside_val in inside_executables:
                            if inside_observable:
                                if TYPE_CHECKING:
                                    assert isinstance(inside_observable, tuple)
                                mpqp_obs, braket_obs = (
                                    inside_observable  # pyright: ignore[reportGeneralTypeIssues]
                                )
                                result.append(
                                    BraketBinding(
                                        c,
                                        input_sets=inside_val,
                                        observables=braket_obs,
                                    )
                                )
                                context.append(
                                    (
                                        cb.circuits[index],
                                        mpqp_obs,
                                        inside_val,
                                    )
                                )
                            else:
                                result.append(
                                    (
                                        BraketBinding(
                                            c,
                                            input_sets=inside_val,
                                        )
                                        if inside_val
                                        else c
                                    )
                                )
                                context.append((cb.circuits[index], inside_val))
                else:
                    if observable:
                        if TYPE_CHECKING:
                            assert isinstance(observable, tuple)
                        mpqp_obs, braket_obs = (
                            observable  # pyright: ignore[reportGeneralTypeIssues]
                        )
                        result.append(
                            BraketBinding(
                                translated[i],  # pyright: ignore
                                input_sets=values,
                                observables=braket_obs,
                            )
                        )
                        c = binding.circuits[i]
                        context.append((c, mpqp_obs, values))
                    else:
                        result.append(
                            BraketBinding(
                                translated[i], input_sets=values  # pyright: ignore
                            )
                            if values
                            else translated[i]
                        )
                        c = binding.circuits[i]
                        context.append((c, values))
                i += 1 if i != -1 else 0

        from braket.program_sets import ProgramSet

        ps = ProgramSet(result, binding.shots)
        return ps, context

    def circuitbinding_to_programset(
        binding: "CircuitBinding", device: "AvailableDevice"
    ) -> "tuple[ProgramSet, list[tuple[Any]]]":

        # Will be used when pauli grouping is implemented
        """
        from mpqp.core.instruction import ExpectationMeasure
        if binding.measurements:
            if (
                isinstance(binding.measurements[0], ExpectationMeasure)
                and binding.measurements[0].optimize_measurement
            ):
                return _cb_to_programset_pauli_grouping(binding, device, True)"""
        return _cb_to_programset(binding, device, 0)
