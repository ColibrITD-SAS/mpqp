from typing import TYPE_CHECKING

from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
)
from mpqp.environment.var_cache import (
    InstalledProviders,
)

if InstalledProviders.QISKIT in _INSTALLED_MPQP_PROVIDERS:
    from mpqp.core.instruction.gates.gate import Gate
    from mpqp.core.instruction.gates.gate_decomposition import resolve_gate
    from mpqp.core.instruction.gates.native_gates import U
    from mpqp.execution._execution_context import ExecutionContext

    if TYPE_CHECKING:
        from qiskit import QuantumCircuit
        from qiskit.primitives.containers import EstimatorPubLike

        from mpqp.core.binding import CircuitBinding
        from mpqp.core.circuit import QCircuit
        from mpqp.execution.devices import IBMDevice

    def qiskit_to_mpqp(qcircuit: "QuantumCircuit"):
        """Translate a qiskit QuantumCircuit into a MPQP QCircuit.
        Note:
            This function make use of the qasm3 package from qiskit which means that some gates in the circuit
            could be replaced by equivalents.

        Args:
            qcircuit: Any Qiskit quantum circuit.
        """
        from qiskit import qasm3

        from mpqp.core.languages import Language
        from mpqp.translation.qasm import open_qasm_3_to_2
        from mpqp.translation.qasm.qasm_to_mpqp import qasm2_parse

        qasm3_code = qasm3.dumps(qcircuit)
        qasm2_code = open_qasm_3_to_2(str(qasm3_code), language=Language.QISKIT)

        qc = qasm2_parse(qasm2_code)
        return qc

    def get_qiskit_gate_set() -> set[type[Gate]]:
        from mpqp.gates import CNOT, PRX, Rx, Rxx, Ry, Ryy, Rz, Rzz

        return {Rx, Ry, Rz, PRX, Rxx, Ryy, Rzz, U, CNOT}

    def mpqp_to_qiskit(
        circuit: "QCircuit",
        skip_pre_measure: bool = False,
        skip_measurements: bool = False,
        printing: bool = False,
    ) -> "QuantumCircuit":
        """Translate a MPQP circuit to a Qiskit equivalent.

        Note:
            If the circuit contains ComposedGate type instructions you need to include the gate in the authorize_gates set for it to avoid decomposition, otherwise, the translation will always
            try to use the gate's decomposition (see example).
        Args:
            circuit: The original MPQP circuit to be translated.
            skip_pre_measure: If set at True will translate the circuit without its pre-measurement circuit (see QCircuit.to_other_language for more information).
            skip_measurements: If set at True will translate the circuit without any measurement.

        Examples:
            >>> circuit = QCircuit([H(0), CNOT(0, 1), BasisMeasure()])
            >>> cirq_with = mpqp_to_qiskit(circuit)
            >>> cirq_without = mpqp_to_qiskit(circuit, skip_measurements=True)
            >>> print(cirq_with) # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐     ┌─┐
            q_0: ┤ H ├──■──┤M├───
                 └───┘┌─┴─┐└╥┘┌─┐
            q_1: ─────┤ X ├─╫─┤M├
                      └───┘ ║ └╥┘
            c: 2/═══════════╩══╩═
                        0  1
            >>> print(cirq_without) # doctest: +NORMALIZE_WHITESPACE
                 ┌───┐
            q_0: ┤ H ├──■──
                 └───┘┌─┴─┐
            q_1: ─────┤ X ├
                      └───┘
            c: 2/══════════
        """
        from qiskit.circuit import Operation, QuantumCircuit
        from qiskit.circuit.quantumcircuit import CircuitInstruction
        from qiskit.quantum_info import Operator

        from mpqp.core.instruction import (
            Barrier,
            BasisMeasure,
            Breakpoint,
            ControlledGate,
            CustomGate,
            ExpectationMeasure,
            Measure,
        )
        from mpqp.core.instruction.gates.custom_controlled_gate import (
            CustomControlledGate,
        )
        from mpqp.core.instruction.gates.gate import Gate
        from mpqp.core.languages import Language

        # to avoid defining twice the same parameter, we keep trace of the
        # added parameters, and we use those instead of new ones when they
        # are used more than once

        qiskit_parameters = set()
        if circuit.nb_cbits == 0:
            new_circ = QuantumCircuit(circuit.nb_qubits)
        else:
            new_circ = QuantumCircuit(circuit.nb_qubits, circuit.nb_cbits)

        if circuit.label is not None:
            new_circ.name = circuit.label

        for instruction in circuit.instructions:
            if isinstance(instruction, (Measure, Breakpoint)):
                continue
            options = (
                {"printing": printing}
                if isinstance(instruction, (CustomGate, CustomControlledGate))
                else {}
            )

            if (
                printing
                and isinstance(instruction, CustomControlledGate)
                and isinstance(instruction.non_controlled_gate, CustomGate)
            ):
                instr = [instruction]

            elif isinstance(instruction, Gate):
                qiskit_gate_set = get_qiskit_gate_set()
                instr = list(
                    resolve_gate(
                        instruction,
                        qiskit_gate_set,
                    )
                )
            else:
                instr = [instruction]

            for instruction in instr:
                qiskit_inst = instruction.to_other_language(
                    Language.QISKIT, qiskit_parameters, **options
                )
                if TYPE_CHECKING:
                    assert isinstance(
                        qiskit_inst, (CircuitInstruction, Operation, Operator)
                    )
                cargs = []

                if isinstance(instruction, CustomGate) and not isinstance(
                    instruction, CustomControlledGate
                ):
                    if TYPE_CHECKING:
                        assert isinstance(qiskit_inst, Operator)
                    if printing and len(instruction.free_symbols) > 0:
                        new_circ.append(
                            qiskit_inst, list(reversed(instruction.targets))
                        )
                    else:
                        new_circ.append(
                            qiskit_inst,
                            list(reversed(instruction.targets)),
                        )
                else:
                    qargs = []
                    if isinstance(instruction, ControlledGate):
                        qargs = list(reversed(instruction.controls)) + list(
                            reversed(instruction.targets)
                        )
                    elif isinstance(instruction, Gate):
                        qargs = list(reversed(instruction.targets))
                    elif isinstance(instruction, Barrier):
                        qargs = range(circuit.nb_qubits)
                    else:
                        raise ValueError(f"Instruction not handled: {instruction}")

                    if TYPE_CHECKING:
                        assert not isinstance(qiskit_inst, Operator)
                    new_circ.append(
                        qiskit_inst,
                        list(qargs),
                        cargs,
                    )
        for measurement in circuit.measurements:
            if not skip_pre_measure:

                for pre_measure in measurement.pre_measure:
                    cargs = []
                    qiskit_pre_measure = pre_measure.to_other_language(
                        Language.QISKIT, qiskit_parameters
                    )
                    new_circ.append(
                        qiskit_pre_measure,
                        list(reversed(pre_measure.targets)),
                        cargs=cargs,
                    )
            if not skip_measurements:
                if isinstance(measurement, ExpectationMeasure):
                    continue
                qiskit_inst = measurement.to_other_language(
                    Language.QISKIT, qiskit_parameters
                )
                if isinstance(measurement, BasisMeasure):
                    if TYPE_CHECKING:
                        assert measurement.c_targets is not None
                else:
                    raise ValueError(f"measurement not handled: {measurement}")

                if TYPE_CHECKING:
                    assert not isinstance(qiskit_inst, Operator)
                new_circ.append(
                    qiskit_inst,
                    [measurement.targets],
                    [measurement.c_targets],
                )

        new_circ.global_phase += (
            circuit.input_g_phase
            + circuit._generated_g_phase  # type: ignore[reportPrivateUsage]
        )
        return new_circ

    def circuitbinding_to_pubs(
        binding: "CircuitBinding", device: "IBMDevice"
    ) -> "list[tuple[EstimatorPubLike, list[ExecutionContext]]]":
        from copy import deepcopy

        from mpqp.core.instruction import ExpectationMeasure
        from mpqp.core.languages import Language
        from mpqp.execution import Job, JobType

        if binding.job_type != JobType.OBSERVABLE:
            raise ValueError(
                "to_other_device is only supported for circuits with expectation measurements."
            )

        unrolled = binding.unroll()

        from mpqp.tools.binding_utils import translate_binding_circuits

        translated_circuits = translate_binding_circuits(binding, device)

        grouped = {}
        for index, (c, values, measure) in enumerate(unrolled):
            grouped.setdefault(id(c), []).append((index, c, values, measure))

        pubs_with_context = []
        for executions in grouped.values():
            _, circuit, _, _ = executions[0]
            q_c = translated_circuits.get(id(circuit))
            if q_c is None:
                raise ValueError(
                    f"Circuit with id {id(circuit)} wasn't found in the binding, please contact the maintainers of this library."
                )
            if TYPE_CHECKING:
                from qiskit.circuit import QuantumCircuit

                assert isinstance(q_c, QuantumCircuit)
            parameter_names = [p.name for p in q_c.parameters]
            params = []
            q_obs = []
            contexts = []
            for index, c, values, measure in executions:
                normalized_values = {str(k): v for k, v in (values or {}).items()}
                missing = [
                    name for name in parameter_names if name not in normalized_values
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
                            obs.pre_transpiled[device]
                            if device in obs.pre_transpiled
                            else obs.to_other_language(Language.QISKIT)
                        )
                        for obs in measure.observables
                    ]
                )
                context = c.without_measurements(deep_copy=False)
                context.add(deepcopy(measure))
                contexts.append(
                    ExecutionContext(
                        job=Job(binding.job_type, context, device, values=values),
                        index=index,
                    )
                )

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
