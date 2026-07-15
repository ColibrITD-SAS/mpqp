from copy import deepcopy
from typing import TYPE_CHECKING

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.native_gates import NativeGate
from mpqp.core.languages import Language
from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
    InstalledProviders,
)
from mpqp.translation.utils import verify_convert_instructions

if InstalledProviders.CIRQ in _INSTALLED_MPQP_PROVIDERS:

    from cirq.circuits.circuit import Circuit as cirq_Circuit
    from cirq.circuits.moment import Moment

    def cirq_to_mpqp(qcircuit: cirq_Circuit | Moment) -> QCircuit:
        """Translate a cirq Circuit to a MPQP QCircuit.
        Note:
            If the provided qcircuit is a Moment it will be translated to a fully fledged cirq circuit.
            Also since we use cirq's to_qasm method some edge cases are limited by the original library.
        Args:
            qcircuit: Any cirq Circuit, or for simpler circuits could be a sole Moment.
        """
        from cirq.circuits.circuit import Circuit as cirq_Circuit
        from cirq.circuits.moment import Moment
        from cirq import ops, MatrixGate
        from mpqp import QCircuit
        from mpqp.gates import CustomGate

        qcircuit = deepcopy(qcircuit)
        from mpqp.translation.qasm.qasm_to_mpqp import (
            parse_qasm2_gates,
            qasm2_parse,
        )

        if isinstance(qcircuit, Moment):
            qcircuit = cirq_Circuit([qcircuit])

        c = QCircuit()
        qubits = ops.QubitOrder.as_qubit_order(ops.QubitOrder.DEFAULT).order_for(
            qcircuit.all_qubits()
        )
        split_index = 0
        for i, moment in enumerate(qcircuit):
            for operation in moment.operations:
                # Cut the moments of the circuit if a matrixGate is encountered
                if isinstance(operation.gate, MatrixGate):
                    matrix = (
                        operation.gate._matrix  # pyright: ignore[reportPrivateUsage]
                    )
                    targets = [qubits.index(qubit) for qubit in operation.qubits]
                    qcircuit.batch_remove([(i, operation)])
                    cir, phase = parse_qasm2_gates(
                        qcircuit.from_moments(qcircuit[split_index:i]).to_qasm()
                    )
                    c += qasm2_parse(cir)
                    c.input_g_phase += phase
                    c.add(CustomGate(matrix, targets))

                    split_index = i
                else:
                    moment_ = moment.expand_to(qubits)

                    qcircuit._moments.pop(i)  # type: ignore[reportPrivateUsage]

                    qcircuit._moments.insert(  # pyright: ignore[reportPrivateUsage]
                        i, moment_
                    )

        if split_index != 0:
            cir, phase = parse_qasm2_gates(
                qcircuit.from_moments(qcircuit[split_index:]).to_qasm()
            )
            c += qasm2_parse(cir)
            c.input_g_phase += phase
        else:
            qasm2_code, gphase = parse_qasm2_gates(qcircuit.to_qasm())
            c = qasm2_parse(qasm2_code)
            c.input_g_phase = gphase
        return c

    def mpqp_to_cirq(
        circuit: QCircuit,
        skip_pre_measure: bool = False,
        skip_measurements: bool = False,
        authorized_gates: set[type[NativeGate]] | None = None,
    ) -> cirq_Circuit:
        """Translate a MPQP circuit to a Cirq equivalent.

        Note:
            If the circuit contains ComposedGate type instructions you need to include the gate in the authorize_gates set for it to avoid decomposition, otherwise, the translation will always
            try to use the gate's decomposition (see example).
        Args:
            circuit: The original MPQP circuit to be translated.
            skip_pre_measure: If set at True will translate the circuit without its pre-measurement circuit (see QCircuit.to_other_language for more information).
            skip_measurements: If set at True will translate the circuit without any measurement.
            authorized_gates: The set of gates allowed on the circuit, if the circuit contains any other gates it raises a ValueError.

        Examples:
            >>> circuit = QCircuit([H(0), CNOT(0,1), BasisMeasure()])
            >>> cirq_with = mpqp_to_cirq(circuit)
            >>> cirq_without = mpqp_to_cirq(circuit, skip_measurements=True)
            >>> print(cirq_with) # doctest: +NORMALIZE_WHITESPACE
            q_0: ───I───H───@───M('')───
                            │   │
            q_1: ───I───────X───M───────
            >>> print(cirq_without) # doctest: +NORMALIZE_WHITESPACE
            q_0: ───I───H───@───
                            │
            q_1: ───I───────X───
        """
        # TODO: add better doc for all providers (ComposedGate) and CustomGate behavior for this one specifically
        from cirq.circuits.circuit import Circuit as CirqCircuit
        from cirq.ops.identity import I
        from cirq.ops.named_qubit import NamedQubit
        from mpqp.core.instruction import (
            Measure,
            Breakpoint,
            CustomGate,
            Barrier,
            ControlledGate,
            CustomControlledGate,
            ExpectationMeasure,
        )

        if authorized_gates is None:
            authorized_gates = set()
        cirq_qubits = [NamedQubit(f"q_{i}") for i in range(circuit.nb_qubits)]
        cirq_circuit = CirqCircuit()

        for qubit in cirq_qubits:
            cirq_circuit.append(I(qubit))

        for instruction in circuit.instructions:
            if not skip_pre_measure:
                if isinstance(instruction, Measure):
                    for pre_measure in instruction.pre_measure:
                        if isinstance(pre_measure, (CustomGate, CustomControlledGate)):
                            instr = verify_convert_instructions(
                                pre_measure, authorized_gates
                            )
                            qasm2_code, gphase = pre_measure.to_other_language(
                                Language.QASM2
                            )  # pyright: ignore[reportGeneralTypeIssues]
                            if TYPE_CHECKING:
                                assert isinstance(qasm2_code, str)
                            from mpqp.translation.qasm.qasm_to_cirq import (
                                qasm2_to_cirq_Circuit,
                            )

                            qasm2_code = (
                                "OPENQASM 2.0;"
                                + "\ninclude \"qelib1.inc\";"
                                + f"\nqreg q[{circuit.nb_qubits}];\n"
                                + qasm2_code
                            )
                            custom_cirq_circuit = qasm2_to_cirq_Circuit(qasm2_code)
                            cirq_circuit += custom_cirq_circuit
                            # TODO: handle gphase in the circuit
                            circuit._generated_g_phase += gphase  # type: ignore[reporPrivateUsage]
                        else:
                            cirq_pre_measure = pre_measure.to_other_language(
                                Language.CIRQ
                            )
                            targets = []
                            for target in pre_measure.targets:
                                targets.append(cirq_qubits[target])
                            cirq_circuit.append(cirq_pre_measure.on(*targets))

            if isinstance(instruction, Gate):
                instr = verify_convert_instructions(instruction, authorized_gates)
            else:
                instr = [instruction]

            for gate in instr:
                if isinstance(gate, (ExpectationMeasure, Barrier, Breakpoint)):
                    continue
                elif isinstance(gate, CustomGate):
                    from cirq.ops.raw_types import Gate as CirqGate

                    custom_gate = instr[0]

                    targets = []
                    for target in custom_gate.targets:
                        targets.append(cirq_qubits[target])

                    cirq_instruction = custom_gate.to_other_language(Language.CIRQ)
                    assert isinstance(cirq_instruction, CirqGate)

                    cirq_circuit.append(cirq_instruction.on(*targets))
                    continue
                elif isinstance(gate, ControlledGate):
                    targets = []
                    for target in gate.targets:
                        targets.append(cirq_qubits[target])
                    controls = []
                    for control in gate.controls:
                        controls.append(cirq_qubits[control])
                    cirq_instruction = gate.to_other_language(Language.CIRQ)
                    cirq_circuit.append(cirq_instruction.on(*controls, *targets))
                else:
                    if skip_measurements and isinstance(gate, Measure):
                        continue
                    targets = []
                    for target in gate.targets:
                        targets.append(cirq_qubits[target])
                    cirq_instruction = gate.to_other_language(Language.CIRQ)
                    if TYPE_CHECKING:
                        assert cirq_instruction
                    cirq_circuit.append(cirq_instruction.on(*targets))

        if circuit.noises:
            from mpqp.execution.providers.google import apply_noise_to_cirq_circuit

            return apply_noise_to_cirq_circuit(
                cirq_circuit,
                circuit.noises,
            )

        if circuit.input_g_phase != 0:
            from cirq import GlobalPhaseGate
            import numpy as np

            cirq_circuit.insert(
                0, GlobalPhaseGate(np.exp(1j * circuit.input_g_phase)).on()
            )

        return cirq_circuit
