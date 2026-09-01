from typing import TYPE_CHECKING

from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
    InstalledProviders,
)

if InstalledProviders.BRAKET in _INSTALLED_MPQP_PROVIDERS:
    if TYPE_CHECKING:
        from braket.circuits import Circuit as braket_Circuit
        from mpqp.core.instruction.gates.native_gates import NativeGate
        from mpqp.core.circuit import QCircuit

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
        instructions = circuit._instructions  # pyright: ignore[reportPrivateUsage]
        if len(circuit.noises) != 0:
            if any(isinstance(instr, CRk) for instr in instructions):
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
                *(inst.connections() for inst in instructions if isinstance(inst, Gate))
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
                instructions = (
                    circuit._instructions
                )  # pyright: ignore[reportPrivateUsage]

        for instruction in instructions + circuit.measurements:
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
