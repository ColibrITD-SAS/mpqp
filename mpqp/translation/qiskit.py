from typing import TYPE_CHECKING
from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
    InstalledProviders,
)

if InstalledProviders.QISKIT in _INSTALLED_MPQP_PROVIDERS:
    if TYPE_CHECKING:
        from mpqp.core.instruction.gates.native_gates import NativeGate
        from mpqp.core.circuit import QCircuit
        from qiskit import QuantumCircuit

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

    def mpqp_to_qiskit(
        circuit: "QCircuit",
        skip_pre_measure: bool = False,
        skip_measurements: bool = False,
        printing: bool = False,
        authorized_gates: set[type["NativeGate"]] | None = None,
    ) -> "QuantumCircuit":
        """Translate a MPQP circuit to a Qiskit equivalent.

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
        from mpqp.core.instruction.gates.gate import Gate
        from mpqp.core.instruction.gates.custom_controlled_gate import (
            CustomControlledGate,
        )
        from mpqp.core.languages import Language

        from mpqp.core.instruction import (
            Measure,
            Breakpoint,
            CustomGate,
            Barrier,
            ControlledGate,
            BasisMeasure,
            ExpectationMeasure,
        )

        # to avoid defining twice the same parameter, we keep trace of the
        # added parameters, and we use those instead of new ones when they
        # are used more than once
        if authorized_gates is None:
            authorized_gates = set()
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
                {"printing": printing} if isinstance(instruction, CustomGate) else {}
            )

            if isinstance(instruction, Gate):
                from mpqp.translation.utils import verify_convert_instructions

                instr = verify_convert_instructions(
                    instruction, authorized_gates, printing
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
