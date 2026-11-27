from typing import TYPE_CHECKING, cast

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.gate_decomposition import resolve_gate
from mpqp.environment.var_cache import (
    _INSTALLED_MPQP_PROVIDERS,  # pyright: ignore[reportPrivateUsage]
)
from mpqp.environment.var_cache import (
    InstalledProviders,
)

if TYPE_CHECKING:
    from typing import Protocol

    from sympy import Expr

    # Cirq's value_equality decorator erases the decorated classes' static types.
    class _CirqEigenGate(Protocol):
        @property
        def exponent(self) -> float | Expr: ...

        @property
        def global_shift(self) -> float: ...

    class _CirqGlobalPhaseGate(Protocol):
        @property
        def coefficient(self) -> complex | Expr: ...


if InstalledProviders.CIRQ in _INSTALLED_MPQP_PROVIDERS:

    from cirq.circuits.circuit import Circuit as cirq_Circuit
    from cirq.circuits.moment import Moment

    from mpqp.core.circuit import QCircuit

    def cirq_to_mpqp(qcircuit: cirq_Circuit | Moment) -> QCircuit:
        """Translate a Cirq circuit or moment to an MPQP circuit.

        Args:
            qcircuit: Circuit or moment to translate.

        Note:
            XX, YY, and ZZ rotations retain their native representation and global
            phase. Other gates use Cirq's QASM export and share its limitations.
        """
        import numpy as np

        from cirq import (
            EigenGate,
            GlobalPhaseGate,
            MatrixGate,
            Operation,
            QasmOutput,
            ops,
            unitary,
        )
        from cirq.circuits.circuit import Circuit as cirq_Circuit
        from cirq.circuits.moment import Moment

        from mpqp import QCircuit
        from mpqp.gates import CustomGate, Rxx, Ryy, Rzz

        from mpqp.translation.qasm.qasm_to_mpqp import (
            parse_qasm2_gates,
            qasm2_parse,
        )

        if isinstance(qcircuit, Moment):
            qcircuit = cirq_Circuit([qcircuit])

        qubits = ops.QubitOrder.as_qubit_order(ops.QubitOrder.DEFAULT).order_for(
            qcircuit.all_qubits()
        )
        c = QCircuit(nb_qubits=len(qubits))
        pending_operations: list[Operation] = []
        rotations = {rotation.cirq_gate: rotation for rotation in (Rxx, Ryy, Rzz)}

        def append_pending_operations() -> None:
            if not pending_operations:
                return
            # Keep one qubit order across QASM segments and native operations.
            code, phase = parse_qasm2_gates(
                str(QasmOutput(pending_operations, qubits=qubits))
            )
            c.add(qasm2_parse(code).instructions)
            c.input_g_phase += phase
            pending_operations.clear()

        for operation in qcircuit.all_operations():
            gate = operation.gate
            if isinstance(gate, GlobalPhaseGate):
                phase_gate = cast("_CirqGlobalPhaseGate", gate)
                c.input_g_phase += float(np.angle(complex(phase_gate.coefficient)))
            elif isinstance(gate, EigenGate) and isinstance(gate, tuple(rotations)):
                append_pending_operations()
                targets = [qubits.index(qubit) for qubit in operation.qubits]
                rotation = next(
                    rotation
                    for cirq_gate, rotation in rotations.items()
                    if isinstance(gate, cirq_gate)
                )
                eigen_gate = cast("_CirqEigenGate", gate)
                angle = (
                    eigen_gate.exponent * np.pi  # pyright: ignore[reportOperatorIssue]
                )
                c.add(rotation(angle, targets[0], targets[1]))
                if eigen_gate.global_shift != -0.5:
                    c.input_g_phase += float(angle * (eigen_gate.global_shift + 0.5))
            elif isinstance(gate, MatrixGate):
                append_pending_operations()
                targets = [qubits.index(qubit) for qubit in operation.qubits]
                c.add(CustomGate(unitary(gate), targets))
            else:
                pending_operations.append(operation)

        append_pending_operations()
        return c

    def get_cirq_gate_set() -> set[type[Gate]]:
        """Return gates directly representable by Cirq."""
        from mpqp.gates import CNOT, Rx, Ry, Rz, PRX, Rzz, Rxx, Ryy

        return {Rx, Ry, Rz, CNOT, PRX, Rzz, Rxx, Ryy}

    def mpqp_to_cirq(
        circuit: QCircuit,
        skip_pre_measure: bool = False,
        skip_measurements: bool = False,
    ) -> cirq_Circuit:
        """Translate a MPQP circuit to a Cirq equivalent.

        Note:
            If the circuit contains ComposedGate type instructions you need to include the gate in the authorize_gates set for it to avoid decomposition, otherwise, the translation will always
            try to use the gate's decomposition (see example).
        Args:
            circuit: The original MPQP circuit to be translated.
            skip_pre_measure: If set at True will translate the circuit without its pre-measurement circuit (see QCircuit.to_other_language for more information).
            skip_measurements: If set at True will translate the circuit without any measurement.

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
            Barrier,
            Breakpoint,
            ControlledGate,
            CustomControlledGate,
            CustomGate,
            ExpectationMeasure,
            Measure,
        )
        from mpqp.core.instruction.gates.gate import Gate

        from mpqp.core.languages import Language

        cirq_qubits = [NamedQubit(f"q_{i}") for i in range(circuit.nb_qubits)]
        cirq_circuit = CirqCircuit()

        for qubit in cirq_qubits:
            cirq_circuit.append(I(qubit))

        instructions = circuit.with_measurement(deep_copy=False)
        for instruction in instructions:
            if not skip_pre_measure:
                if isinstance(instruction, Measure):
                    for pre_measure in instruction.pre_measure:
                        if isinstance(pre_measure, (CustomGate, CustomControlledGate)):
                            resolved_pre_measure = resolve_gate(
                                pre_measure,
                                get_cirq_gate_set(),
                            )
                            qasm2_code, gphase = resolved_pre_measure[
                                0
                            ].to_other_language(Language.QASM2)
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
                instructions = resolve_gate(
                    instruction,
                    get_cirq_gate_set(),
                )
            else:
                instructions = (instruction,)

            for gate in instructions:
                if isinstance(gate, (ExpectationMeasure, Barrier, Breakpoint)):
                    continue
                elif isinstance(gate, CustomGate):
                    from cirq.ops.raw_types import Gate as CirqGate

                    custom_gate = instructions[0]

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
            import numpy as np
            from cirq import GlobalPhaseGate

            cirq_circuit.insert(
                0, GlobalPhaseGate(np.exp(1j * circuit.input_g_phase)).on()
            )

        return cirq_circuit

    from typing import Any, Optional

    from cirq import Gate as cirqGate

    if TYPE_CHECKING:
        from cirq import Qid

        from mpqp.tools.generics import Matrix

    class cirqCustomGate(cirqGate):
        def __init__(
            self,
            matrix: "Matrix",
            decomposition: Any = None,
            label: Optional[str] = None,
        ):
            import numpy as np

            self.decomposition = decomposition
            self.matrix = matrix
            self.label = label
            self._nb_qubits = int(np.log2(len(matrix)))
            super(cirqCustomGate, self)

        def _num_qubits_(self) -> int:
            return self._nb_qubits

        def _unitary_(self) -> "Matrix":
            return self.matrix

        def _decompose_(self, qubits: list["Qid"]):
            if self.decomposition is None:
                pass
            else:
                return self.decomposition(qubits)

        def _circuit_diagram_info_(self, args: list[float]) -> str | list[str]:
            # we keep args for later implementation
            if self.label:
                return [self.label] * self._nb_qubits
            return ["CustomGate"] * self._nb_qubits

        def __str__(self) -> str:
            if self.label:
                return f"{self.label}"
            return f"MPQP custom gate"
