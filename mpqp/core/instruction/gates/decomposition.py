from dataclasses import dataclass

from mpqp.core.instruction.gates.custom_controlled_gate import CustomControlledGate
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.native_gates import ComposedGate
from mpqp.core.instruction.instruction import Instruction
from mpqp.tools.errors import UnsupportedGateError


@dataclass(frozen=True)
class GateResolution:
    source: Gate
    gates: tuple[Gate, ...]
    decomposed: bool


def resolve_instructions(
    instructions: list[Instruction],
    gate_set: set[type[Gate]],
) -> list[Instruction]:
    resolved: list[Instruction] = []

    for instruction in instructions:
        if isinstance(instruction, Gate):
            resolved.extend(resolve_gate(instruction, gate_set))
        else:
            resolved.append(instruction)

    return resolved


def resolve_composed_gate(
    gate: Gate,
    gate_set: set[type[Gate]],
    resolving: frozenset[type[Gate]] = frozenset(),
) -> GateResolution:
    gate_type = type(gate)

    if gate_type in gate_set:
        return GateResolution(gate, (gate,), decomposed=False)

    if not isinstance(gate, ComposedGate):
        raise UnsupportedGateError(gate, gate_set)

    if gate_type in resolving:
        raise ValueError(f"Cyclic decomposition detected for {gate_type.__name__}.")

    decomposition = gate.decompose()

    if not decomposition or any(child is gate for child in decomposition):
        raise ValueError(f"{gate_type.__name__} returned an invalid decomposition.")

    resolved: list[Gate] = []
    missing: set[type[Gate]] = set()

    for child in decomposition:
        try:
            result = resolve_composed_gate(
                child,
                gate_set,
                resolving | {gate_type},
            )
            resolved.extend(result.gates)
        except UnsupportedGateError:
            missing.add(type(child))

    if missing:
        raise UnsupportedGateError(gate, gate_set, missing)

    return GateResolution(gate, tuple(resolved), decomposed=True)


def resolve_gate(
    gate: Gate,
    gate_set: set[type[Gate]],
) -> tuple[Gate, ...]:
    if isinstance(gate, CustomControlledGate) and isinstance(
        gate.non_controlled_gate,
        CustomGate,
    ):
        return (gate.to_custom_gate(),)

    if not isinstance(gate, ComposedGate):
        return (gate,)

    return resolve_composed_gate(gate, gate_set).gates
