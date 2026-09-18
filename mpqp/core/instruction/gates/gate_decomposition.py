from dataclasses import dataclass

from mpqp.core.instruction.gates.custom_controlled_gate import CustomControlledGate
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.native_gates import ComposedGate
from mpqp.core.instruction.instruction import Instruction
from mpqp.tools.errors import UnsupportedGateError


@dataclass(frozen=True)
class GateResolution:
    """
    Result of resolving a gate against a target gate set.

    Args:
        source: Original gate passed to the resolution process.
        gates: Gates resulting from the resolution. This contains either the
            original gate when it is directly supported, or its resolved
            decomposition.
        decomposed: Whether the source gate was decomposed during the resolution.
    """

    source: Gate
    gates: tuple[Gate, ...]
    decomposed: bool


def resolve_instructions(
    instructions: list[Instruction],
    gate_set: set[type[Gate]],
) -> list[Instruction]:
    """
    Resolve the composed gates contained in a sequence of instructions.

    Gates are processed using the funtion `resolve_gate`. Instructions that are not
    gates are preserved unchanged and in their original order.

    Args:
        instructions: Instructions to resolve.
        gate_set: Gate types directly supported by the target language or
            provider.

    Returns:
        A list containing the resolved gates and the unchanged non-gate
        instructions.

    Example:
        >>> from mpqp.gates import CNOT, Rz, Rzz
        >>> resolved = resolve_instructions([Rzz(1.0, 0, 1)], {CNOT, Rz})
        >>> [type(gate).__name__ for gate in resolved]
        ['CNOT', 'Rz', 'CNOT']
    """

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
    """
    Resolve a gate recursively against a target gate set.

    If the gate type is directly supported, the gate is returned unchanged.
    Otherwise, the gate is recursively decomposed until all resulting gates
    belong to ``gate_set``.

    Args:
        gate: Gate to resolve.
        gate_set: Gate types directly supported by the target language or
            provider.
        resolving: Gate types currently being resolved. This internal state is
            used to detect cyclic decompositions.

    Returns:
        A :class:`GateResolution` containing the original gate, the resolved
        gates and whether a decomposition occurred.

    Raises:
        UnsupportedGateError: If the gate, or one of the gates produced by its
            decomposition, cannot be represented using ``gate_set``.
        ValueError: If the decomposition is empty, directly contains its source
            gate, or contains a cycle.

    Example:
        >>> from mpqp.gates import PRX, Rx, Rz
        >>> result = resolve_composed_gate(PRX(1.0, 0.5, 0), {Rx, Rz})
        >>> [type(gate).__name__ for gate in result.gates]
        ['Rz', 'Rx', 'Rz']
    """
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
        if type(child) in gate_set:
            resolved.append(child)
        elif isinstance(child, ComposedGate):
            result = resolve_composed_gate(
                child,
                gate_set,
                resolving | {gate_type},
            )
            resolved.extend(result.gates)
        else:
            missing.add(type(child))

    if missing:
        raise UnsupportedGateError(gate, gate_set, missing)

    return GateResolution(gate, tuple(resolved), decomposed=True)


def resolve_gate(
    gate: Gate,
    gate_set: set[type[Gate]],
) -> tuple[Gate, ...]:
    """Resolve a gate into gates supported by a target gate set.

    A custom controlled gate wrapping a custom gate is first converted into an
    equivalent custom gate. Non-composed gates are returned unchanged. Composed
    gates are resolved recursively using :func:`resolve_composed_gate`.

    Args:
        gate: Gate to resolve.
        gate_set: Gate types directly supported by the target language or
            provider.

    Returns:
        A tuple containing the original gate, its converted custom gate, or the
        gates resulting from its recursive decomposition.

    Example:
        >>> from mpqp.gates import CNOT, Rz, Rzz
        >>> resolved = resolve_gate(Rzz(1.0, 0, 1), {CNOT, Rz})
        >>> [type(gate).__name__ for gate in resolved]
        ['CNOT', 'Rz', 'CNOT']
    """
    if isinstance(gate, CustomControlledGate) and isinstance(
        gate.non_controlled_gate,
        CustomGate,
    ):
        return (gate.to_custom_gate(),)

    if not isinstance(gate, ComposedGate):
        return (gate,)

    return resolve_composed_gate(gate, gate_set).gates
