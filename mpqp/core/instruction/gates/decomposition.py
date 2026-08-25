from dataclasses import dataclass

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.native_gates import ComposedGate


@dataclass(frozen=True)
class GateResolution:
    source: Gate
    gates: tuple[Gate, ...]
    decomposed: bool


class UnsupportedGateError(ValueError):
    def __init__(
        self,
        gate: Gate,
        gate_set: set[type[Gate]],
        missing_gates: set[type[Gate]] | None = None,
    ):
        missing = missing_gates or {type(gate)}
        missing_names = ", ".join(sorted(g.__name__ for g in missing))
        available_names = ", ".join(sorted(g.__name__ for g in gate_set))

        super().__init__(
            f"{type(gate).__name__} cannot be represented with the target "
            f"gate set. Missing gates: {missing_names}. "
            f"Available gates: {available_names}."
        )


def resolve_gate(
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
            result = resolve_gate(
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
