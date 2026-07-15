from mpqp.core.instruction.gates.custom_controlled_gate import CustomControlledGate
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.gate import Gate


def verify_convert_instructions(
    gate: Gate, authorized_gates: set[type[Gate]], printing: bool = False
) -> list[Gate]:
    """Function used to verify if the instruction is contained in the gate set.
    If the gate is a ComposedGate and not explicitly in the authorized_gates set it will check if it's decomposition is present in the gate set.
    """
    if len(authorized_gates) != 0:
        if type(gate) not in authorized_gates:
            raise ValueError(
                f"The gate {type(gate)} are not in the set of authorized gates: f{authorized_gates}"
            )
        else:
            return [gate]
    if (
        isinstance(gate, CustomControlledGate)
        and isinstance(gate.non_controlled_gate, CustomGate)
        and not printing
    ):
        # If the CustomControlledGate contains itself a custom gate it's better to return a bigger custom gate for compatibilities issues.
        # If the non_controlled_gate is a NativeGate it shouldn't pose a problem.
        # TODO: check how to decompose a ComposeGate that is inside a CCG, (example: a C-PRX)
        return [gate.to_custom_gate()]

    return [gate]
