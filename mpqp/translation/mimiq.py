"""Native circuit translation for MIMIQ's Python SDK."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from mpqp.core.instruction import Barrier, BasisMeasure, Breakpoint, Measure
from mpqp.core.instruction.gates import ControlledGate, Gate
from mpqp.core.instruction.gates.gate_decomposition import resolve_gate
from mpqp.gates import (
    PRX,
    SWAP,
    H,
    Id,
    P,
    Rk,
    Rk_dagger,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    S,
    S_dagger,
    T,
    U,
    X,
    Y,
    Z,
)

if TYPE_CHECKING:
    from mimiqcircuits import Circuit
    from mimiqcircuits import Gate as MimiqGate

    from mpqp.core.circuit import QCircuit


@lru_cache(maxsize=1)
def _gate_constructors() -> dict[type[Gate], Callable[..., Any]]:
    import mimiqcircuits as mc

    return {
        H: mc.GateH,
        Id: mc.GateID,
        X: mc.GateX,
        Y: mc.GateY,
        Z: mc.GateZ,
        S: mc.GateS,
        S_dagger: mc.GateSDG,
        T: mc.GateT,
        SWAP: mc.GateSWAP,
        P: mc.GateP,
        Rx: mc.GateRX,
        Ry: mc.GateRY,
        Rz: mc.GateRZ,
        Rxx: mc.GateRXX,
        Ryy: mc.GateRYY,
        Rzz: mc.GateRZZ,
        PRX: mc.GateR,
        U: mc.GateU,
    }


def _gate_to_mimiq(gate: Gate) -> MimiqGate:
    import mimiqcircuits as mc

    if isinstance(gate, ControlledGate):
        return mc.Control(len(gate.controls), _gate_to_mimiq(gate.non_controlled_gate))
    if isinstance(gate, (Rk, Rk_dagger)):
        return mc.GateP(gate.theta)
    constructor = _gate_constructors().get(type(gate))
    if constructor is not None:
        return constructor(*getattr(gate, 'parameters', ()))
    # The canonical matrix acts only on the gate's own targets, even when
    # their physical qubit indices are far apart.
    return mc.GateCustom(np.asarray(gate.to_canonical_matrix()))


def _push_gate(circuit: Circuit, gate: Gate) -> None:
    gate_set = set(_gate_constructors()) | {Rk, Rk_dagger}
    # Controlled gates retain their native control operation and never need
    # a dense matrix spanning their control qubits.
    instructions = (
        (gate,) if isinstance(gate, ControlledGate) else resolve_gate(gate, gate_set)
    )
    for instruction in instructions:
        targets = list(instruction.targets)
        if isinstance(instruction, ControlledGate):
            targets = list(instruction.controls) + targets
        circuit.push(_gate_to_mimiq(instruction), *targets)


def mpqp_to_mimiq(
    circuit: QCircuit,
    skip_pre_measure: bool = False,
    skip_measurements: bool = False,
) -> Circuit:
    """Translate a circuit without modifying its instructions or measurements.

    Gates and measurement targets keep their MPQP qubit indices. Global phase
    and unused qubits are preserved. Classical measurements are written to
    the registers selected by ``c_targets``.

    Args:
        circuit: Circuit to translate.
        skip_pre_measure: Omit basis changes preceding measurements.
        skip_measurements: Omit final measurement operations.

    Returns:
        A native MIMIQ circuit.

    Raises:
        ValueError: If the circuit has no qubits.
        NotImplementedError: If the circuit contains noise or an unsupported
            instruction or measurement.

    Example:
        >>> from mpqp import BasisMeasure, H, Language, QCircuit
        >>> circuit = QCircuit([H(0), BasisMeasure(shots=32)])
        >>> translated = circuit.to_other_language(Language.MIMIQ) # doctest: +MIMIQ
        >>> translated.num_qubits(), translated.num_bits() # doctest: +MIMIQ
        (1, 1)
    """
    import mimiqcircuits as mc

    if circuit.nb_qubits == 0:
        raise ValueError('MIMIQ requires at least one qubit.')
    if circuit.noises:
        raise NotImplementedError(
            'MPQP noise translation to MIMIQ is not supported yet.'
        )
    translated = mc.Circuit()
    for instruction in circuit.instructions:
        if isinstance(instruction, (Measure, Breakpoint)):
            continue
        if isinstance(instruction, Gate):
            _push_gate(translated, instruction)
        elif isinstance(instruction, Barrier):
            targets = instruction.targets or list(range(circuit.nb_qubits))
            translated.push(mc.Barrier(len(targets)), *targets)
        else:
            raise NotImplementedError(
                f'MIMIQ cannot translate {type(instruction).__name__}.'
            )

    if circuit.input_g_phase != 0:
        translated.push(mc.GateU(0, 0, 0, circuit.input_g_phase), 0)
    # Circuit size is inferred from its instructions in the native SDK.
    if translated.num_qubits() < circuit.nb_qubits:
        translated.push(mc.GateID(), circuit.nb_qubits - 1)

    for measurement in circuit.measurements:
        if not isinstance(measurement, BasisMeasure):
            raise NotImplementedError('MIMIQ currently supports BasisMeasure only.')
        if not skip_pre_measure:
            for gate in measurement.pre_measure:
                _push_gate(translated, gate)
        if not skip_measurements:
            if measurement.c_targets is None:
                raise ValueError(
                    'Measurement classical targets have not been resolved.'
                )
            for qubit, bit in zip(measurement.targets, measurement.c_targets):
                translated.push(mc.Measure(), qubit, bit)
    return translated
