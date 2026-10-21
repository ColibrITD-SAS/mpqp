"""Immutable, provider-independent snapshots used for result lookup.

Keys describe exact structure, rather than numerical equivalence. Display labels,
inferred-size bookkeeping and execution caches are deliberately excluded. This is
an internal protocol: mutable MPQP objects themselves remain unchanged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum as PythonEnum
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from aenum import Enum
from sympy import Basic, srepr

ObjectKey = tuple[Hashable, ...]

if TYPE_CHECKING:
    from mpqp.core.circuit import QCircuit
    from mpqp.execution.devices import AvailableDevice


@dataclass(frozen=True)
class ResultKey:
    """Independent submitted criteria identifying one group of executions."""

    circuit: ObjectKey
    device: ObjectKey
    measurement: ObjectKey


def result_key(circuit: QCircuit, device: AvailableDevice) -> ResultKey:
    """Capture criteria from a job circuit or an input before execution."""
    measurement = circuit.measurements[0] if circuit.measurements else None
    return ResultKey(object_key(circuit), object_key(device), object_key(measurement))


def object_key(value: Any) -> ObjectKey:
    """Copy supported domain values into an immutable structural key.

    Tuples compose recursively, including future ``(circuit, measurement)`` keys.
    Unsupported objects fail explicitly rather than falling back to identity or
    a display representation. No provider translation or circuit matrix is used.
    """
    if value is None or isinstance(value, (str, bool, np.bool_)):
        return ("scalar", value.item() if isinstance(value, np.bool_) else value)
    if isinstance(value, (Enum, PythonEnum)):
        return (
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            getattr(value, "name"),
        )
    if isinstance(value, Basic):
        return ("symbolic", srepr(value))
    if isinstance(value, (int, float, complex, np.number)):
        # Python numeric equality normalizes e.g. 1, 1.0, and 1+0j. Retain
        # arbitrary-size integers instead of converting all values to complex.
        real, imag = value.real, value.imag
        if not isinstance(real, (int, np.integer)) and not math.isfinite(real):
            raise ValueError("Non-finite numbers cannot be used in result lookup keys")
        if not math.isfinite(imag):
            raise ValueError("Non-finite numbers cannot be used in result lookup keys")
        return (
            "number",
            real.item() if isinstance(real, np.generic) else real,
            imag.item() if isinstance(imag, np.generic) else imag,
        )
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, np.ndarray):
        return ("array", value.shape, tuple(object_key(item) for item in value.flat))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(object_key(item) for item in value))
    if not type(value).__module__.startswith("mpqp."):
        raise TypeError(
            f"Result lookup keys do not support {type(value).__qualname__}; "
            "custom domain classes need an explicit structural key implementation"
        )
    return _domain_key(value)


def _domain_key(value: Any) -> ObjectKey:
    """Snapshot domain fields without introspecting provider properties."""
    from mpqp.core.circuit import QCircuit
    from mpqp.core.instruction.barrier import Barrier
    from mpqp.core.instruction.breakpoint import Breakpoint
    from mpqp.core.instruction.gates import ControlledGate, Gate
    from mpqp.core.instruction.gates.custom_gate import CustomGate
    from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
    from mpqp.core.instruction.gates.native_gates import NativeGate
    from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate
    from mpqp.core.instruction.measurement import BasisMeasure
    from mpqp.core.instruction.measurement.basis import Basis
    from mpqp.core.instruction.measurement.expectation_value import (
        ExpectationMeasure,
        Observable,
    )
    from mpqp.core.instruction.measurement.pauli_string import (
        PauliString,
        PauliStringAtom,
        PauliStringMonomial,
    )
    from mpqp.noise.noise_model import NoiseModel

    def fields(*names: str) -> ObjectKey:
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            tuple((name, object_key(getattr(value, name))) for name in names),
        )

    if isinstance(value, QCircuit):
        return fields(
            "nb_qubits",
            "nb_cbits",
            "instructions",
            "measurements",
            "noises",
            "input_g_phase",
        )
    if isinstance(value, ControlledGate):
        names = ["targets", "controls", "non_controlled_gate"]
        if isinstance(value, ParametrizedGate):
            names.append("parameters")
        return fields(*names)
    if isinstance(value, CustomGate):
        return fields("targets", "definition")
    if isinstance(value, ParametrizedGate):
        if isinstance(value, NativeGate):
            return fields("targets", "parameters")
        return fields("targets", "parameters", "definition")
    if isinstance(value, Gate):
        return fields("targets")
    if isinstance(value, UnitaryMatrix):
        return fields("matrix")
    if isinstance(value, BasisMeasure):
        return fields("targets", "c_targets", "shots", "basis")
    if isinstance(value, ExpectationMeasure):
        return fields(
            "targets",
            "shots",
            "observables",
            "optim_diagonal",
            "commuting_type",
            "grouping_method",
            "optimize_measurement",
        )
    if isinstance(value, Observable):
        # Select the source representation without populating lazy caches or
        # constructing an exponentially large matrix from a Pauli string.
        # Observable labels name entries in multi-observable result dictionaries.
        source = value._key_source  # pyright: ignore[reportPrivateUsage]
        return fields("nb_qubits", "label", source)
    if isinstance(value, Basis):
        return fields("nb_qubits", "basis_vectors", "symbols", "basis_vectors_labels")
    if isinstance(value, PauliStringAtom):
        return fields("label", "matrix")
    if isinstance(value, PauliStringMonomial):
        return fields("coef", "atoms")
    if isinstance(value, PauliString):
        return fields("nb_qubits", "monomials")
    if isinstance(value, NoiseModel):
        names = ["targets", "gates"]
        names.extend(
            name for name in ("dimension", "prob", "gamma") if hasattr(value, name)
        )
        return fields(*names)
    if isinstance(value, Barrier):
        return fields("targets", "size")
    if isinstance(value, Breakpoint):
        return fields("targets", "enabled", "draw_circuit")
    raise TypeError(f"Result lookup keys do not support {type(value).__qualname__}")


def encode_key(key: ObjectKey) -> str:
    """Serialize key data without executable object representations."""
    return json.dumps((1, key), allow_nan=False)


def decode_key(data: str) -> ObjectKey:
    """Restore immutable key data without importing or evaluating stored names."""
    version, restored = _decode_payload(data)
    if version != 1:
        raise ValueError(f"Unsupported result lookup key version: {version}")
    return restored


def encode_result_key(key: ResultKey) -> str:
    """Store submitted criteria as versioned JSON in the legacy metadata column."""
    return json.dumps((2, (key.circuit, key.device, key.measurement)), allow_nan=False)


def decode_result_key(data: str, device: AvailableDevice) -> ResultKey:
    """Restore criteria, including circuit-only metadata from earlier drafts."""
    version, value = _decode_payload(data)
    if version == 2:
        if len(value) != 3 or not all(isinstance(item, tuple) for item in value):
            raise ValueError("Invalid submitted result criteria")
        circuit, device_key, measurement = cast(
            tuple[ObjectKey, ObjectKey, ObjectKey], value
        )
        return ResultKey(circuit, device_key, measurement)
    if version == 1:
        # Recover the original measurement embedded in the submitted circuit.
        if len(value) != 4 or value[:3] != ("object", "mpqp.core.circuit", "QCircuit"):
            raise ValueError("Invalid legacy circuit key")
        fields = dict(cast(tuple[tuple[str, ObjectKey], ...], value[3]))
        measurements = cast(tuple[ObjectKey, ...], fields["measurements"][1])
        measurement = measurements[0] if measurements else object_key(None)
        return ResultKey(value, object_key(device), measurement)
    raise ValueError(f"Unsupported result lookup key version: {version}")


def _decode_payload(data: str) -> tuple[int, ObjectKey]:
    version, value = json.loads(data)
    if version not in (1, 2):
        raise ValueError(f"Unsupported result lookup key version: {version}")

    def restore(item: Any) -> Hashable:
        if isinstance(item, list):
            return tuple(restore(child) for child in item)
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise ValueError("Invalid result lookup key data")

    restored = restore(value)
    if not isinstance(restored, tuple):
        raise ValueError("Invalid result lookup key data")
    return version, restored
