"""Immutable, provider-independent snapshots used for result lookup.

Keys describe exact structure and labels, rather than numerical equivalence.
Inferred-size bookkeeping and execution caches are deliberately excluded. This is
an internal protocol: mutable MPQP objects themselves remain unchanged.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum as PythonEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Iterable, cast

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
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_hash", hash((self.circuit, self.device, self.measurement))
        )

    def __hash__(self) -> int:
        return self._hash

    def __reduce__(
        self,
    ) -> tuple[type[ResultKey], tuple[ObjectKey, ObjectKey, ObjectKey]]:
        # Hashes depend on the Python process; reconstruct instead of serializing.
        return ResultKey, (self.circuit, self.device, self.measurement)


def result_key(
    circuit: QCircuit, device: AvailableDevice, snapshot: Snapshot | None = None
) -> ResultKey:
    """Capture criteria from a job circuit or an input before execution."""
    snapshot = snapshot if snapshot is not None else Snapshot()
    measurement = circuit.measurements[0] if circuit.measurements else None
    return ResultKey(snapshot(circuit), snapshot(device), snapshot(measurement))


class Snapshot:
    """Reuse snapshots only during one capture, retaining sources against id reuse."""

    def __init__(self) -> None:
        self._memo: dict[int, tuple[Any, ObjectKey]] = {}

    def __call__(self, value: Any) -> ObjectKey:
        # Avoid generic dispatch and memo bookkeeping for common Python scalars.
        value_type = type(value)
        if value is None or value_type in (str, bool):
            return ("scalar", value)
        if value_type in (int, float, complex):
            return _number_key(value)
        # Scalar keys are cheaper to construct than to memoize.
        memoize = isinstance(value, (np.ndarray, list, tuple)) or type(
            value
        ).__module__.startswith("mpqp.")
        if not memoize:
            return _object_key(value, self)
        previous = self._memo.get(id(value))
        if previous is not None:
            return previous[1]
        key = _object_key(value, self)
        self._memo[id(value)] = (value, key)
        return key


def object_key(value: Any) -> ObjectKey:
    """Capture an immutable structural key without provider translation."""
    return Snapshot()(value)


def _object_key(value: Any, snapshot: Snapshot) -> ObjectKey:
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
        return _number_key(value)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, np.ndarray):
        return _array_key(value, snapshot)
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(snapshot(item) for item in value))
    if not type(value).__module__.startswith("mpqp."):
        raise TypeError(
            f"Result lookup keys do not support {type(value).__qualname__}; "
            "custom domain classes need an explicit structural key implementation"
        )
    return _domain_key(value, snapshot)


def _number_key(value: Any) -> ObjectKey:
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


def _array_key(value: np.ndarray[Any, Any], snapshot: Snapshot) -> ObjectKey:
    # Native floats and exactly representable integers use compiled bulk copies.
    # A common endian/layout and positive zero make equal numeric dtypes agree.
    kind = value.dtype.kind
    fast = (kind == "f" and value.dtype.itemsize <= 8) or (
        kind == "c" and value.dtype.itemsize <= 16
    )
    if kind in "iu":
        fast = bool(
            np.all(value <= 2**53) and (kind == "u" or np.all(value >= -(2**53)))
        )
    if fast:
        if not np.all(np.isfinite(value)):
            raise ValueError("Non-finite numbers cannot be used in result lookup keys")
        packed = np.array(value, dtype="<c16", order="C", copy=True)
        parts = packed.reshape(-1).view("<f8")
        parts[parts == 0] = 0
        return ("array-v2", value.shape, packed.tobytes(), ())
    return _pack_array(value.shape, (snapshot(item) for item in value.flat))


def _pack_array(shape: Hashable, items: Iterable[ObjectKey]) -> ObjectKey:
    """Keep nonnumeric or higher precision entries as exact sparse exceptions."""
    entries = list(items)
    packed = np.zeros(len(entries), dtype="<c16")
    exceptions: list[tuple[int, ObjectKey]] = []
    for index, key in enumerate(entries):
        if key[0] == "number":
            real, imag = key[1:]
            number = 0j
            try:
                number = complex(cast(Any, real), cast(Any, imag))
                # Compare integers to integers: NumPy's mixed float/int equality
                # can otherwise round away differences above 2**53.
                exact = (
                    all(
                        original
                        == (
                            int(converted)
                            if isinstance(original, (int, np.integer))
                            else converted
                        )
                        for original, converted in (
                            (real, number.real),
                            (imag, number.imag),
                        )
                    )
                    and math.isfinite(number.real)
                    and math.isfinite(number.imag)
                )
            except (OverflowError, ValueError):
                exact = False
            if exact:
                packed[index] = number
                continue
        exceptions.append((index, key))
    parts = packed.view("<f8")
    parts[parts == 0] = 0
    return ("array-v2", shape, packed.tobytes(), tuple(exceptions))


def _domain_key(value: Any, snapshot: Callable[[Any], ObjectKey]) -> ObjectKey:
    """Snapshot domain fields without introspecting provider properties."""
    (
        QCircuit,
        Barrier,
        Breakpoint,
        ControlledGate,
        Gate,
        CustomGate,
        UnitaryMatrix,
        NativeGate,
        ParametrizedGate,
        BasisMeasure,
        Basis,
        ExpectationMeasure,
        Observable,
        PauliString,
        PauliStringAtom,
        PauliStringMonomial,
        NoiseModel,
    ) = _domain_types()

    def fields(*names: str) -> ObjectKey:
        if hasattr(value, "label") and "label" not in names:
            names = (*names, "label")
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            tuple((name, snapshot(getattr(value, name))) for name in names),
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
        source = value._key_source
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
    return json.dumps((1, key), allow_nan=False, default=_json_bytes)


def decode_key(data: str) -> ObjectKey:
    """Restore immutable key data without importing or evaluating stored names."""
    version, restored = _decode_payload(data)
    if version != 1:
        raise ValueError(f"Unsupported result lookup key version: {version}")
    return restored


def encode_result_key(key: ResultKey) -> str:
    """Store submitted criteria as versioned JSON in the legacy metadata column."""
    return json.dumps(
        (4, (key.circuit, key.device, key.measurement)),
        allow_nan=False,
        default=_json_bytes,
    )


def decode_result_key(
    data: str, device: AvailableDevice, circuit: QCircuit
) -> ResultKey:
    """Restore criteria; older label-free keys fall back to the recorded job."""
    version, value = _decode_payload(data)
    if version in (3, 4):
        if len(value) != 3 or not all(isinstance(item, tuple) for item in value):
            raise ValueError("Invalid submitted result criteria")
        circuit_key, device_key, measurement = cast(
            tuple[ObjectKey, ObjectKey, ObjectKey], value
        )
        return ResultKey(circuit_key, device_key, measurement)
    if version in (1, 2):
        # These formats omitted labels, which cannot be recovered reliably from
        # the original snapshots. Never treat a missing label as a wildcard.
        return result_key(circuit, device)
    raise ValueError(f"Unsupported result lookup key version: {version}")


def _decode_payload(data: str) -> tuple[int, ObjectKey]:
    version, value = json.loads(data)
    if version not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported result lookup key version: {version}")

    def restore(item: Any) -> Hashable:
        if isinstance(item, list):
            restored = tuple(restore(child) for child in item)
            if len(restored) == 3 and restored[0] == "array":
                return _pack_array(
                    restored[1], cast(tuple[ObjectKey, ...], restored[2])
                )
            return restored
        if isinstance(item, dict) and set(item) == {"bytes"}:
            return base64.b64decode(item["bytes"], validate=True)
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise ValueError("Invalid result lookup key data")

    restored = restore(value)
    if not isinstance(restored, tuple):
        raise ValueError("Invalid result lookup key data")
    return version, restored


def _json_bytes(value: Any) -> dict[str, str]:
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Unsupported lookup key value: {type(value).__qualname__}")


@lru_cache(maxsize=1)
def _domain_types() -> tuple[type, ...]:
    """Resolve domain types once, after core imports have completed."""
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

    return (
        QCircuit,
        Barrier,
        Breakpoint,
        ControlledGate,
        Gate,
        CustomGate,
        UnitaryMatrix,
        NativeGate,
        ParametrizedGate,
        BasisMeasure,
        Basis,
        ExpectationMeasure,
        Observable,
        PauliString,
        PauliStringAtom,
        PauliStringMonomial,
        NoiseModel,
    )
