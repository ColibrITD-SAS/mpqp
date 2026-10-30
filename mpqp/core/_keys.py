"""Provider-independent structural comparisons captured during result filtering.

Keys describe exact structure and labels, rather than numerical equivalence.
Inferred-size bookkeeping and execution caches are deliberately excluded. This is
an internal protocol: mutable MPQP objects themselves remain unchanged.
Snapshots live only for one query, so later queries see current metadata.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from enum import Enum as PythonEnum
from functools import lru_cache
from typing import Any, Callable, Iterable, cast

import numpy as np
from aenum import Enum
from sympy import Basic, srepr

ObjectKey = tuple[Hashable, ...]


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
        memoize, encoder = _key_encoder(value_type)
        if not memoize:
            return encoder(value, self)
        previous = self._memo.get(id(value))
        if previous is not None:
            return previous[1]
        key = encoder(value, self)
        self._memo[id(value)] = (value, key)
        return key


@lru_cache(maxsize=None)
def _key_encoder(value_type: type) -> tuple[bool, Callable[[Any, Snapshot], ObjectKey]]:
    """Resolve traversal once per concrete type, without caching mutable values."""
    if issubclass(value_type, np.ndarray):
        return True, _array_key
    if issubclass(value_type, (list, tuple)):
        return True, _sequence_key
    if value_type.__module__.startswith("mpqp.") and not issubclass(
        value_type, (Enum, PythonEnum, type)
    ):
        native_encoder = _native_gate_encoder(value_type)
        return True, native_encoder if native_encoder is not None else _domain_key
    return False, _object_key


def _native_gate_encoder(
    value_type: type,
) -> Callable[[Any, Snapshot], ObjectKey] | None:
    """Specialize common gates while preserving the general structural format."""
    if value_type.__module__ != "mpqp.core.instruction.gates.native_gates":
        return None
    from mpqp.core.instruction.gates import ControlledGate
    from mpqp.core.instruction.gates.native_gates import NativeGate
    from mpqp.core.instruction.gates.parametrized_gate import ParametrizedGate

    if not issubclass(value_type, NativeGate):
        return None
    controlled = issubclass(value_type, ControlledGate)
    parametrized = issubclass(value_type, ParametrizedGate)
    module, name = value_type.__module__, value_type.__qualname__

    def encode(value: Any, snapshot: Snapshot) -> ObjectKey:
        if not hasattr(value, "label"):
            return _domain_key(value, snapshot)
        fields = [("targets", _numeric_sequence_key(value.targets, snapshot))]
        if controlled:
            fields.extend(
                (
                    ("controls", _numeric_sequence_key(value.controls, snapshot)),
                    ("non_controlled_gate", snapshot(value.non_controlled_gate)),
                )
            )
        if parametrized:
            fields.append(
                ("parameters", _numeric_sequence_key(value.parameters, snapshot))
            )
        fields.append(("label", snapshot(value.label)))
        return ("object", module, name, tuple(fields))

    return encode


def _numeric_sequence_key(value: Any, snapshot: Snapshot) -> ObjectKey:
    """Copy short numeric gate fields without recursively dispatching each item."""
    if type(value) not in (list, tuple):
        return snapshot(value)
    keys: list[ObjectKey] = []
    for item in value:
        item_type = type(item)
        if item_type is int:
            keys.append(("number", item, 0))
        elif item_type is float:
            if not math.isfinite(item):
                raise ValueError(
                    "Non-finite numbers cannot be used in result lookup keys"
                )
            keys.append(("number", item, 0))
        elif item_type is complex:
            if not math.isfinite(item.real) or not math.isfinite(item.imag):
                raise ValueError(
                    "Non-finite numbers cannot be used in result lookup keys"
                )
            keys.append(("number", item.real, item.imag))
        else:
            # Symbolic parameters and unusual user edits retain general handling.
            return snapshot(value)
    return ("sequence", tuple(keys))


def _sequence_key(value: Any, snapshot: Snapshot) -> ObjectKey:
    return ("sequence", tuple(snapshot(item) for item in value))


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
        packed = np.asarray(value, dtype="<c16", order="C")
        bits = packed.reshape(-1).view("<u8")
        negative_zeros = bits == (1 << 63)
        if np.any(negative_zeros):
            # Only negative zero needs normalization. Borrow already canonical
            # input for the final bytes copy, but never write into a source.
            if np.shares_memory(packed, value):
                packed = packed.copy()
            packed.reshape(-1).view("<u8")[negative_zeros] = 0
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


def _domain_key(value: Any, snapshot: Snapshot) -> ObjectKey:
    """Read only explicitly supported structural fields."""
    names = _domain_fields(type(value))(value)
    if hasattr(value, "label") and "label" not in names:
        names = (*names, "label")
    return (
        "object",
        type(value).__module__,
        type(value).__qualname__,
        tuple((name, snapshot(getattr(value, name))) for name in names),
    )


@lru_cache(maxsize=None)
def _domain_fields(value_type: type) -> Callable[[Any], tuple[str, ...]]:
    """Select domain fields once per concrete type, after core imports complete."""
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

    def fixed(*names: str) -> Callable[[Any], tuple[str, ...]]:
        return lambda value: names

    if issubclass(value_type, QCircuit):
        return fixed(
            "nb_qubits",
            "nb_cbits",
            "instructions",
            "measurements",
            "noises",
            "input_g_phase",
        )
    if issubclass(value_type, ControlledGate):
        names = ["targets", "controls", "non_controlled_gate"]
        if issubclass(value_type, ParametrizedGate):
            names.append("parameters")
        return fixed(*names)
    if issubclass(value_type, CustomGate):
        return fixed("targets", "definition")
    if issubclass(value_type, ParametrizedGate):
        if issubclass(value_type, NativeGate):
            return fixed("targets", "parameters")
        return fixed("targets", "parameters", "definition")
    if issubclass(value_type, Gate):
        return fixed("targets")
    if issubclass(value_type, UnitaryMatrix):
        return fixed("matrix")
    if issubclass(value_type, BasisMeasure):
        return fixed("targets", "c_targets", "shots", "basis")
    if issubclass(value_type, ExpectationMeasure):
        return fixed(
            "targets",
            "shots",
            "observables",
            "optim_diagonal",
            "commuting_type",
            "grouping_method",
            "optimize_measurement",
        )
    if issubclass(value_type, Observable):
        # Capture the source representation, without touching lazy caches.
        return lambda value: ("nb_qubits", "label", value._key_source)
    if issubclass(value_type, Basis):
        return fixed("nb_qubits", "basis_vectors", "symbols", "basis_vectors_labels")
    if issubclass(value_type, PauliStringAtom):
        return fixed("label", "matrix")
    if issubclass(value_type, PauliStringMonomial):
        return fixed("coef", "atoms")
    if issubclass(value_type, PauliString):
        return fixed("nb_qubits", "monomials")
    if issubclass(value_type, NoiseModel):
        return lambda value: (
            "targets",
            "gates",
            *(name for name in ("dimension", "prob", "gamma") if hasattr(value, name)),
        )
    if issubclass(value_type, Barrier):
        return fixed("targets", "size")
    if issubclass(value_type, Breakpoint):
        return fixed("targets", "enabled", "draw_circuit")
    raise TypeError(f"Result lookup keys do not support {value_type.__qualname__}")
