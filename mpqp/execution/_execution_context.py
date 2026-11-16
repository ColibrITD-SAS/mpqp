"""Internal execution metadata shared by binding translation and extraction."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pytket.circuit import Circuit as tket_Circuit

    from mpqp.core.circuit import QCircuit
    from mpqp.execution.job import Job


@dataclass(frozen=True)
class ExecutionContext:
    """Associate a job with its occurrence index in binding unroll order."""

    job: "Job"
    index: int


class PreparedBindingExecutionGroup(NamedTuple):
    """Group binding executions before translating their circuit.

    These executions use the same original circuit and measurement, so the
    provider can translate the circuit once and reuse it for every parameter
    set stored in their contexts.
    """

    original_circuit: "QCircuit"
    execution_circuit: "QCircuit"
    contexts: list[ExecutionContext]


class TranslatedBindingExecutionGroup(NamedTuple):
    """Pair one translated TKET circuit with the executions that use it.

    The circuit can be compiled once and then reused with the parameter values
    stored in each execution context.
    """

    circuit: "tket_Circuit"
    contexts: list[ExecutionContext]
