"""Internal execution metadata shared by binding translation and extraction."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpqp.execution.job import Job


@dataclass(frozen=True)
class ExecutionContext:
    """Associate a job with its occurrence index in binding unroll order."""

    job: "Job"
    index: int
