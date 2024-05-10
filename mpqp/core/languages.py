"""The execution of a circuit is always ran on a user selected device. Prior to
this execution, the circuit is first translated in the SDK we selected to
support this device. This is being done on the ``to_other_language`` present on
most objects of MPQP, taking a ``language`` argument. This language corresponds
to said SDK, and you can find the list of available languages in the
:class:`Language` enum."""

from enum import Enum


class Language(Enum):
    QISKIT = 0
    MY_QLM = 1
    BRAKET = 2
    CIRQ = 3
