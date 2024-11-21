"""A circuit is always executed on a user-selected device. Prior to
this execution, the circuit is first translated into the SDK selected to
support this device. This is done by the ``to_other_language`` function present on
most objects of ``MPQP``, which takes a ``language`` argument. This language 
corresponds to the appropriate SDK, and you can find the list of available languages 
in the :class:`Language` enum."""

from enum import Enum, auto


class Language(Enum):
    """Enumerate containing all the supported languages."""

    QISKIT = auto()
    MY_QLM = auto()
    BRAKET = auto()
    CIRQ = auto()
    QASM2 = auto()
    QASM3 = auto()
