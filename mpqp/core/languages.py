"""A circuit is always executed on a user-selected device. Prior to
this execution, the circuit is first translated into the SDK selected to
support this device. This is done by the ``to_other_language`` function present on
most objects of ``MPQP``, which takes a ``language`` argument. This language 
corresponds to the appropriate SDK, and you can find the list of available languages in the
:class:`Language` enum.

.. warning::
    The current version of ``qiskit`` we are using depends on a package with a 
    known vulnerability. The risks associated with this vulnerability are not 
    high enough to justify immediate action, but we will fix this as soon as 
    possible by upgrading the ``qiskit`` version. You can find information about 
    this vulnerability here: 
    https://github.com/ColibrITD-SAS/mpqp/security/dependabot/1.
"""

from enum import Enum


class Language(Enum):
    """Enumerate containing all the supported languages."""

    QISKIT = 0
    MY_QLM = 1
    BRAKET = 2
    CIRQ = 3
