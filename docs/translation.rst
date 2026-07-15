Translation
===========
.. note::
    This module is separated in two parts:

    - QASM converter
    - Direct MPQP to provider and provider to MPQP translations

.. code-block:: python
    :class: import

    from mpqp.translation import *

QASM converter
^^^^^^^^^^^^^^

.. code-block:: python
    :class: import

    from mpqp.translation.qasm import *

To allow interoperability with different providers, we leverage the OpenQASM
standard, broadly adopted by the community. We describe here how to convert
OpenQASM from a version to another, and how to generate supported providers'
circuits from OpenQASM code.

.. note::
    To learn more about how we generate OpenQASM code from a 
    :class:`~mpqp.core.circuit.QCircuit`, have a look at the
    :meth:`~mpqp.core.circuit.QCircuit.to_other_language`.

OpenQASM2.0 and OpenQASM3.0 utility 
-----------------------------------

.. automodule:: mpqp.translation.qasm.open_qasm_2_and_3

From OpenQASM to the providers
------------------------------

We use OpenQASM as the standard allowing us to translate between various SDKs,
each conversion method is listed bellow.

Qiskit
------

.. automodule:: mpqp.translation.qasm.qasm_to_qiskit

MyQLM
-----

.. automodule:: mpqp.translation.qasm.qasm_to_myqlm
.. automodule:: mpqp.translation.qasm.myqlm_to_mpqp

Braket
------

.. automodule:: mpqp.translation.qasm.qasm_to_braket

Cirq
----

.. automodule:: mpqp.translation.qasm.qasm_to_cirq


Direct translations
^^^^^^^^^^^^^^^^^^^

This part of the module host the functions called by the methods to_other_language and from_other_language
of the QCircuit object.

.. code-block:: python
    :class: import

    from mpqp.translation import *

Qiskit
------

.. automodule:: mpqp.translation.qiskit

Braket
------

.. automodule:: mpqp.translation.braket

Cirq
----

.. automodule:: mpqp.translation.cirq