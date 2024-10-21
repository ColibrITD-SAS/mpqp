Measurements
============

.. code-block:: python
    :class: import

    from mpqp.measures import *

A measurement can be used to retrieve information about your quantum state. In
``mpqp``, you can add it to a circuit as any other instruction (either when
initializing the circuit, or using the
:meth:`~mpqp.core.circuit.QCircuit.add` circuit method). All kind of
measurements are listed bellow, check them out for usage example.

However, if no measurement is added to the circuit before running it, the user
will retrieve the state as a
:class:`~mpqp.execution.result.StateVector` in the computational
basis (see section :ref:`Results`).

.. note::
    In the current version, we only allow one measurement per circuit.

The measurement
---------------

.. automodule:: mpqp.core.instruction.measurement.measure

Measuring in a basis
--------------------

Choice of the basis
^^^^^^^^^^^^^^^^^^^

.. automodule:: mpqp.core.instruction.measurement.basis

The BasisMeasure
^^^^^^^^^^^^^^^^

.. automodule:: mpqp.core.instruction.measurement.basis_measure


Measuring using an observable
-----------------------------

.. automodule:: mpqp.core.instruction.measurement.expectation_value

Pauli String
^^^^^^^^^^^^

.. automodule:: mpqp.core.instruction.measurement.pauli_string
