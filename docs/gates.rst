Gates
=====

.. code-block:: python
    :class: import

    from mpqp.gates import *

Our gates class definitions are very declarative: if the gate operates on only
one qubit, it takes
:class:`~mpqp.core.instruction.gates.gate.SingleQubitGate` as
parent, if it is a rotation gate, it takes
:class:`~mpqp.core.instruction.gates.native_gates.RotationGate` as parent,
etc. This allows us to factorize a lot of common behaviors.\ [#traits]_

If you are not a library developer, the most important section of this page for
you is very likely the :ref:`NativeGates` one.

The Gate class
--------------

.. automodule:: mpqp.core.instruction.gates.gate

Controlled Gates
----------------

.. automodule:: mpqp.core.instruction.gates.controlled_gate

Parametrized Gates
------------------

.. automodule:: mpqp.core.instruction.gates.parametrized_gate

.. _NativeGates:

Native Gates
------------

.. automodule:: mpqp.core.instruction.gates.native_gates

.. [#traits] This in fact is somewhat twisting the way inheritance usually 
   works in python, to make it into a feature existing in other languages, such 
   as traits in rust.

The GateDefinition
------------------

.. automodule:: mpqp.core.instruction.gates.gate_definition

Custom Gates
------------

.. automodule:: mpqp.core.instruction.gates.custom_gate
