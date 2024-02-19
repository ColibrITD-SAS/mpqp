Gates
=====

.. code-block:: python

    from mpqp.gates import *

Our gates class definitions are very declarative: if the gate operates on only
one qubit, it takes
:class:`SingleQubitGate<mpqp.core.instruction.gates.gate.SingleQubitGate>` as
parent, if it is a rotation gate, it takes
:class:`RotationGate<mpqp.core.instruction.gates.gate.RotationGate>` as parent,
etc... This allows us to factorize a lot of common behaviors.\ [#traits]_ 

The Gate class
--------------

.. automodule:: mpqp.core.instruction.gates.gate

The gate definition
-------------------

.. automodule:: mpqp.core.instruction.gates.gate_definition

Controlled Gates
----------------

.. automodule:: mpqp.core.instruction.gates.controlled_gate

Parametrized Gates
------------------

.. automodule:: mpqp.core.instruction.gates.parametrized_gate

Native Gates
------------

.. automodule:: mpqp.core.instruction.gates.native_gates

.. [#traits] This in fact is somewhat twisting the way inheritance usually 
   works in python, to make it into a feature existing in other languages, such 
   as traits in rust.