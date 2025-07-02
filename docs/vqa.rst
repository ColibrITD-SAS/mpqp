.. _VQA:

Variational Quantum Algorithms
==============================

.. code-block:: python
    :class: import

    from mpqp.execution.vqa import *

In order to support Variational Quantum Algorithms (VQA for short), the
parametrized gates of our circuits accept `sympy <https://sympy.org>`_'s
symbolic variable as arguments.

A symbolic variable is a variable aimed at being a numeric value but without the
value attributed. It can be created as such:

.. code-block:: python

    from sympy import symbols

    theta, k = symbols("Θ k")

This concept exists more or less in all quantum circuit libraries: ``braket``
has ``FreeExpression``, ``qiskit`` has ``Parameter``, ``qlm`` has ``Variable``,
``cirq`` uses ``sympy``'s ``Symbol``, etc...

Once you define a circuit with variables, you have two options:

1. either the measure of the circuit is an 
   :class:`~mpqp.core.instruction.measurement.expectation_value.ExpectationMeasure`
   and can directly feed it in the optimizer;
2. or you can define a custom cost function for more complicated cases.

Detailed example for those two options can be found in our example notebooks.

.. automodule:: mpqp.execution.vqa.vqa

.. automodule:: mpqp.execution.vqa.optimizer

Quantum Approximate Optimization Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QUBO
++++

These classes are used to generate a Quadratic Unconstrained Binary Operation (QUBO) which can be used in many optimization problems.
In the context of MPQP, these classes are used in the QAOA module to encode the problem to optimize in the function :func:`~mpqp.execution.vqa.qaoa.qaoa_solver`.

.. autoclass:: mpqp.execution.vqa.qubo.Qubo

.. autoclass:: mpqp.execution.vqa.qubo.QuboAtom

QAOA
++++

This module is one implementation of a particular type of VQA : QAOA. 

| This algorithm works by generating a circuit of alternating operators : cost operators and mixer operators.
| Cost operators are generated with the cost hamiltonian which represents the problem we want to optimize.
| Mixer operators are here to "search" for solutions, they can be custom to the problem but a generic set does exist.

.. automodule:: mpqp.execution.vqa.qaoa
