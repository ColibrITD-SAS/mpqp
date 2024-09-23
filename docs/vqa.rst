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