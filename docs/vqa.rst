.. _VQA:

Variational Quantum Algorithms
==============================

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

See full example of this module in this :doc:`dedicated notebook <notebooks/8_TSP_QAOA>`.


QUBO
++++

.. automodule:: mpqp.execution.vqa.qubo

QAOA
++++

.. automodule:: mpqp.execution.vqa.qaoa

Sequential VQA module
---------------------

``VQAModule`` owns copies of the supplied circuits and prepares their parametric
provider representation once. Each evaluation binds a fresh copy and runs one
circuit at a time. Multiple circuits are evaluated sequentially; batched bindings,
multiple PUBs and simultaneous jobs are outside this API's current scope.

.. code-block:: python

    import numpy as np
    from sympy import symbols
    from mpqp import QCircuit, Ry, ExpectationMeasure, Observable, pZ, IBMDevice
    from mpqp.execution.vqa.vqa import VQAModule, OptimizerData, Optimizer

    theta, scale = symbols("theta scale")
    circuit = QCircuit([
        Ry(2 * theta, 0),
        ExpectationMeasure(Observable(pZ)),
    ])

    def loss(params, results):
        expectation = results[0].expectation_values
        return (params[1] * expectation - 0.5) ** 2 + 0.01 * params[1] ** 2

    vqa = VQAModule(
        circuit,
        IBMDevice.AER_SIMULATOR,
        parameters=[theta, scale],
        cost_function=loss,
    )
    result = vqa.minimize(OptimizerData(
        Optimizer.BFGS, init_params=[0.3, 1.0], maxiter=100,
    ))
    print(result.loss, result.angles)

``parameters`` defines the vector order explicitly and can include parameters
used only in the classical cost. If omitted, circuit symbols follow SymPy's
sorting order, exposed by ``vqa.variables``. Gate expressions such as
``2 * theta`` are compiled once from their original symbolic expression.
No provider parameter name is parsed with ``eval``.

``evaluate(params)`` returns a tuple of raw MPQP results in circuit order.
``cost(params)`` applies the custom cost to these results. Without a custom cost,
all expectation values are summed, including dictionaries of observables.
For sampling, supply a circuit with ``BasisMeasure``; for statevectors, supply
one without measurements and consume its probabilities or amplitudes in the
custom cost. ``shots=None`` preserves each measurement's configuration; an
integer overrides it only for that evaluation or optimization stage.
For ``BasisMeasure``, switching between zero and positive shots changes the
measurement instructions. The module prepares and caches the second parametric
variant on first use; subsequent evaluations reuse it.

A residual function for ``scipy.optimize.least_squares`` can call ``evaluate``
and construct its residual vector from the returned results. An analytical
Jacobian can be supplied to that optimizer directly. For ``minimize``, use
``OptimizerData(jac=...)`` to differentiate the entire objective, including its
classical terms. Otherwise SciPy uses numerical differentiation. A two-point
parameter-shift formula is not applied automatically to nonlinear costs,
scaled gate angles or parameters shared by several gates.

For example, a residual vector can combine a quantum prediction and a classical
regularization term while reusing the module above:

.. code-block:: python

    from scipy.optimize import least_squares

    def residuals(params):
        expectation = vqa.evaluate(params)[0].expectation_values
        return np.array([
            params[1] * expectation - 0.5,
            np.sqrt(0.01) * params[1],
        ])

    fitted = least_squares(residuals, x0=[0.3, 1.0], method="trf")

The optimizer calls ``residuals`` with each new parameter vector. A custom
Jacobian must likewise evaluate at its argument rather than depend on values
left over from a previous cost call. Apply parameter shift to the appropriate
quantum expectations, then the chain rule to the residuals and classical terms.

``VQAResult.optimizer_results`` retains the SciPy result, including convergence
status. ``loss_total`` records objective evaluations. Optimizer options and the
user's circuits are not mutated. The older ``minimize(eval_func=...)`` argument
still overrides the complete objective; it bypasses the built-in execution.
Custom optimizer callables receive ``(objective, initial_parameters, options)``
and return ``(loss, parameters)``; callback, bounds and Jacobian configuration
through ``OptimizerData`` is available for SciPy optimizers only.
``Optimizer.CMAES`` uses the optional ``cma`` package and returns its best
evaluated candidate. It supports callbacks; set CMA-ES bounds and ``sigma0`` in
``optimizer_options``. Its result exposes ``fun`` and ``x`` rather than SciPy
convergence metadata.

The prepared execution path follows the existing Qiskit and Braket binding
support. The regression tests exercise local Qiskit Aer; remote devices are not
validated by those tests. Changing device or circuit structure requires a new
module. Executions copy prepared circuits for isolation; provider execution may
still perform its own internal preparation.
