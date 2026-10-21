Execution
=========

Execution is the core of this library. Our goal is to allow you to run a circuit
on any hardware without you having to rewrite your circuit in the providers'SDK. 
We introduce here how execution works in ``MPQP``, both in local simulator and 
in remote QPUs.

Languages
---------

.. automodule:: mpqp.core.languages

.. _Devices:

Devices
-------

.. automodule:: mpqp.execution.devices

.. _SimulatedDevices:

Simulated Devices
-----------------

.. automodule:: mpqp.execution.simulated_devices

Running a circuit
-----------------

.. automodule:: mpqp.execution.runner

Helpers for remote jobs
-----------------------

.. automodule:: mpqp.execution.remote_handler

Jobs
----

.. automodule:: mpqp.execution.job

.. _Results:

Results
-------

.. automodule:: mpqp.execution.result

Looking up batch results
~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~mpqp.execution.result.BatchResult` accepts the submitted circuit as an
index. No freezing is required:

.. code-block:: python

    from copy import deepcopy
    from mpqp import QCircuit, H, X, IBMDevice, run

    circuit = QCircuit([H(0)])
    other = QCircuit([X(0)])
    batch = run([circuit, other], IBMDevice.AER_SIMULATOR)

    result = batch[circuit]
    assert batch[deepcopy(circuit)] is result
    assert batch[0] is result

Lookup compares exact circuit structure: dimensions, ordered instructions,
parameters, measurements, noise, and input global phase. It ignores circuit and
instruction display labels, inferred-size bookkeeping, and execution caches.
Observable labels remain significant because they name values in
multi-observable results. Numerically close matrices and different observable
representations are not interchangeable lookup keys. Existing public equality
and numerical equivalence methods are unchanged.

The batch captures independent immutable keys before execution, including before
symbolic substitution and provider adaptation. Later edits do not change stored
entries:

.. code-block:: python

    original = deepcopy(circuit)
    circuit.add(X(0))
    assert batch[original] is result
    # batch[circuit] now raises KeyError: this modified content was not submitted.

Repeated circuits and runs on multiple devices retain every individual result.
When several results match, circuit indexing raises ``ValueError``; use
:meth:`~mpqp.execution.result.BatchResult.get_all` to retrieve them in execution
order:

.. code-block:: python

    repeated = run([other, deepcopy(other)], IBMDevice.AER_SIMULATOR)
    matches = repeated.get_all(other)
    assert len(matches) == 2

Integer indexing, slicing, iteration, and ``len(batch)`` refer to individual runs
in execution order. ``batch.results`` returns an independent list; modifying that
list does not edit the batch. A missing circuit raises ``KeyError`` for indexing
and returns an empty list from ``get_all``. Unsupported custom component classes
raise ``TypeError`` rather than being matched using incomplete information.

Saved batches preserve submitted keys as optional JSON metadata. Older records
and manually constructed results without a submitted key use their job circuit
when the batch is constructed. That circuit may already contain substitutions or
provider adjustments.
