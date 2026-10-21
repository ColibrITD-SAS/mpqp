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

Filtering batch results
~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~mpqp.execution.result.BatchResult` can filter results by submitted
circuit, device, and measurement. No freezing is required. Every call to
:meth:`~mpqp.execution.result.BatchResult.get_matching` returns an independent list,
including repeated runs in execution order:

.. code-block:: python

    from copy import deepcopy
    from mpqp import QCircuit, H, X, BasisMeasure, IBMDevice, run

    circuit = QCircuit([H(0)])
    other = QCircuit([X(0), BasisMeasure([0], shots=16)])
    batch = run([circuit, other], IBMDevice.AER_SIMULATOR)

    results = batch.get_matching(circuit=circuit)
    assert batch.get_matching(circuit=deepcopy(circuit)) == results
    assert results == [batch[0]]

    # Supplied criteria combine with AND.
    samples = batch.get_matching(
        device=IBMDevice.AER_SIMULATOR,
        measurement=other.measurements[0],
    )
    assert samples == [batch[1]]

No criteria returns all results; criteria with no matches return an empty list.
Omitting ``measurement`` leaves measurements unrestricted. Passing
``measurement=None`` selects runs without an explicit measurement, which is
different from a zero-shot :class:`~mpqp.core.instruction.measurement.basis_measure.BasisMeasure`.
Passing ``circuit=None`` or ``device=None`` leaves that criterion unrestricted.

Circuit matching compares the complete submitted structure: dimensions, ordered instructions,
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
    assert batch.get_matching(circuit=original) == results
    assert batch.get_matching(circuit=circuit) == []

Repeated circuits and runs on multiple devices retain every individual result:

.. code-block:: python

    repeated = run([other, deepcopy(other)], IBMDevice.AER_SIMULATOR)
    matches = repeated.get_matching(circuit=other)
    assert len(matches) == 2

Integer indexing, slicing, iteration, and ``len(batch)`` refer to individual runs
in execution order. ``batch.results`` returns an independent list; modifying that
list does not edit the batch. Indexing accepts integer positions and slices;
use ``get_matching`` to filter by content. Unsupported criterion types and custom
component classes raise ``TypeError`` rather than being matched using incomplete information.

Saved batches preserve all three submitted criteria as optional versioned JSON
metadata. Earlier circuit-only metadata retains the original circuit and
measurement and uses the saved job device. Older records without metadata and
manually constructed results use their job metadata when the batch is
constructed. That metadata may already contain substitutions or provider
adjustments.
