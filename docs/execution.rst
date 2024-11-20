Execution
=========

.. code-block:: python
    :class: import

    from mpqp.execution import *

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