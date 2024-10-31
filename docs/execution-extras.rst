=======================
Remote devices handling
=======================

In order to facilitate the handling of remote QPUs, helper functions were
implemented. Most of them are aimed at internal usage even though you can
absolutely use them yourself. As a user, the elements of this page of interest
for you are most likely:

- IBM's :func:`~mpqp.execution.connection.ibm_connection.get_all_job_ids`;
- Eviden's :func:`~mpqp.execution.connection.qlm_connection.get_all_job_ids`;
- AWS's :func:`~mpqp.execution.connection.aws_connection.get_all_task_ids`;
- IonQ's :func:`~mpqp.execution.connection.ionq_connection.get_ionq_job_ids`;
- Azure's :func:`~mpqp.execution.connection.azure_connection.get_all_job_ids`;
- The :ref:`con-setup` section.

To setup your access to remote QPUs, see the :ref:`Remote setup` section.

Provider specifics
------------------

Even though most of our interfaces use abstractions such that you do not need to
know on which provider's QPU your code is running, we need at some point to
tackle the specifics of each providers. Most (hopefully all soon) of it is
tackle in these modules.

To see which devices are available, see :ref:`Devices`.

IBM
^^^

Connection
__________

.. automodule:: mpqp.execution.connection.ibm_connection

Execution
__________

.. automodule:: mpqp.execution.providers.ibm

Atos/Eviden
^^^^^^^^^^^

Connection
__________

.. automodule:: mpqp.execution.connection.qlm_connection

Execution
__________

.. automodule:: mpqp.execution.providers.atos

AWS
^^^

Connection
__________

.. automodule:: mpqp.execution.connection.aws_connection

Execution
__________

.. automodule:: mpqp.execution.providers.aws

Google
^^^^^^

Connection
__________

.. automodule:: mpqp.execution.connection.google_connection

.. _cirq-exec:

Execution
__________

.. automodule:: mpqp.execution.providers.google

IonQ
^^^^

Connection
__________

.. automodule:: mpqp.execution.connection.ionq_connection

Azure
^^^^^

Connection
__________

.. automodule:: mpqp.execution.connection.azure_connection

Execution
__________

IonQ's hardware is accessible through cirq, so see Circ's :ref:`cirq-exec` 
section for the functions used for IonQ's hardware.

.. _con-setup:

Connection setup
----------------

Connection setup script 
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mpqp_scripts.setup_connections

The details on how to get these information can be found in the section 
:ref:`Remote setup`.

On disk configuration manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mpqp.execution.connection.env_manager

