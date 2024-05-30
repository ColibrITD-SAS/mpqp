=======================
Remote devices handling
=======================

In order to facilitate the handling of remote QPUs, helper functions were
implemented. Most of them are aimed at internal usage even though you can
absolutely use them yourself. As a user, the elements of this page of interest
for you are most likely:

- IBM's :func:`get_all_job_ids<mpqp.execution.connection.ibm_connection.get_all_job_ids>`;
- Eviden's :func:`get_all_job_ids<mpqp.execution.connection.qlm_connection.get_all_job_ids>`;
- AWS's :func:`get_all_task_ids<mpqp.execution.connection.aws_connection.get_all_task_ids>`;
- CIRQ's :func:`get_all_job_ids<mpqp.execution.connection.cirq_connection.get_all_job_ids>`;
- The :ref:`con-setup` section.

Provider specifics
------------------

Even though most of our interfaces use abstractions such that you do not need to
know on which provider's QPU your code is running, we need at some point to
tackle the specifics of each providers. Most (hopefully all soon) of it is
tackle in these modules.

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

Execution
__________

.. automodule:: mpqp.execution.providers.google

.. _con-setup:

Connection setup
^^^^^^^^^^^^^^^^

key Connection
______________

.. automodule:: mpqp.execution.connection.key_connection

Connection setup script 
-----------------------

.. automodule:: mpqp_scripts.setup_connections

The details on how to get these information can be found in the section 
:ref:`Remote setup`.

On disk configuration manager
-----------------------------

.. automodule:: mpqp.execution.connection.env_manager

