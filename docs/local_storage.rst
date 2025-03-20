Local Storage
=============

.. code-block:: python
    :class: import

    from mpqp.local_storage import *

The local_storage module is responsible for handling the storage of results and
jobs data locally on the user's device using a SQLite database. The functions
defined here are for the most part callable directly on the
:class:`~mpqp.execution.result.Result`, 
:class:`~mpqp.execution.result.BatchResult` and :class:`~mpqp.execution.job.Job`
classes. 

Functions provided here are decomposed in two broad categories: 

- the ones used to interact with MPQP objects by the end user, presented in
  sections :ref:`Saving`, :ref:`Loading` and :ref:`Deleting`,
- the ones used to interact more closely with the database, aimed more at
  internal use, presented in sections :ref:`Setup` and :ref:`Querying`.

.. _Saving:

Saving
------

.. automodule:: mpqp.local_storage.save

.. _Loading:

Loading
-------

.. automodule:: mpqp.local_storage.load

.. _Deleting:

Deleting
--------

.. automodule:: mpqp.local_storage.delete

.. _Setup:

The Setup
---------

.. automodule:: mpqp.local_storage.setup

.. _Querying:

Querying
--------

.. automodule:: mpqp.local_storage.queries
