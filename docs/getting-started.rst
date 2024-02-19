Getting started
===============

Installation
------------

.. TODO: grab the compatibility matrix from MyQLM and relax our requirements 
.. when possible, test on many different configurations (tox or other ?)

Start by installing mpqp, the python version supported are between 3.9 and 3.10

.. code-block:: console

   $ pip install mpqp

Your first circuit
------------------

.. code-block:: python

    >>> from mpqp import QCircuit
    >>> from mpqp.gates import X, CNOT
    >>> from mpqp.measures import BasisMeasure
    >>> print(QCircuit([X(0),CNOT(0, 1), BasisMeasure([0, 1], shots=100)]))
         ┌───┐     ┌─┐
    q_0: ┤ X ├──■──┤M├───
         └───┘┌─┴─┐└╥┘┌─┐
    q_1: ─────┤ X ├─╫─┤M├
              └───┘ ║ └╥┘
    c: 2/═══════════╩══╩═
                    0  1



Setup remote connection
-----------------------

After you installed MPQP package using ``pip install``, the script
:mod:`setup_connections.py<mpqp.execution.connection.setup_connections>` can be called from everywhere, not only in
``mpqp`` folder, using the following command line:

.. code-block:: console

    $ setup_connections

This script will allow you to configure and save your personal account to connect to remote machines.
Depending on the provider, different credentials can be asked. Information concerning which provider is configured and
related credentials are stored in the ``~/.mpqp`` file.

IBM Quantum
^^^^^^^^^^^

Each IBM Quantum account is associated to a unique token. It is accessible by first logging in the
`IBM Quantum Platform <https://quantum.ibm.com/>`_ and then looking for the ``API Token`` on the top right (that you can
copy). This token is sufficient to configure your account, and to be able to submit job to remote devices. When inputting
you token in the MPQP configuration script, this will configure the account for all your current environment, meaning
that this account will still be configured outside of MPQP.

QLMaaS / Qaptiva
^^^^^^^^^^^^^^^^

QLM propose several way of setting up the account to submit jobs on their simulators. We made the choice to use the
``username`` and ``password`` credentials to identify yourself on the QLM. When configuring the connection with
``setup_connections`` script, we ask you the between configuring the account only in the scope of MPQP, or for your whole
environment.

AWS Braket
^^^^^^^^^^

For setting up your AWS Braket account, we call the CLI ``aws configure`` that handles it for us. It will ask you your
``AWS Access Key ID``, ``AWS Secret Access Key`` and ``Default region name``. Note that it will configure the account
not only in MPQP scope.


Execute examples
----------------

.. code-block:: console

    $ python -m example.bell_pair
    $ python -m example.demonstration
    $ python -m example.observable_job