Getting started
===============

Installation
------------

.. TODO: grab the compatibility matrix from MyQLM and relax our requirements 
.. when possible, test on many different configurations (tox or other ?)

For now, we support python versions 3.9 to 3.11, and both Windows, Linux and 
MacOS (specifically, Linux was validated on Ubuntu LTS 20.04, while Ubuntu 18.04 
is not supported, so your milage may vary).

.. code-block:: console

   $ pip install mpqp

.. note::
    For mac users, additional steps are required before the installation, 
    specifically because of the ``myqlm`` library. To run these steps, you can 
    either follow their instructions on 
    `this page <https://myqlm.github.io/01_getting_started/%3Amyqlm%3Amacos.html#macos>`_
    or run the script we created to facilitate this step:

    .. code-block:: bash

        curl -L https://raw.githubusercontent.com/ColibrITD-SAS/mpqp/main/mac-install.sh | bash -s -- <your-python-bin>
        
    where ``<your-python-bin>`` is the binary you use to invoke python. I could 
    for instance be ``python``, ``python3``, ``python3.9``, etc...

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
:mod:`setup_connections.py<mpqp.execution.connection.setup_connections>` can be
called from everywhere, not only in ``mpqp`` folder, using the following command
line:

.. code-block:: console

    $ setup_connections

This script will allow you to configure and save your personal account to
connect to remote machines. Depending on the provider, different credentials can
be asked. Information concerning which provider is configured and related
credentials are stored in the ``~/.mpqp`` file.

IBM Quantum
^^^^^^^^^^^

Each IBM Quantum account is associated to a unique token. It is accessible by
first logging in the `IBM Quantum Platform <https://quantum.ibm.com/>`_ and then
looking for the ``API Token`` on the top right (that you can copy). This token
is sufficient to configure your account, and to be able to submit jobs to remote
devices. When inputting your token in the MPQP configuration script, this will
configure the account for all your current environments, meaning that this
account will still be configured outside of MPQP.

QLMaaS / Qaptiva
^^^^^^^^^^^^^^^^

QLM proposes several ways of setting up the account to submit jobs on their
simulators. We made the choice to use the ``username`` and ``password``
credentials to identify yourself on the QLM. When configuring the connection
with ``setup_connections`` script, we ask you to choose between configuring
the account only in the scope of MPQP, or for your whole environment.

AWS Braket
^^^^^^^^^^

For setting up your AWS Braket account, we call the CLI ``aws configure`` that
handles it for us. It will ask you your ``AWS Access Key ID``, ``AWS Secret
Access Key`` and ``Default region name``. Note that it will configure the
account not only in MPQP scope.


Execute examples
----------------

.. code-block:: console

    $ python -m example.scripts.bell_pair
    $ python -m example.scripts.demonstration
    $ python -m example.scripts.observable_job
