Getting started
===============

Installation
------------

.. TODO: grab the compatibility matrix from MyQLM and relax our requirements 
.. when possible, test on many different configurations (tox or other ?)

For now, we support Python versions 3.9 to 3.11 on Windows, Linux and 
MacOS (specifically, Linux was validated on Ubuntu LTS 20.04, while Ubuntu 18.04 
is not supported, so your milage may vary).

.. code-block:: console

   $ pip install mpqp

.. note::
    For Mac users, additional steps are required before installation, 
    specifically because of the ``myqlm`` library. To run these steps, you can 
    either follow their instructions on 
    `this page <https://myqlm.github.io/01_getting_started/%3Amyqlm%3Amacos.html#macos>`_
    or run the script we created to facilitate this step:

    .. code-block:: bash

        curl -L https://raw.githubusercontent.com/ColibrITD-SAS/mpqp/main/mac-install.sh | bash -s -- <your-python-bin>
        
    where ``<your-python-bin>`` is the binary you use to invoke python. For instance, it could
    be ``python``, ``python3``, or ``python3.9``.

Your first circuit
------------------

A circuit is created by providing :class:`~mpqp.core.circuit.QCircuit`
with a list of :class:`~mpqp.core.instruction.instruction.Instruction` 
(gates and measurements). To run a circuit, you can then use the 
:func:`~mpqp.execution.runner.run` function.

.. code-block:: python

    >>> from mpqp import QCircuit
    >>> from mpqp.gates import X, CNOT
    >>> from mpqp.measures import BasisMeasure
    >>> print(QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)]))
         ┌───┐     ┌─┐
    q_0: ┤ X ├──■──┤M├───
         └───┘┌─┴─┐└╥┘┌─┐
    q_1: ─────┤ X ├─╫─┤M├
              └───┘ ║ └╥┘
    c: 2/═══════════╩══╩═
                    0  1

.. _Remote setup:

Set up remote accesses
---------------------

Installing MPQP gives you access to ``setup_connections``, a script facilitating
the setup of remote QPU connections. The three supported providers (Qiskit,
Qaptiva and Braket) can be set up from this script.  

To run the script, simply run the following command in your terminal:

.. code-block:: console

    $ setup_connections

Each of these providers has their own set of data needed to set up the connection, 
summed up here:

- IBM Quantum (Qiskit): For this provider, you only need your account ``API
  token``, which you can find on your `account page <https://quantum.ibm.com/account>`_.
- Atos/Eviden (Qaptiva/QLM): For this provider, several connection methods
  exist. For now, we only support the username/password method. You should have
  received you username and password by email.
- AWS (Braket): For this provider, you will need more information. All of it can
  be found in your 
  `AWS console <https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin>`_.
  In the console, go to ``IAM service``. In the ``Users`` tab, click on your
  username. In the ``Security credential`` tab, you'll find an ``Access keys`` 
  section. In this section, you can create a new access key for ``MPQP``. You 
  should save this because you will not be able to recover it later.
  This will give you your key and your secret, but for the configuration, you 
  also need a region (for example ``us-east-1``). In short, one needs:

  + ``AWS Access Key ID``,
  + ``AWS Secret Access Key`` and
  + ``Default region name``.

To see which devices are available, see :ref:`Devices`.

Execute examples
----------------

A few examples are provided in the ``examples`` folder of the repo. To try them
out, you can either download them individually from `our GitHub repository 
<https://github.com/ColibrITD-SAS/mpqp>`_ or clone the repository and
execute them as follows:

.. code-block:: console

    $ python -m examples.scripts.bell_pair
    $ python -m examples.scripts.demonstration
    $ python -m examples.scripts.observable_job
