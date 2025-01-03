Getting started
===============

Installation
------------

.. TODO: grab the compatibility matrix from MyQLM and relax our requirements 
.. when possible, test on many different configurations (tox or other ?)

For now, we support Python versions 3.9 to 3.11, and all of Windows, Linux and 
MacOS (specifically, Linux was validated on Ubuntu LTS 20.04, while Ubuntu 18.04 
is not supported, so your milage may vary).

To install mpqp, you can run in a terminal

.. code-block:: console

   $ pip install mpqp

And if you have already a previous version and want to update to the latest
version, run instead

.. code-block:: console

   $ pip install -U mpqp

.. note::
    For Mac users, additional steps are required before installation, 
    specifically because of the ``myqlm`` library. To run these steps, you can 
    either follow their instructions on 
    `this page <https://myqlm.github.io/01_getting_started/%3Amyqlm%3Amacos.html#macos>`_
    or run the script we created to facilitate this step:

    .. code-block:: console

        $ curl -L https://raw.githubusercontent.com/ColibrITD-SAS/mpqp/main/mac-install.sh | bash -s -- <your-python-bin>
        
    where ``<your-python-bin>`` is the binary you use to invoke python. For instance, it could
    be ``python``, ``python3``, or ``python3.9``.

.. warning::
    The migration from ``qiskit`` version ``0.x`` to ``1.x`` caused a few issues. 
    In case you had a ``qiskit 0.x`` environment, you might encounter an error 
    such as 
    
    .. code-block:: bash

        ImportError: Qiskit is installed in an invalid environment that has both Qiskit >=1.0 and an earlier version...

    To fix this, we provide a script you can run that will fix your environment.
    In a terminal, simply run.

    .. code-block:: console

        $ update_qiskit

    Alternatively, if you want to keep your old qiskit environment, you can also
    create a new virtual environment.

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
    >>> from mpqp.execution import run, IBMDevice
    >>> circuit = QCircuit([X(0), CNOT(0, 1), BasisMeasure([0, 1], shots=100)])
    >>> print(circuit)
         ┌───┐     ┌─┐
    q_0: ┤ X ├──■──┤M├───
         └───┘┌─┴─┐└╥┘┌─┐
    q_1: ─────┤ X ├─╫─┤M├
              └───┘ ║ └╥┘
    c: 2/═══════════╩══╩═
                    0  1
    >>> print(run(circuit, IBMDevice.AER_SIMULATOR))
    Result: IBMDevice, AER_SIMULATOR
     Counts: [0, 0, 0, 100]
     Probabilities: [0, 0, 0, 1]
     Samples:
      State: 11, Index: 3, Count: 100, Probability: 1
     Error: None

.. _Remote setup:

Set up remote accesses
----------------------

Installing MPQP gives you access to ``setup_connections``, a script facilitating
the setup of remote QPU connections. The supported providers (Qiskit,
Qaptiva, Braket, Azure and IonQ) can be set up from this script.  

To run the script, simply run the following command in your terminal:

.. code-block:: console

    $ setup_connections

Each of these providers has their own set of data needed to set up the connection, 
detailed up in subsections below.

To see which devices are available, checkout the :ref:`Devices` section.

IBM Quantum (Qiskit)
^^^^^^^^^^^^^^^^^^^^

For this provider, you only need your account ``API token``, which you can find on your
`account page <https://quantum.ibm.com/account>`_. The token will be configured once for all users.


Atos/Eviden (QLM/Qaptiva)
^^^^^^^^^^^^^^^^^^^^^^^^^

For this provider, several connection methods exist. For now, we only support the username/password method.
You should have received you username and password by email.


AWS Braket
^^^^^^^^^^

For configuring access to AWS Braket, you first need to have ``awscli`` installed on your machine. To check if it is
already installed, you can run this command:

.. code-block:: console

    $ aws --version

- For ``Windows``, installing ``mpqp`` can be sufficient since the Python package ``aws-configure`` (in the requirements) also installs ``awscli`` locally. If it is not the case, you can execute the following command (in a terminal where you have admin access) to install ``awscliV2``:

    .. code-block:: console

        > msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi

- For ``MacOS``, on some versions, the installation of ``aws-configure`` can be sufficient. If it is not the case, you can install it using ``brew``:

    .. code-block:: console

        $ brew install awscli

    or execute the script we prepared for installing ``awscliv2``:

    .. code-block:: console

        $ ./mpqp/mpqp_scripts/awscli_installation/mac_awscli_install.sh

- For ``Linux``, one can use the dedicated script for installing ``awscliv2``:

    .. code-block:: console

        $ ./mpqp/mpqp_scripts/awscli_installation/linux_awscli_install.sh


Amazon Web Services propose two different ways to authenticate for access remote services (including remote
simulators and QPUs via Braket): the ``IAM`` authentication, and the ``SSO`` one. When you run the ``setup_connections``
script and select AWS configuration, you will have to choose between one of the two above.


- IAM: All the necessary credentials can be found in your `AWS console <https://console.aws.amazon.com/console/home?nc2=h_ct&src=header-signin>`_.
    In the console, go to ``IAM``. In the ``Users`` tab, click on your
    username. In the ``Security credential`` tab, you'll find an ``Access keys``
    section. In this section, you can create a new access key for ``MPQP/Braket`` (with "Local code" or
    "Third-party service" as usecase), or use an existing one. You
    should save the secret access key because you will not be able to recover it later.
    This will give you your key and your secret, but for the configuration, you
    also need to input a region (for example ``us-east-1``). In short, when running ``setup_connections``,
    it will execute the ``aws configure`` command that will ask you the following credentials:

    + ``AWS Access Key ID``,
    + ``AWS Secret Access Key``,
    + ``Default region name``.

- SSO: Standing for "Single-Sign-On", SSO enables organizations to simplify and strengthen password security by giving
    access to all connected services with a signe login. It is the recommended way to authenticate to Amazon Web Services.
    To recover your SSO credentials, you have to follow the ``SSO start url`` provided by your AWS administrator, (for
    example https://d-4859u1689s.awsapps.com/start ).

    You will need you username and password attached (and potentially MFA) to login. Then, in the ``AWS Access Portal``,
    you can find the different sso sessions and profile associated with your company account. Click on the
    ``Access key`` (with the key symbol) to retrieve your SSO credentials. When running ``setup_connections``,
    you will be asked for:

    + ``AWS Access Key ID``,
    + ``AWS Secret Access Key``,
    + ``AWS Session Token``,
    + ``Default region name``.


Microsoft Azure
^^^^^^^^^^^^^^^

For this provider, you need to have an Azure account and create an
Azure Quantum workspace. To create an Azure Quantum workspace, follow the
steps on:
`Azure Quantum workspace <https://learn.microsoft.com/en-us/azure/quantum/how-to-create-workspace?tabs=tabid-quick>`_.
Once you have your Quantum workspace, you can go to the ``Overview`` section,
where you will find your ``Resource ID`` and ``Location``.

You might encounter a pop-up requesting Azure authentication for each Azure
job submission. This occurs because your security token is reset at the end of
each session. In order to avoid this, you can use the Azure CLI.

First, install the Azure CLI: refer to the guide on their website
`How to install the Azure CLI <https://learn.microsoft.com/en-us/cli/azure/install-azure-cli>`_.

Next, log in to Azure:

- Using interactive login:

.. code-block:: console

    $ az login

- For username and password authentication (note that this method doesn't work
  with Microsoft accounts or accounts with two-factor authentication):

.. code-block:: console

    $ az login -u johndoe@contoso.com -p=VerySecret

For additional details and options, see the documentation of
`az login <https://learn.microsoft.com/en-us/cli/azure/reference-index?view=azure-cli-latest#az-login>`_.

IonQ (Cirq)
^^^^^^^^^^^

For this provider, you need to have an IonQ account and create an
  ``API token``. You can obtain it from the IonQ Console under 
  `IonQ setting keys <https://cloud.ionq.com/settings/keys>`_.



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

For more information, please refer to the `notebook page <./examples.html>`_.
