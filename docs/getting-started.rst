Getting started
===============

Installation
------------

.. TODO: grab the compatibility matrix from MyQLM and relax our requirements 
.. when possible, test on many different configurations (tox or other ?)

For now, we support Python versions 3.10 to 3.13, and all of Windows, Linux and 
MacOS (specifically, Linux was validated on Ubuntu LTS 20.04, while Ubuntu 18.04 
is not supported, so your milage may vary).

To install mpqp, you can run in a terminal:

.. code-block:: console

   $ pip install mpqp

If you have already a previous version and want to update to the latest version,
run instead:

.. code-block:: console

   $ pip install -U mpqp

.. note::
    To keep the installation lightweight and avoid installing unnecessary
    dependencies, each provider is distributed as a **separate pip extra**.
    By default, only the core functionalities of ``mpqp`` and qiskit local
    simulation are installed, which means that you can create and manipulate circuits, 
    but you won't be able to run them on any backend but IBM's qiskit aer backends.

Add more providers
--------------------------

``mpqp`` provides integrations with several quantum SDKs and execution backends.

- **Installing all providers**:

    .. code-block:: console

        $ pip install mpqp[all]


- **Qiskit**:

    .. code-block:: console

        $ pip install mpqp[qiskit]

.. literalinclude:: requirements_providers/qiskit.txt
   :language: text

- **Amazon Braket**:

    .. code-block:: console

        $ pip install mpqp[braket]

.. literalinclude:: requirements_providers/braket.txt
   :language: text

- **Cirq**:

    .. code-block:: console
        
        $ pip install mpqp[cirq]

.. literalinclude:: requirements_providers/cirq.txt
   :language: text

- **Azure Quantum**:

    .. code-block:: console

        $ pip install mpqp[azure]

.. literalinclude:: requirements_providers/azure.txt
   :language: text

- **myQLM**:

    .. code-block:: console

        $ pip install mpqp[myqlm]

.. literalinclude:: requirements_providers/myqlm.txt
   :language: text

- **Quantinuum Nexus**:

    .. code-block:: console

        $ pip install mpqp[quantinuum]

.. literalinclude:: requirements_providers/quantinuum.txt
   :language: text


You can also combine extras, for example, to install both Qiskit and Braket support:

    .. code-block:: console

        $ pip install mpqp[qiskit,braket]

.. note::
    For Mac users, additional steps are required before installation, 
    specifically because of the ``myqlm`` library. To run these steps, you can 
    either follow their instructions on 
    `this page <https://myqlm.github.io/01_getting_started/%3Amyqlm%3Amacos.html#macos>`_
    or run the script we created to facilitate this step:

    .. code-block:: console

        $ curl -L https://raw.githubusercontent.com/ColibrITD-SAS/mpqp/main/mac-install.sh | bash -s -- <your-python-bin>
        
    where ``<your-python-bin>`` is the binary you use to invoke python. For instance, it could
    be ``python``, ``python3``, or ``python3.10``.

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

    >>> from mpqp import QCircuit, X, CNOT, BasisMeasure, run, IBMDevice
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
Qaptiva, Braket, Azure, IonQ and Quantinuum) can be set up from this
script.

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

Run ``setup_connections``, select Amazon Braket, and choose one of the two
connection methods below. MPQP validates the selected identity with AWS STS
before saving the connection.

- **Option 1 — Existing AWS profile**: Use a profile already configured in the
    standard AWS configuration files. MPQP passes its name to Boto3, which
    resolves the profile's authentication method. This supports IAM Identity
    Center, assume-role, and other profiles supported by Boto3. MPQP stores only
    the selected name, it does not copy or modify the profile.

    For an IAM Identity Center profile, follow this flow:

    1. Configure the AWS profile once with AWS CLI v2:

       .. code-block:: console

           $ aws configure sso --profile my-aws-profile

    2. Connect MPQP to the profile:

       .. code-block:: console

           $ setup_connections

    Select ``Option 1: Existing AWS profile`` and enter ``my-aws-profile``.
    For IAM Identity Center, enter the profile name, not the ``sso-session``
    name stored in the AWS config file.

    If you later remove the Amazon Braket connection from MPQP, the AWS profile
    remains available. Run ``setup_connections`` and select the same profile to
    reconnect, you do not need to configure it again. Only renew the session if
    it has expired:

    .. code-block:: console

        $ aws sso login --profile my-aws-profile

- **Option 2 — Short-term credentials**: Use credentials supplied by an AWS
    portal or another temporary credential source. MPQP asks for:

    + ``AWS access key ID``
    + ``AWS secret access key``
    + ``AWS session token``
    + ``AWS region``

    MPQP stores them in an AWS profile named ``mpqp-temporary``. Removing the
    Amazon Braket connection deletes this profile. When the credentials expire,
    obtain new credentials and configure this option again.

AWS CLI v2 is required only for configuring and renewing IAM Identity Center
profiles. The Python package does not install it. Follow the official `AWS CLI
v2 installation guide
<https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_
and select the instructions for your operating system:

- Linux, including x86-64 and ARM64
- macOS
- Windows

After installation, check that CLI v2 is available:

.. code-block:: console

    aws --version

The output should start with ``aws-cli/2.x.x``, for example
``aws-cli/2.22.12``.

The MPQP Docker image already includes CLI v2. If AWS CLI cannot open a
browser, for example inside Docker, add ``--use-device-code`` at the end of the
``aws configure sso`` or ``aws sso login`` command, for example:

.. code-block:: console

    $ aws sso login --profile my-aws-profile --use-device-code

AWS CLI will then display the instructions for signing in from a browser.

Quantinuum Nexus
^^^^^^^^^^^^^^^^

To use Quantinuum Nexus, you need a Nexus account with access to the required
projects and quotas. See the `Quantinuum Nexus authentication and quotas guide
<https://docs.quantinuum.com/nexus/trainings/notebooks/basics/auth_quotas.html>`_
for more information.

After installing the ``quantinuum`` extra, run the connection setup script:

.. code-block:: console

    $ setup_connections

Select ``Quantinuum Nexus``. MPQP calls ``qnx.login()`` to start the Nexus
browser authentication flow and then asks whether you want to use an existing
Nexus project or create a new one. If a browser cannot be opened, QNexus
displays a login link in the terminal instead.

QNexus stores and manages the authentication tokens. MPQP stores the selected
project name in its local configuration and activates that project when a
remote job is submitted. Running the setup again allows you to authenticate
again or select another project. More information about Nexus projects is
available in the `QNexus projects API reference
<https://docs.quantinuum.com/nexus/nexus_api/projects.html>`_.

MPQP provides local TKET simulators that do not require Nexus authentication.
The Nexus connection is required for the remote Nexus-hosted Aer and Qulacs
simulators, the noiseless H-Series ``QUANTINUUMDevice.H1_1LE`` and
``QUANTINUUMDevice.H2_1LE`` simulators, and the noisy H-Series
``QUANTINUUMDevice.H1_EMULATOR`` and ``QUANTINUUMDevice.H2_EMULATOR``
emulators.

Depending on the selected backend, MPQP supports sample, state-vector, and
observable jobs. Observable evaluation can be exact or shot-based. The
supported job types for each device are listed in the :ref:`Devices` section.
See the `Nexus backend configuration guide
<https://docs.quantinuum.com/nexus/trainings/notebooks/basics/alternate_backends/index.html>`_
for details about the corresponding Nexus backends.

For example, the following circuit is executed remotely on the Nexus-hosted
Aer simulator:

.. code-block:: python

    from mpqp import BasisMeasure, H, QCircuit, QUANTINUUMDevice, run

    circuit = QCircuit([H(0), BasisMeasure(shots=100)])
    result = run(circuit, QUANTINUUMDevice.NEXUS_AER_SIMULATOR)
    print(result)


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
