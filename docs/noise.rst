Noisy Simulations
=================

Quantum systems are subject to noise during computations, from application of
gates, imperfection of the measurement or an interaction with the environment.
Being able to simulate and validate quantum algorithms in a noisy setup is
crucial in the direction of achieving any advantage on current NISQ machines.
This section provides an overview on how to instantiate noise models, how to
specify target qubits and gates, and what are the predefined noise models.

We strongly encourage the user to have a look at the 
:doc:`dedicated notebook <notebooks/6_Noise_Simulation>`, where all the 
details about the manipulation, usage, and simulation of noise models are 
presented.

For more details on how to use noise models taken from real hardware, you can
look at :ref:`SimulatedDevices`.

.. code-block:: python
    :class: import

    from mpqp.noise import *


.. note::
    Noisy simulations are supported for :class:`IBMDevice`, :class:`AtosDevice` and :class:`AWSDevice`.


Noise models
------------

In order to represent a general noise model, we introduce the abstract class
:class:`~mpqp.noise.noise_model.NoiseModel`. It regroups all the
attributes and methods common to all predefined noise models.

While currently abstract, the class is designed with extensibility in mind to
support parameterized noise models in the future. This feature will enable users
to define noise models with adjustable parameters, offering greater flexibility
in simulating and analyzing the effects of noise on quantum circuits.


.. autoclass:: mpqp.noise.noise_model.NoiseModel

Depolarizing Noise Model
------------------------

.. autoclass:: mpqp.noise.noise_model.Depolarizing

BitFlip Noise Model
-------------------

.. autoclass:: mpqp.noise.noise_model.BitFlip

Amplitude Damping Noise Model
-----------------------------

.. autoclass:: mpqp.noise.noise_model.AmplitudeDamping

Phase Damping Noise Model
-----------------------------

.. autoclass:: mpqp.noise.noise_model.PhaseDamping