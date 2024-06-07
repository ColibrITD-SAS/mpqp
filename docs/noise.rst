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

.. code-block:: python
    :class: import

    from mpqp.noise import *


.. note::
    Noisy simulations are, for the moment, only supported for QLM and AWS 
    Braket devices.


Noise models
------------

In order to represent a general noise model, we introduce the abstract class
:class:`NoiseModel<mpqp.noise.noise_model.NoiseModel>`. It regroups all the
attributes and methods common to all predefined noise models.

While currently abstract, the class is designed with extensibility in mind to
support parameterized noise models in the future. This feature will enable users
to define noise models with adjustable parameters, offering greater flexibility
in simulating and analyzing the effects of noise on quantum circuits.


.. autoclass:: mpqp.noise.noise_model.NoiseModel

.. note::
    Only the predefined :class:`Depolarizing<mpqp.noise.noise_model.Depolarizing>` 
    noise model is available for the moment. More will come in the future.

Depolarizing Noise Model
-------------------------

.. autoclass:: mpqp.noise.noise_model.Depolarizing