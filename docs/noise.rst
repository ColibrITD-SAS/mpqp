Noisy Simulations
=================

Quantum systems as subject to noise during computations, from application of gates, imperfection of the measurement or
an interaction with the environment. Being able to simulate and validate quantum algorithms in a noisy setup is crucial
in the direction of achieving any advantage on current NISQ machines. This section provides an overview on how to instantiate
noise models, how to attach them to a circuit, qubits and gates, and how to run the simulations.

.. code-block:: python

    from mpqp.noise import *


Noise models
------------

In order to represent a general noise model, we introduce the abstract class
:class:`NoiseModel<mpqp.noise.noise_model.NoiseModel>`. It regroups all the attributes and methods common to all
predefined noise models.



While currently abstract, the class is designed with extensibility in mind to support parameterized noise models in the future. 
This feature will enable users to define noise models with adjustable parameters, offering greater flexibility in simulating and analyzing 
the effects of noise on quantum circuits.

.. autoclass:: mpqp.noise.noise_model.NoiseModel

Depolarizing Noise Model
-------------------------

.. autoclass:: mpqp.noise.noise_model.Depolarizing

    .. note::
        The Depolarizing noise model is currently supported in AWS Braket.

