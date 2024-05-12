Noisy Simulation
================

.. code-block:: python

    from mpqp.noise.noise_model import *

Introduction
------------

This section provides an overview of noise models in the MPQP library, including their purpose and usage.

Noise Models
------------

This section describes different noise models available in MPQP.

The NoiseModel class in the MPQP library is an abstract foundation for representing various types of noise encountered in quantum circuits. 
It allows developers to define how different noise models are applied to specific qubits or the entire circuit. This flexible framework empowers 
the simulation and mitigation of noise effects, making the development of robust quantum algorithms and workflows possible.

While currently abstract, the class is designed with extensibility in mind to support parameterized noise models in the future. 
This feature will enable users to define noise models with adjustable parameters, offering greater flexibility in simulating and analyzing 
the effects of noise on quantum circuits.

.. autoclass:: mpqp.noise.noise_model.NoiseModel

Depolarizing Noise Model
-------------------------

.. autoclass:: mpqp.noise.noise_model.Depolarizing

    .. note::
        The Depolarizing noise model is currently supported in AWS Braket.

