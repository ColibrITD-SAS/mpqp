from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence

from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel

from mpqp.execution import IBMDevice
from mpqp.noise import NoiseModel


class IBMDeviceHasNoiseModel(IBMDevice):

    def has_noise_model(self):
        """Returns True if we can generate a noise model from the remote device"""
        pass

    def to_qiskit_noise_model(self) -> "Qiskit_NoiseModel":
        from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel

        return Qiskit_NoiseModel.from_backend(...)


class DeviceNoise(NoiseModel):
    pass