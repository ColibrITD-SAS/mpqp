"""To execute a circuit on a noisy simulator that reproduces the noise model of a machine,
one can use a :class:`SimulatedDevice`. Inheriting from :class:`~mpqp.execution.devices.AvailableDevice`
"""

# TODO: finish doc


from typing import Union, TYPE_CHECKING
from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit_aer.backends.aer_simulator import AerSimulator
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit.providers import Backend

from mpqp.execution import AvailableDevice, IBMDevice


class SimulatedDevice(AvailableDevice):
    """Abstract class used to define simulators reproducing the noise of a real device."""

    # TODO : comment
    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return True

    def is_noisy_simulator(self) -> bool:
        return True

    def is_remote(self) -> bool:
        return False


@typechecked
class AbstractIBMSimulatedDevice(SimulatedDevice):
    """Abstract class regrouping methods specific to an ``IBMSimulatedDevice``."""

    def supports_statevector(self):
        return True

    def to_noisy_simulator(self) -> "AerSimulator":
        """
        TODO comment
        Returns:

        """
        from qiskit_aer.backends.aer_simulator import AerSimulator
        from mpqp.execution.connection.ibm_connection import get_backend

        return AerSimulator.from_backend(get_backend(self))

    # @staticmethod
    # def get_ibm_fake_providers() -> (
    #     list[tuple[str, Union[type["Backend"], type["FakeBackendV2"]]]]
    # ):
    #     from qiskit_ibm_runtime import fake_provider
    #     from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
    #
    #     fake_imports = fake_provider.__dict__
    #     return [
    #         (p, fake_imports[p])
    #         for p in fake_imports.keys()
    #         if p.startswith("Fake")
    #         and issubclass(fake_imports[p], FakeBackendV2)
    #         and not p.startswith("FakeProvider")
    #     ]

    @staticmethod
    def fill_fake_backends() -> list[tuple[str, str]]:
        # TODO comment

        return [
            ("IBM_FAKE" + device.name[3:], device.value)
            for device in IBMDevice
            if device.name.startswith("IBM_") and device != IBMDevice.IBM_LEAST_BUSY
        ]


IBMSimulatedDevice = AbstractIBMSimulatedDevice(
    'IBMSimulatedDevice', AbstractIBMSimulatedDevice.fill_fake_backends()
)
"""Enum regrouping all so called IBM "fake devices" used to simulate noise of real hardware.

The members of this Enum are generated dynamically from ``IBMDevice``."""
# TODO finish doc
