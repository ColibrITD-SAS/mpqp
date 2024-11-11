"""To execute a circuit on a noisy simulator that reproduces the noise model of a machine,
one can use a :class:`SimulatedDevice`. Inheriting from :class:`~mpqp.execution.devices.AvailableDevice`
"""

# TODO: finish doc


from typing import TYPE_CHECKING, Union

from mpqp.execution import AvailableDevice
from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit_aer.backends.aer_simulator import AerSimulator
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit.providers import Backend


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
        return False

    def to_noisy_simulator(self) -> "AerSimulator":
        """Instantiates and returns an ``AerSimulator`` (from qiskit_aer) with the noise model corresponding
        to this IBM fake device."""
        from qiskit_aer.backends.aer_simulator import AerSimulator

        return AerSimulator.from_backend(self.value())

    def to_noise_model(self) -> "Qiskit_NoiseModel":
        """Generates the qiskit ``NoiseModel`` corresponding to this IBM fake device."""
        from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel

        return Qiskit_NoiseModel.from_backend(self.value())

    @staticmethod
    def get_ibm_fake_providers() -> list[tuple[str, type["FakeBackendV2"]]]:
        from qiskit_ibm_runtime import fake_provider

        fake_imports = fake_provider.__dict__
        return [
            (p, fake_imports[p])
            for p in fake_imports.keys()
            if p.startswith("Fake")
            and not p.startswith(("FakeProvider", "FakeFractional"))
        ]


IBMSimulatedDevice = AbstractIBMSimulatedDevice(
    'IBMSimulatedDevice', AbstractIBMSimulatedDevice.get_ibm_fake_providers()
)
"""Enum regrouping all so called IBM "fake devices" used to simulate noise of real hardware.

The members of this Enum are generated dynamically from ``qiskit_ibm_runtime.fake_provider``."""
