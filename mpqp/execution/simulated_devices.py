"""To execute a circuit on a noisy simulator that reproduces the noise model of a machine,
one can use a :class:`SimulatedDevice`. Inheriting from :class:`~mpqp.execution.devices.AvailableDevice`, it is the
mother class of all noisy devices reproducing real hardware for several providers.

For the moment, only IBM simulated devices are available (so called `FakeBackend`), but the structure is ready to allow
other simulated devices (QLM has this feature for instance."""

from typing import TYPE_CHECKING, Any, Iterator, Optional

from mpqp.environment.typechecked import conditional_typechecked
from mpqp.execution import AvailableDevice

if TYPE_CHECKING:
    from qiskit_aer.backends.aer_simulator import AerSimulator
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2


class SimulatedDevice(AvailableDevice):
    """A class used to define simulators reproducing the noise of a real device.
    It implements the abstract methods of ``AvailableDevice``, and is used as a blueprint for all possible
    simulated devices (IBM, QLM, ...)."""

    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return True

    def is_noisy_simulator(self) -> bool:
        return True

    def is_remote(self) -> bool:
        return False


@conditional_typechecked
class StaticIBMSimulatedDevice(SimulatedDevice):
    """A class regrouping methods specific to an ``IBMSimulatedDevice``."""

    def supports_observable(self) -> bool:
        return True

    def supports_observable_ideal(self) -> bool:
        return True

    def supports_samples(self) -> bool:
        return False

    def supports_state_vector(self):
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
        from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

        fake_imports = fake_provider.__dict__
        return [
            (name, device)
            for name, device in fake_imports.items()
            if name.startswith("Fake")
            and not name.startswith(("FakeProvider", "FakeFractional"))
            and issubclass(device, FakeBackendV2)
            and "cairo" not in name.lower()
        ]


class _LazyIBMSimulatedDevice:
    _instance: Optional[type[StaticIBMSimulatedDevice]] = None

    @classmethod
    def _init(cls) -> None:
        if cls._instance is None:
            providers = StaticIBMSimulatedDevice.get_ibm_fake_providers()
            cls._instance = StaticIBMSimulatedDevice("IBMSimulatedDevice", providers)
            for name, _ in providers:
                setattr(cls, name, cls._instance[name])

    def __getattr__(self, name: str) -> Any:
        self._init()
        return getattr(self._instance, name)

    def __iter__(self) -> Iterator[Any]:
        self._init()
        assert self._instance is not None, "Instance not initialized"
        return iter(self._instance)

    def __getitem__(self, key: Any) -> Any:
        self._init()
        assert self._instance is not None, "Instance not initialized"
        return self._instance[key]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._init()
        assert self._instance is not None, "Instance not initialized"
        return self._instance(*args, **kwargs)


IBMSimulatedDevice = _LazyIBMSimulatedDevice()
"""Enum regrouping all so called IBM "fake devices" used to simulate noise of real hardware.

The members of this Enum are generated dynamically from ``qiskit_ibm_runtime.fake_provider``."""
