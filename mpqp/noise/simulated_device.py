from __future__ import annotations

from typing import TYPE_CHECKING, Union

from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit_aer.backends.aer_simulator import AerSimulator
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
    from qiskit.providers import Backend

from mpqp.execution import AvailableDevice


@typechecked
def get_ibm_fake_providers() -> (
    list[tuple[str, Union[type["Backend"], type["FakeBackendV2"]]]]
):
    from qiskit_ibm_runtime import fake_provider

    fake_imports = fake_provider.__dict__
    return [
        (p, fake_imports[p])
        for p in fake_imports.keys()
        if p.startswith("Fake") and not p.startswith("FakeProvider")
    ]


class SimulatedDevice(AvailableDevice):
    pass


class AbstractIBMSimulatedDevice(SimulatedDevice):

    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return True

    def is_noisy_simulator(self) -> bool:
        return True

    def is_remote(self) -> bool:
        return False

    def supports_statevector(self):
        return True

    def to_noisy_simulator(self) -> "AerSimulator":
        from qiskit_aer.backends.aer_simulator import AerSimulator

        return AerSimulator.from_backend(self.value())


IBMSimulatedDevice = AbstractIBMSimulatedDevice(
    'IBMSimulatedDevice', get_ibm_fake_providers()
)
