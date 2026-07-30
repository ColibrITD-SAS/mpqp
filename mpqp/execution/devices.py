"""An :class:`AvailableDevice` is a device on which one can run or submit a
circuit. While it is an abstract class, all its concrete implementations are
enums with a few methods, required by :class:`AvailableDevice`.

Each supported provider has its available devices listed as these enums:

- :class:`IBMDevice`,
- :class:`ATOSDevice`,
- :class:`AWSDevice`,
- :class:`GOOGLEDevice`.
- :class:`AZUREDevice`.
- :class:`QUANTINUUMDevice`

Not all combinations of :class:`AvailableDevice` and
:class:`~mpqp.execution.job.JobType` are possible. Here is the list of
compatible jobs types and devices.

For more information about handling Remote devices, please refer to the `Remote devices handling <execution-extras.html>`_ section.

.. csv-table:: Job/Device Compatibility Matrix
   :file: ../../docs/resources/job-device_compat.csv
   :widths: 7, 25, 6, 7, 10, 10, 15
   :header-rows: 1
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from enum import Enum, auto

from mpqp.core.instruction.gates import Gate
from mpqp.core.instruction.gates.native_gates import *
from mpqp.environment.env_manager import get_env_variable


class AvailableDevice(Enum):
    """Class used to define a generic device (quantum computer or simulator)."""

    @abstractmethod
    def is_remote(self) -> bool:
        """Indicates whether a device is remote or not.

        Returns:
            ``True`` if this device is remote.
        """
        pass

    @abstractmethod
    def is_gate_based(self) -> bool:
        """Indicates whether a device is gate-based or not.

        Returns:
            ``True`` if this device is a gate-based simulator/QPU."""
        pass

    @abstractmethod
    def is_simulator(self) -> bool:
        """Indicates whether a device is a simulator or not.

        Returns:
            ``True`` if this device is a simulator."""
        pass

    @abstractmethod
    def is_noisy_simulator(self) -> bool:
        """Indicates whether a device can simulate noise or not.

        Returns:
            ``True`` if this device can simulate noise.
        """
        pass

    def has_reduced_gate_set(self) -> bool:
        """Indicates whether a simulator does not handle all the native gates.

        Returns:
            ``True`` if this device only handles a restricted set of gates."""
        return False

    @abstractmethod
    def supports_samples(self) -> bool:
        pass

    @abstractmethod
    def supports_state_vector(self) -> bool:
        pass

    @abstractmethod
    def supports_observable(self) -> bool:
        pass

    @abstractmethod
    def supports_observable_ideal(self) -> bool:
        pass

    def compatible_gates(self, native_set: bool = False) -> set[type[Gate]]:
        """Returns the set of gates supported by the devices.

        Args:
            native_set: If True returns the set of gates of the device without any transpilation.
                (For example: optimization_level=0 on Qiskit or Verbatim box on Braket)
        """
        return set()


class IBMDevice(AvailableDevice):
    """Enum regrouping all available devices provided by IBM Quantum.

    Warning:
        Since previous versions, many devices have been disabled by IBM. This may
        affect your code. We are currently investigating this issue to check if a
        workaround is possible for some of them (like replacing a simulator by
        an equivalent one for instance).
    """

    AER_SIMULATOR = "automatic"
    AER_SIMULATOR_STATEVECTOR = "statevector"
    AER_SIMULATOR_DENSITY_MATRIX = "density_matrix"
    AER_SIMULATOR_STABILIZER = "stabilizer"
    AER_SIMULATOR_EXTENDED_STABILIZER = "extended_stabilizer"
    AER_SIMULATOR_MATRIX_PRODUCT_STATE = "matrix_product_state"

    IBM_RENSSELAER = "ibm_rensselaer"
    IBM_KAWASAKI = "ibm_kawasaki"
    IBM_QUEBEC = "ibm_quebec"

    # NightHawk chips
    IBM_BOSTON = "ibm_boston"
    IBM_KINGSTON = "ibm_kingston"
    IBM_PITTSBURGH = "ibm_pittsburgh"
    IBM_FEZ = "ibm_fez"
    IBM_MARRAKESH = "ibm_marrakesh"
    IBM_AACHEN = "IBM_AACHEN"

    # Heron chips
    IBM_MIAMI = "ibm_miami"
    IBM_BERLIN = "ibm_berlin"

    IBM_CLEVELAND = "ibm_cleveland"
    IBM_PEEKSKILL = "ibm_peekskill"

    IBM_LEAST_BUSY = "ibm_least_busy"

    def is_remote(self) -> bool:
        return self.name.startswith("IBM")

    def is_gate_based(self) -> bool:
        return True

    def has_reduced_gate_set(self) -> bool:
        return self in {
            IBMDevice.AER_SIMULATOR_STABILIZER,
            IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER,
        }

    def is_simulator(self) -> bool:
        return self in {
            IBMDevice.AER_SIMULATOR,
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            IBMDevice.AER_SIMULATOR_DENSITY_MATRIX,
            IBMDevice.AER_SIMULATOR_STABILIZER,
            IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER,
            IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
        }

    def is_noisy_simulator(self) -> bool:
        return self.is_simulator()

    def supports_samples(self) -> bool:
        return True

    def supports_state_vector(self) -> bool:
        return self in {
            IBMDevice.AER_SIMULATOR,
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
        }

    def supports_observable(self) -> bool:
        return self not in {
            IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER,
        }

    def supports_observable_ideal(self) -> bool:
        return self in {
            IBMDevice.AER_SIMULATOR,
            IBMDevice.AER_SIMULATOR_STATEVECTOR,
            IBMDevice.AER_SIMULATOR_DENSITY_MATRIX,
            IBMDevice.AER_SIMULATOR_STABILIZER,
            IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
        }

    def compatible_gates(self, native_set: bool = False) -> set[type[Gate]]:
        if self == IBMDevice.AER_SIMULATOR_STABILIZER:
            warnings.warn(
                UserWarning(
                    f"For {self} the gates Rx, Ry and Rz are allowed but only at angles 0, π, π/2 and 3*π/2"
                )
            )
            return {Rx, Ry, Rz, X, Y, Z, H, CNOT, CZ, S, S_dagger, SWAP}
        elif self == IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER:
            warnings.warn(
                UserWarning(
                    f"For {self} the gates Rx, Ry and Rz are allowed but only at angles 0, π, π/2 and 3*π/2"
                )
            )
            return {Rx, Ry, Rz, X, Y, Z, H, CNOT, CZ, S, S_dagger, SWAP}
        else:
            compatibilities: dict[IBMDeviceFamily, set[type[Gate]]] = {
                IBMDeviceFamily.HERON: {CZ, Id, Rx, Rz, X},  # add Rzz
                IBMDeviceFamily.NIGHTHAWK: {CZ, Id, Rx, Rz, X},
            }
            family = {
                IBMDevice.IBM_MIAMI: IBMDeviceFamily.HERON,
                IBMDevice.IBM_BERLIN: IBMDeviceFamily.HERON,
                IBMDevice.IBM_BOSTON: IBMDeviceFamily.NIGHTHAWK,
                IBMDevice.IBM_KINGSTON: IBMDeviceFamily.NIGHTHAWK,
                IBMDevice.IBM_PITTSBURGH: IBMDeviceFamily.NIGHTHAWK,
                IBMDevice.IBM_FEZ: IBMDeviceFamily.NIGHTHAWK,
                IBMDevice.IBM_MARRAKESH: IBMDeviceFamily.NIGHTHAWK,
                IBMDevice.IBM_AACHEN: IBMDeviceFamily.NIGHTHAWK,
            }
            if self in family and family[self] in compatibilities:
                return compatibilities[family[self]]
            else:
                return set()


class IBMDeviceFamily(Enum):
    """Enum regrouping all device families defined by IBM."""

    HERON = auto()
    NIGHTHAWK = auto()


class ATOSDevice(AvailableDevice):
    """Enum regrouping all available devices provided by ATOS."""

    MYQLM_PYLINALG = auto()
    MYQLM_CLINALG = auto()

    QLM_LINALG = auto()
    QLM_MPS = auto()
    QLM_MPO = auto()
    QLM_NOISYQPROC = auto()

    def is_remote(self):
        return self.name.startswith("QLM")

    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return True

    def is_noisy_simulator(self) -> bool:
        return self in {ATOSDevice.QLM_NOISYQPROC, ATOSDevice.QLM_MPO}

    @staticmethod
    def from_str_remote(name: str):
        """Returns the first remote ATOSDevice matching the given name.

        Args:
            name: A substring of the desired device's name.

        Raises:
            ValueError: If no device corresponding to the given name could be
                found.

        Examples:
            >>> ATOSDevice.from_str_remote('NoisyQProc')
            <ATOSDevice.QLM_NOISYQPROC: 6>
            >>> ATOSDevice.from_str_remote('linalg')
            <ATOSDevice.QLM_LINALG: 3>
            >>> ATOSDevice.from_str_remote('Mps')
            <ATOSDevice.QLM_MPS: 4>

        """
        u_name = name.upper()
        for elem in ATOSDevice:
            if u_name in elem.name and elem.is_remote():
                return elem
        raise ValueError(f"No device found for name `{name}`.")

    def supports_samples(self) -> bool:
        return True

    def supports_state_vector(self) -> bool:
        return self in {
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            ATOSDevice.QLM_LINALG,
            ATOSDevice.QLM_MPS,
        }

    def supports_observable(self) -> bool:
        return self in {
            ATOSDevice.MYQLM_PYLINALG,
            ATOSDevice.MYQLM_CLINALG,
            ATOSDevice.QLM_LINALG,
        }

    def supports_observable_ideal(self) -> bool:
        return True


class AWSDevice(AvailableDevice):
    """Enum regrouping all available devices provided by AWS."""

    BRAKET_LOCAL_SIMULATOR = "LocalSimulator"

    BRAKET_SV1_SIMULATOR = "quantum-simulator/amazon/sv1"
    BRAKET_DM1_SIMULATOR = "quantum-simulator/amazon/dm1"
    BRAKET_TN1_SIMULATOR = "quantum-simulator/amazon/tn1"

    IONQ_ARIA_1 = "qpu/ionq/Aria-1"
    IONQ_ARIA_2 = "qpu/ionq/Aria-2"
    IONQ_FORTE_1 = "qpu/ionq/Forte-1"
    IONQ_FORTE_ENTERPRISE_1 = "qpu/ionq/Forte-Enterprise-1"
    QUERA_AQUILA = "qpu/quera/Aquila"
    RIGETTI_ANKAA_3 = "qpu/rigetti/Ankaa-3"
    IQM_GARNET = "qpu/iqm/Garnet"
    IQM_EMERALD = "qpu/iqm/Emerald"

    def is_remote(self):
        return self != AWSDevice.BRAKET_LOCAL_SIMULATOR

    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return "SIMULATOR" in self.name

    def is_noisy_simulator(self) -> bool:
        return self in [
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            AWSDevice.BRAKET_DM1_SIMULATOR,
        ]

    def get_arn(self) -> str:
        """Retrieve the AWSDevice arn from this AWSDevice element.

        Returns:
            The arn of the device.

        Examples:
            >>> AWSDevice.IONQ_ARIA_1.get_arn()
            'arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1'
            >>> AWSDevice.BRAKET_SV1_SIMULATOR.get_arn()
            'arn:aws:braket:::device/quantum-simulator/amazon/sv1'
            >>> AWSDevice.RIGETTI_ANKAA_3.get_arn()
            'arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3'

        """
        region = self.get_region()
        if self.is_simulator():
            region = ""
        return "arn:aws:braket:" + region + "::device/" + self.value

    def get_region(self) -> str:
        """Retrieve the AWS region from this AWSDevice element.

        Returns:
            The region of the device.

        Raises:
            ValueError: If called on a local (non-remote) simulator that has no AWS region.

        Examples:
            >>> AWSDevice.IONQ_ARIA_1.get_region()
            'us-east-1'
            >>> AWSDevice.BRAKET_SV1_SIMULATOR.get_region() == get_env_variable("AWS_DEFAULT_REGION")
            True
            >>> AWSDevice.RIGETTI_ANKAA_3.get_region()
            'us-west-1'

        """
        if not self.is_remote():
            raise ValueError(
                "Cannot retrieve AWS region for non-remote device (local simulator)"
            )
        elif self == AWSDevice.RIGETTI_ANKAA_3:
            return "us-west-1"

        elif self in [AWSDevice.IQM_GARNET, AWSDevice.IQM_EMERALD]:
            return "eu-north-1"
        elif self in [
            AWSDevice.IONQ_ARIA_1,
            AWSDevice.IONQ_ARIA_2,
            AWSDevice.IONQ_FORTE_1,
            AWSDevice.IONQ_FORTE_ENTERPRISE_1,
            AWSDevice.QUERA_AQUILA,
        ]:
            return "us-east-1"
        else:
            return get_env_variable("AWS_DEFAULT_REGION")

    @staticmethod
    def from_arn(arn: str):
        """Returns the right AWSDevice from the arn given in parameter.

        Args:
            arn: The AWS arn identifying the AWSDevice.

        Examples:
            >>> AWSDevice.from_arn('arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1')
            <AWSDevice.IONQ_ARIA_1: 'qpu/ionq/Aria-1'>
            >>> AWSDevice.from_arn('arn:aws:braket:::device/quantum-simulator/amazon/sv1')
            <AWSDevice.BRAKET_SV1_SIMULATOR: 'quantum-simulator/amazon/sv1'>

        """
        for elem in AWSDevice:
            if elem.value in arn:
                return elem
        raise ValueError(f"No device found for ARN `{arn}`.")

    def supports_samples(self) -> bool:
        return True

    def supports_state_vector(self) -> bool:
        return self in {
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            AWSDevice.BRAKET_SV1_SIMULATOR,
        }

    def supports_observable(self) -> bool:
        return True

    def supports_observable_ideal(self) -> bool:
        return self in {
            AWSDevice.BRAKET_LOCAL_SIMULATOR,
            AWSDevice.BRAKET_TN1_SIMULATOR,
        }


class GOOGLEDevice(AvailableDevice):
    """Enum regrouping all available devices provided by Google."""

    CIRQ_LOCAL_SIMULATOR = "LocalSimulator"
    PROCESSOR_RAINBOW = "rainbow"
    PROCESSOR_WEBER = "weber"
    IONQ_SIMULATOR = "simulator"
    IONQ_QPU = "qpu"

    def is_remote(self):
        if self.name.startswith("IONQ"):
            return True
        return False

    def is_ionq(self):
        """``True`` if the device is from ``IonQ``."""
        return self.name.startswith("IONQ")

    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return "SIMULATOR" in self.name

    def is_processor(self) -> bool:
        """
        Check if the device is a processor.

        Returns:
            True if the device is a processor, False otherwise.
        """
        return self.name.startswith("PROCESSOR")

    def is_noisy_simulator(self) -> bool:
        return self in [
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
            GOOGLEDevice.IONQ_SIMULATOR,
        ]

    def has_reduced_gate_set(self) -> bool:
        return self in {
            GOOGLEDevice.PROCESSOR_RAINBOW,
            GOOGLEDevice.PROCESSOR_WEBER,
            GOOGLEDevice.IONQ_SIMULATOR,
            GOOGLEDevice.IONQ_QPU,
        }

    def supports_samples(self) -> bool:
        return True

    def supports_state_vector(self) -> bool:
        return self in {
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        }

    def supports_observable(self) -> bool:
        return self in {
            GOOGLEDevice.PROCESSOR_RAINBOW,
            GOOGLEDevice.PROCESSOR_WEBER,
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        }

    def supports_observable_ideal(self) -> bool:
        return self in {
            GOOGLEDevice.PROCESSOR_RAINBOW,
            GOOGLEDevice.PROCESSOR_WEBER,
            GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        }


class AZUREDevice(AvailableDevice):
    """Enum regrouping all available devices provided by Azure."""

    IONQ_SIMULATOR = "ionq.simulator"
    IONQ_QPU = "ionq.qpu"
    IONQ_QPU_ARIA_1 = "ionq.qpu.aria-1"
    IONQ_QPU_ARIA_2 = "ionq.qpu.aria-2"

    QUANTINUUM_SIM_H1_1 = "quantinuum.qpu.h1-1"
    QUANTINUUM_SIM_H1_1SC = "quantinuum.sim.h1-1sc"
    QUANTINUUM_SIM_H1_1E = "quantinuum.sim.h1-1e"

    RIGETTI_SIM_QVM = "rigetti.sim.qvm"
    RIGETTI_SIM_QPU_ANKAA_3 = "rigetti.qpu.ankaa-3"

    def is_remote(self):
        return True

    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return self == AZUREDevice.IONQ_SIMULATOR

    def is_noisy_simulator(self) -> bool:
        raise NotImplementedError(
            'Noisy simulations are not yet implemented for Azure.'
        )

    def is_ionq(self):
        return self.name.startswith("IONQ")

    def supports_samples(self) -> bool:
        return True

    def supports_state_vector(self) -> bool:
        return False

    def supports_observable(self) -> bool:
        return False

    def supports_observable_ideal(self) -> bool:
        return False


class QUANTINUUMDevice(AvailableDevice):
    """Enum regrouping all available devices provided by Quantinuum."""

    TKET_AER_SIMULATOR = "tket-aer"
    TKET_AER_STATE_SIMULATOR = "tket-aer-state"

    NEXUS_AER_SIMULATOR = "aer"
    NEXUS_AER_STATE_SIMULATOR = "aer-state"
    NEXUS_QULACS_SIMULATOR = "qulacs"

    H1_1LE = "H1-1LE"
    H2_1LE = "H2-1LE"
    H1_EMULATOR = "H1-Emulator"
    H2_EMULATOR = "H2-Emulator"

    def is_remote(self) -> bool:
        return self not in {
            QUANTINUUMDevice.TKET_AER_SIMULATOR,
            QUANTINUUMDevice.TKET_AER_STATE_SIMULATOR,
        }

    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return True

    def is_noisy_simulator(self) -> bool:
        return self in {
            QUANTINUUMDevice.H1_EMULATOR,
            QUANTINUUMDevice.H2_EMULATOR,
        }

    def supports_samples(self) -> bool:
        return self not in {
            QUANTINUUMDevice.NEXUS_AER_STATE_SIMULATOR,
            QUANTINUUMDevice.TKET_AER_STATE_SIMULATOR,
        }

    def supports_state_vector(self) -> bool:
        return self in {
            QUANTINUUMDevice.NEXUS_AER_STATE_SIMULATOR,
            QUANTINUUMDevice.NEXUS_QULACS_SIMULATOR,
            QUANTINUUMDevice.TKET_AER_STATE_SIMULATOR,
        }

    def supports_observable(self) -> bool:
        return True

    def supports_observable_ideal(self) -> bool:
        return self in {
            QUANTINUUMDevice.NEXUS_AER_STATE_SIMULATOR,
            QUANTINUUMDevice.NEXUS_QULACS_SIMULATOR,
            QUANTINUUMDevice.TKET_AER_STATE_SIMULATOR,
        }
