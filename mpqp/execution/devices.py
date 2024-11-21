"""An :class:`AvailableDevice` is a device on which one can run or submit a 
circuit. While it is an abstract class, all its concrete implementations are
enums with a few methods, required by :class:`AvailableDevice`.

Each supported provider has its available devices listed as these enums:

- :class:`IBMDevice`,
- :class:`ATOSDevice`,
- :class:`AWSDevice`,
- :class:`GOOGLEDevice`.
- :class:`AZUREDevice`.

Not all combinations of :class:`AvailableDevice` and 
:class:`~mpqp.execution.job.JobType` are possible. Here is the list of
compatible jobs types and devices.

For more information about handling Remote devices, please refer to the `Remote devices handling <execution-extras.html>`_ section.

.. csv-table:: Job/Device Compatibility Matrix
   :file: ../../docs/resources/job-device_compat.csv
   :widths: 7, 25, 6, 7, 10, 10, 15
   :header-rows: 1
"""

from abc import abstractmethod
from enum import Enum, auto

from mpqp.execution.connection.env_manager import get_env_variable


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

    IBM_SHERBROOKE = "ibm_sherbrooke"
    IBM_BRISBANE = "ibm_brisbane"
    IBM_KYIV = "ibm_kyiv"

    IBM_FEZ = "ibm_fez"
    IBM_RENSSELAER = "ibm_rensselaer"
    IBM_BRUSSELS = "ibm_brussels"
    IBM_KAWASAKI = "ibm_kawasaki"
    IBM_QUEBEC = "ibm_quebec"
    IBM_TORINO = "ibm_torino"
    IBM_NAZCA = "ibm_nazca"
    IBM_STRASBOURG = "ibm_strasbourg"

    # RETIRED - IBM_OSAKA = "ibm_osaka"
    # RETIRED - IBM_KYOTO = "ibm_kyoto"
    # RETIRED - IBM_CUSCO = "ibm_cusco"
    # RETIRED - IBM_ITHACA = "ibm_ithaca"
    IBM_CLEVELAND = "ibm_cleveland"
    # RETIRED - IBM_CAIRO = "ibm_cairo"
    # RETIRED - IBM_HANOI = "ibm_hanoi"
    # RETIRED - IBM_ALGIERS = "ibm_algiers"
    # RETIRED - IBM_KOLKATA = "ibm_kolkata"
    # RETIRED - IBM_MUMBAI = "ibm_mumbai"
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
            # IBMDevice.AER_SIMULATOR_EXTENDED_STABILIZER,
            IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
        }


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
    QUERA_AQUILA = "qpu/quera/Aquila"
    RIGETTI_ANKAA_2 = "qpu/rigetti/Ankaa-2"
    IQM_GARNET = "qpu/iqm/Garnet"

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
            >>> AWSDevice.RIGETTI_ANKAA_2.get_arn()
            'arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2'

        """
        region = self.get_region()
        if self.is_simulator():
            region = ""
        return "arn:aws:braket:" + region + "::device/" + self.value

    def get_region(self) -> str:
        """Retrieve the AWS region from this AWSDevice element.

        Returns:
            The region of the device.

        Examples:
            >>> AWSDevice.IONQ_ARIA_1.get_region()
            'us-east-1'
            >>> AWSDevice.BRAKET_SV1_SIMULATOR.get_region() == get_env_variable("AWS_DEFAULT_REGION")
            True
            >>> AWSDevice.RIGETTI_ANKAA_2.get_region()
            'us-west-1'

        """
        if not self.is_remote():
            raise ValueError("No arn for a local simulator")
        elif self == AWSDevice.RIGETTI_ANKAA_2:
            return "us-west-1"

        elif self == AWSDevice.IQM_GARNET:
            return "eu-north-1"
        elif self in [
            AWSDevice.IONQ_ARIA_1,
            AWSDevice.IONQ_ARIA_2,
            AWSDevice.IONQ_FORTE_1,
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
    RIGETTI_SIM_QPU_ANKAA_2 = "rigetti.qpu.ankaa-2"

    MICROSOFT_ESTIMATOR = "microsoft.estimator"

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
        return not self == AZUREDevice.MICROSOFT_ESTIMATOR

    def supports_state_vector(self) -> bool:
        return False

    def supports_observable(self) -> bool:
        return False

    def supports_observable_ideal(self) -> bool:
        return False
