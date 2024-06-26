"""An :class:`AvailableDevice` is a device on which one can run or submit a 
circuit. While it is an abstract class, all it's concrete implementations are
enums with a few methods, required by :class:`AvailableDevice`.

Each supported provider has its available devices listed as these enums, which
you can find bellow:

- :class:`IBMDevice`,
- :class:`ATOSDevice`,
- :class:`AWSDevice`,
- :class:`GOOGLEDevice`.

Not all combinations of :class:`AvailableDevice` and 
:class:`JobType<mpqp.execution.job.JobType>` are possible. Here is the list of
compatible jobs types and devices.

.. csv-table:: Job/Device Compatibility Matrix
   :file: ../../docs/resources/job-device_compat.csv
   :widths: 7, 25, 7, 10, 10, 15
   :header-rows: 1
"""

from abc import abstractmethod
from enum import Enum, auto

from mpqp.execution.connection.env_manager import get_env_variable


class AvailableDevice(Enum):
    """Class used to define a generic device (quantum computer, or simulator)."""

    @abstractmethod
    def is_remote(self) -> bool:
        """Indicates whether a device is remote or not.

        Returns:
            ``True`` if this device is remote.
        """
        pass

    @abstractmethod
    def is_gate_based(self) -> bool:
        """Indicates whether a device is gate based or not.

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


class IBMDevice(AvailableDevice):
    """Enum regrouping all available devices provided by IBM Quantum.

    Warning:
        Since previous version, many devices were disabled by IBM. This may
        affect your code. We are currently investigating the issue to check if a
        workaround is possible for some of them (like replacing a simulator by
        an equivalent one for instance).
    """

    # PULSE_SIMULATOR = "pulse_simulator"

    AER_SIMULATOR = "aer_simulator"
    AER_SIMULATOR_STATEVECTOR = "aer_simulator_statevector"
    # TODO: many devices are no longer working, explore why
    # AER_SIMULATOR_DENSITY_MATRIX = "aer_simulator_density_matrix"
    # AER_SIMULATOR_STABILIZER = "aer_simulator_stabilizer"
    # AER_SIMULATOR_MATRIX_PRODUCT_STATE = "aer_simulator_matrix_product_state"
    # AER_SIMULATOR_EXTENDED_STABILIZER = "aer_simulator_extended_stabilizer"
    # AER_SIMULATOR_UNITARY = "aer_simulator_unitary"
    # AER_SIMULATOR_SUPEROP = "aer_simulator_superop"

    # IBMQ_SIMULATOR_STATEVECTOR = "simulator_statevector"
    # IBMQ_SIMULATOR_STABILIZER = "simulator_stabilizer"
    # IBMQ_SIMULATOR_EXTENDED_STABILIZER = "simulator_extended_stabilizer"
    # IBMQ_SIMULATOR_MPS = "simulator_mps"
    # IBMQ_QASM_SIMULATOR = "ibmq_qasm_simulator"

    IBM_BRISBANE = "ibm_brisbane"
    IBM_OSAKA = "ibm_osaka"
    IBM_KYOTO = "ibm_kyoto"

    IBM_SHERBROOKE = "ibm_sherbrooke"
    IBM_KYIV = "ibm_kyiv"
    IBM_NAZCA = "ibm_nazca"
    IBM_CUSCO = "ibm_cusco"
    IBM_ITHACA = "ibm_ithaca"
    IBM_TORINO = "ibm_torino"
    IBM_QUEBEC = "ibm_quebec"
    IBM_KAWASAKI = "ibm_kawasaki"
    IBM_CLEVELAND = "ibm_cleveland"
    IBM_CAIRO = "ibm_cairo"
    IBM_HANOI = "ibm_hanoi"
    IBM_ALGIERS = "ibm_algiers"
    IBM_KOLKATA = "ibm_kolkata"
    IBM_MUMBAI = "ibm_mumbai"
    IBM_PEEKSKILL = "ibm_peekskill"

    IBM_RANDOM_SMALL_DEVICE = "ibm_small_device"
    IBM_SMALL_DEVICES_LEAST_BUSY = "ibm_least_busy"

    def is_remote(self) -> bool:
        return self.name.startswith("IBM")

    def is_gate_based(self) -> bool:
        return True
        # return self != IBMDevice.PULSE_SIMULATOR

    def is_simulator(self) -> bool:
        return "simulator" in self.value

    def is_noisy_simulator(self) -> bool:
        raise NotImplementedError()
        # TODO: determine which devices can simulate noise or not for Qiskit remote, or local
        noise_support_devices = {
            IBMDevice.IBMQ_SIMULATOR_STATEVECTOR: True,
            IBMDevice.AER_SIMULATOR_STABILIZER: True,
            IBMDevice.IBMQ_SIMULATOR_EXTENDED_STABILIZER: False,
            IBMDevice.IBMQ_SIMULATOR_MPS: False,
            IBMDevice.IBMQ_QASM_SIMULATOR: True,
        }
        return self in noise_support_devices


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
        """Returns the first remote ATOSDevice containing the name given in parameter.

        Args:
            name: A string containing the name of the device.

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


class AWSDevice(AvailableDevice):
    """Enum regrouping all available devices provided by AWS Braket."""

    BRAKET_LOCAL_SIMULATOR = "LocalSimulator"

    BRAKET_SV1_SIMULATOR = "quantum-simulator/amazon/sv1"
    BRAKET_DM1_SIMULATOR = "quantum-simulator/amazon/dm1"
    BRAKET_TN1_SIMULATOR = "quantum-simulator/amazon/tn1"

    BRAKET_IONQ_HARMONY = "qpu/ionq/Harmony"
    BRAKET_IONQ_ARIA_1 = "qpu/ionq/Aria-1"
    BRAKET_IONQ_ARIA_2 = "qpu/ionq/Aria-2"
    BRAKET_IONQ_FORTE_1 = "qpu/ionq/Forte-1"
    BRAKET_OQC_LUCY = "qpu/oqc/Lucy"
    BRAKET_QUERA_AQUILA = "qpu/quera/Aquila"
    BRAKET_RIGETTI_ASPEN_M_3 = "qpu/rigetti/Aspen-M-3"

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
        """Retrieve the AwsDevice arn from this AWSDevice element.

        Returns:
            The arn of the device.

        Examples:
            >>> AWSDevice.BRAKET_IONQ_HARMONY.get_arn()
            'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony'
            >>> AWSDevice.BRAKET_SV1_SIMULATOR.get_arn()
            'arn:aws:braket:::device/quantum-simulator/amazon/sv1'
            >>> AWSDevice.BRAKET_RIGETTI_ASPEN_M_3.get_arn()
            'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3'

        """
        region = self.get_region()
        if self.is_simulator():
            region = ""
        return "arn:aws:braket:" + region + "::device/" + self.value

    def get_region(self) -> str:
        """Retrieve the Aws region from this AWSDevice element.

        Returns:
            The region of the device.

        Examples:
            >>> AWSDevice.BRAKET_IONQ_HARMONY.get_region()
            'us-east-1'
            >>> AWSDevice.BRAKET_SV1_SIMULATOR.get_region() == get_env_variable("AWS_DEFAULT_REGION")
            True
            >>> AWSDevice.BRAKET_RIGETTI_ASPEN_M_3.get_region()
            'us-west-1'

        """
        if not self.is_remote():
            raise ValueError("No arn for a local simulator")
        elif self == AWSDevice.BRAKET_RIGETTI_ASPEN_M_3:
            return "us-west-1"
        elif self == AWSDevice.BRAKET_OQC_LUCY:
            return "eu-west-2"
        elif self in [
            AWSDevice.BRAKET_IONQ_HARMONY,
            AWSDevice.BRAKET_IONQ_ARIA_1,
            AWSDevice.BRAKET_IONQ_ARIA_2,
            AWSDevice.BRAKET_IONQ_FORTE_1,
            AWSDevice.BRAKET_QUERA_AQUILA,
        ]:
            return "us-east-1"
        else:
            return get_env_variable("AWS_DEFAULT_REGION")

    @staticmethod
    def from_arn(arn: str):
        """Returns the right AWSDevice from the arn given in parameter.

        Args:
            arn: The AWS arn identifying the AwsDevice.

        Examples:
            >>> AWSDevice.from_arn('arn:aws:braket:us-east-1::device/qpu/ionq/Harmony')
            <AWSDevice.BRAKET_IONQ_HARMONY: 'qpu/ionq/Harmony'>
            >>> AWSDevice.from_arn('arn:aws:braket:::device/quantum-simulator/amazon/sv1')
            <AWSDevice.BRAKET_SV1_SIMULATOR: 'quantum-simulator/amazon/sv1'>

        """
        for elem in AWSDevice:
            if elem.value in arn:
                return elem
        raise ValueError(f"No device found for ARN `{arn}`.")


class GOOGLEDevice(AvailableDevice):
    """Enum regrouping all available devices provided by CIRQ."""

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
