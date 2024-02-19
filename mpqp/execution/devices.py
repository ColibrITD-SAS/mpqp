from enum import Enum, auto
from abc import abstractmethod

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


class IBMDevice(AvailableDevice):
    """Enum regrouping all available devices provided by IBM Quantum."""

    PULSE_SIMULATOR = "pulse_simulator"

    AER_SIMULATOR = "aer_simulator"
    AER_SIMULATOR_STATEVECTOR = "aer_simulator_statevector"
    AER_SIMULATOR_DENSITY_MATRIX = "aer_simulator_density_matrix"
    AER_SIMULATOR_STABILIZER = "aer_simulator_stabilizer"
    AER_SIMULATOR_MATRIX_PRODUCT_STATE = "aer_simulator_matrix_product_state"
    AER_SIMULATOR_EXTENDED_STABILIZER = "aer_simulator_extended_stabilizer"
    AER_SIMULATOR_UNITARY = "aer_simulator_unitary"
    AER_SIMULATOR_SUPEROP = "aer_simulator_superop"

    IBMQ_SIMULATOR_STATEVECTOR = "simulator_statevector"
    IBMQ_SIMULATOR_STABILIZER = "simulator_stabilizer"
    IBMQ_SIMULATOR_EXTENDED_STABILIZER = "simulator_extended_stabilizer"
    IBMQ_SIMULATOR_MPS = "simulator_mps"
    IBMQ_QASM_SIMULATOR = "ibmq_qasm_simulator"

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
        return self != IBMDevice.PULSE_SIMULATOR

    def is_simulator(self) -> bool:
        return "simulator" in self.value


class ATOSDevice(AvailableDevice):
    """Enum regrouping all available devices provided by ATOS."""

    MYQLM_PYLINALG = auto()
    MYQLM_CLINALG = auto()

    QLM_LINALG = auto()
    QLM_MPS = auto()
    QLM_MPS_LEGACY = auto()
    QLM_MPO = auto()
    QLM_STABS = auto()
    QLM_FEYNMAN = auto()
    QLM_BDD = auto()
    QLM_NOISY_QPROC = auto()
    QLM_SQA = auto()
    QLM_QPEG = auto()
    QLM_CLASSICAL_QPU = auto()

    def is_remote(self):
        return self.name.startswith("QLM")

    def is_gate_based(self) -> bool:
        return True

    def is_simulator(self) -> bool:
        return True


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

    def get_arn(self) -> str:
        """
        Retrieve the AwsDevice arn from this AWSDevice element.

        Examples:
            >>> AWSDevice.BRAKET_IONQ_HARMONY.get_arn()
            'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony'
            >>> AWSDevice.BRAKET_SV1_SIMULATOR.get_arn()
            'arn:aws:braket:::device/quantum-simulator/amazon/sv1'
            >>> AWSDevice.BRAKET_RIGETTI_ASPEN_M_3.get_arn()
            'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3'

        Returns:
            a string representing the arn of the device
        """
        if not self.is_remote():
            raise ValueError("No arn for a local simulator")
        if self.is_simulator():
            region = ""
        elif self == AWSDevice.BRAKET_RIGETTI_ASPEN_M_3:
            region = "us-west-1"
        elif self == AWSDevice.BRAKET_OQC_LUCY:
            region = "eu-west-2"
        elif self in [
            AWSDevice.BRAKET_IONQ_HARMONY,
            AWSDevice.BRAKET_IONQ_ARIA_1,
            AWSDevice.BRAKET_IONQ_ARIA_2,
            AWSDevice.BRAKET_IONQ_FORTE_1,
            AWSDevice.BRAKET_QUERA_AQUILA,
        ]:
            region = "us-east-1"
        else:
            region = get_env_variable("AWS_DEFAULT_REGION")
        return "arn:aws:braket:" + region + "::device/" + self.value

    @staticmethod
    def from_arn(arn: str):
        """
        Returns the right AWSDevice from the arn given in parameter.

        Examples:
            >>> AWSDevice.from_arn('arn:aws:braket:us-east-1::device/qpu/ionq/Harmony')
            <AWSDevice.BRAKET_IONQ_HARMONY: 'qpu/ionq/Harmony'>
            >>> AWSDevice.from_arn('arn:aws:braket:::device/quantum-simulator/amazon/sv1')
            <AWSDevice.BRAKET_SV1_SIMULATOR: 'quantum-simulator/amazon/sv1'>

        Args:
            arn: The AWS arn identifying the AwsDevice.

        """
        for elem in AWSDevice:
            if elem.value in arn:
                return elem
        return None
