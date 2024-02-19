class InstructionParsingError(ValueError):
    """Raised when an QASM instruction encountered by the parser is malformed"""


class NumberQubitsError(ValueError):
    """Raised when the number of qubits defining an instruction, a gate, or a measurement, is not coherent with the
    related objets (circuit, matrix, observable, etc.)"""


class ResultAttributeError(AttributeError):
    """Raised when one tries to access the attribute of the result that is incoherent with the associated job."""


class DeviceJobIncompatibleError(ValueError):
    """Raised when one tries to run a job with a JobType that is not suitable for the selected device
    (for example SAMPLE job on a statevector simulator)"""


class RemoteExecutionError(ConnectionError):
    """Raised when an error occurred during a remote connection, submission or execution"""


class IBMRemoteExecutionError(RemoteExecutionError):
    """Raised when an error occurred during the remote execution process of job(s) on an IBM device"""


class QLMRemoteExecutionError(RemoteExecutionError):
    """Raised when an error occurred during the remote execution process of job(s) on the remote QLM"""


class AWSBraketRemoteExecutionError(RemoteExecutionError):
    """Raised when an error occurred during the remote execution process of job(s) on the remote Amazon Braket"""
