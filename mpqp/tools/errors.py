"""You will find here the custom exceptions we created in order to provide
clearer errors. When relevant, we also append the trace of the error raised by a
provider's SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpqp.execution.result import JobType
    from mpqp.core.instruction.gates.gate import Gate


class InstructionParsingError(ValueError):
    """Raised when an QASM instruction encountered by the parser is malformed."""


class InstructionAfterMeasurementError(ValueError):
    """Raised when one tries to add an instruction after a measurement in a
    circuit."""


class NumberQubitsError(ValueError):
    """Raised when the number of qubits defining an instruction, a gate, or a
    measurement, is not coherent with the related objects (circuit, matrix,
    observable, etc...)."""


class NumberQubitsWarning(UserWarning):
    """Raised when the number of qubits defining an instruction, a gate, or a
    measurement, is not coherent with the related objects (circuit, matrix,
    observable, etc...)."""


class ResultAttributeError(AttributeError):
    """Raised when one tries to access the attribute of the result that is
    incoherent with the associated job."""


class DeviceJobIncompatibleError(ValueError):
    """Raised when one tries to run a job with a JobType that is not suitable
    for the selected device (for example SAMPLE job on a statevector simulator)."""


class DeviceJobIncompatibleWarning(UserWarning):
    """A warning is issued when a job is not compatible with the selected device."""


class RemoteExecutionError(ConnectionError):
    """Raised when an error occurred during a remote connection, submission or
    execution."""


class IBMRemoteExecutionError(RemoteExecutionError):
    """Raised when an error occurred during the remote execution process of
    job(s) on an IBM device."""


class IBMNoiseModelGeneration(UserWarning):
    """Warning for potential compatibility issues with IBM noise model."""


class QLMRemoteExecutionError(RemoteExecutionError):
    """Raised when an error occurred during the remote execution process of
    job(s) on the remote QLM."""


class AWSBraketRemoteExecutionError(RemoteExecutionError):
    """Raised when an error occurred during the remote execution process of
    job(s) on the remote Amazon Braket."""


class UnsupportedBraketFeaturesWarning(UserWarning):
    """Warning for potential compatibility issues with Braket."""


class OpenQASMTranslationWarning(UserWarning):
    """Warning for potential translation error when exporting to OpenQASM."""


class AdditionalGateNoiseWarning(UserWarning):
    """Warning for additional noise on native gate used in the decomposition of
    noisy gate."""


class NonReversibleWarning(UserWarning):
    """Warning for nonreversible instruction used in inverse function."""


class UnsupportedGateError(ValueError):
    def __init__(
        self,
        gate: Gate,
        gate_set: set[type[Gate]],
        missing_gates: set[type[Gate]] | None = None,
    ):
        missing = missing_gates or {type(gate)}
        missing_names = ", ".join(sorted(g.__name__ for g in missing))
        available_names = ", ".join(sorted(g.__name__ for g in gate_set))

        super().__init__(
            f"{type(gate).__name__} cannot be represented with the target "
            f"gate set. Missing gates: {missing_names}. "
            f"Available gates: {available_names}."
        )


def result_error_message(type: JobType) -> str:
    """Function to give more precision upon errors when getting data from results."""
    from mpqp.execution.result import JobType

    if type == JobType.OBSERVABLE:
        msg = "Since your job is of type OBSERVABLE you have access to the following data:\n- expectation_values"
    elif type == JobType.SAMPLE:
        msg = "Since your job is of type SAMPLE you have access to the following data:\n-counts \n-probabilities"
    elif type == JobType.STATE_VECTOR:
        msg = "Since your job is of type STATE_VECTOR you have access to the following data:\n-counts \n-probabilities"
    return (
        msg
        + "\nNote: The type of the job in MPQP is dependant of the type of measurement done in the circuit."
    )
