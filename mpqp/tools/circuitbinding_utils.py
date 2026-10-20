from typing import Any

from mpqp.core.circuit import QCircuit
from mpqp.core.circuitbinding import CircuitBinding
from mpqp.execution.devices import AvailableDevice


def __translate_circuitbinding_circuits(
    cb: CircuitBinding, device: AvailableDevice, result: dict[int, Any] = {}
):
    """Recursive function to iterate every circuits in cb and store their translation in result."""
    for circuit in cb.circuits:
        if isinstance(circuit, CircuitBinding):
            __translate_circuitbinding_circuits(circuit, device, result)
        else:
            if result.get(id(circuit)) is not None:
                from warnings import warn

                warn(
                    "Several instances of the same circuit were found in the same CircuitBinding, it may lead to repeated executions."
                )
            else:
                result.update(
                    {
                        id(circuit): (circuit + QCircuit(cb.noises)).to_other_device(
                            device
                        )
                    }
                )
    return result


def translate_circuitbinding_circuits(
    cb: CircuitBinding, device: AvailableDevice
) -> dict[int, Any]:
    """Returns a dictionary containing the transpiled circuits of a
    circuitbinding with their memory address as keys.
    This functions' goal is to only translate circuits once and prevent altering the original object through transpiled_circuits.

    This is of use for runs not supported natively by the equivalent of circuit bindings.
    """
    return __translate_circuitbinding_circuits(cb, device, {})


def __translated_circuitbinding_measurements(
    cb: CircuitBinding, device: AvailableDevice, result: dict[int, Any]
) -> dict[int, Any]:
    for circuit in cb.circuits:
        if isinstance(circuit, CircuitBinding):
            __translated_circuitbinding_measurements(circuit, device, result)
    if cb.measurements is None:
        return result
    for measurement in cb.measurements:
        if result.get(id(measurement)) is not None:
            from warnings import warn

            warn(
                "Several instances of the same measurement were found in the same CircuitBinding, it may lead to repeated executions."
            )
        else:
            result.update(
                {id(measurement): QCircuit([measurement]).to_other_device(device)}
            )
    return result


def translate_circuitbinding_measurements(
    cb: CircuitBinding, device: AvailableDevice
) -> dict[int, Any]:
    """Returns a dictionary containing the transpiled measurements of a
    circuitbinding with their memory address as keys.
    This functions' goal is to only translate measurements once and prevent altering the original object.

    This is of use for runs not supported natively by the equivalent of circuit bindings.
    """
    return __translated_circuitbinding_measurements(cb, device, {})
