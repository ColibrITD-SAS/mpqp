"""Translation helpers for circuit binding execution and their fallbacks.

:class:`~mpqp.core.binding.CircuitBinding` groups related runs so MPQP can
optimize their execution and package them into a single submission to the
appropriate providers. Beyond reducing submission overhead, compatible runs can
share hardware executions. For example, expectation values of different Pauli
strings for the same circuit can be estimated from shared measurement samples
when their measurement bases are compatible. Such reuse can reduce QPU execution
time and cost; the available optimizations depend on the provider and the
requested computation.

When a provider does not support this execution mode for the requested runs,
MPQP falls back to ordinary batches of individual executions. The helpers in
this module support that fallback: they recursively visit nested bindings and
collect provider translations separately for their circuits and measurements.

:func:`translate_binding_circuits` and :func:`translate_binding_measurements`
return dictionaries keyed by the original objects' :func:`id`. Each distinct
object is translated once per call, so repeated executions can reuse its
translation. These keys identify objects within the current process; they do not
encode execution order or distinguish repeated occurrences of the same object.
Execution paths use :meth:`~mpqp.core.binding.CircuitBinding.unroll` to
associate the translations with individual parameter sets and measurements.
"""

from typing import TYPE_CHECKING, Any, Optional, TypeVar

from mpqp.core.binding import (
    BindingExecution,
    BindingOptions,
    BindingParameters,
    CircuitBinding,
)
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.execution.devices import AvailableDevice, AWSDevice, IBMDevice

if TYPE_CHECKING:
    from qiskit_aer import AerSimulator


def __translate_binding_circuits(
    cb: CircuitBinding,
    device: AvailableDevice,
    result: dict[int, Any] = {},
    skip_pre_measure: bool = False,
    backend_sim: Optional["AerSimulator"] = None,
):
    """Recursive function to iterate every circuits in ``cb`` and store their translation in result."""
    for circuit in cb.circuits:
        if isinstance(circuit, CircuitBinding):
            __translate_binding_circuits(circuit, device, result)
        else:
            if result.get(id(circuit)) is not None:
                from warnings import warn

                warn(
                    "Several instances of the same circuit were found in the same CircuitBinding, it may lead to repeated executions."
                )

            else:
                if isinstance(device, AWSDevice):
                    if cb.noises:
                        translated_circuit = (
                            circuit + QCircuit(cb.noises)
                        ).to_other_device(device)
                    else:
                        translated_circuit = circuit.transpiled_for_device(device)
                    result.update({id(circuit): translated_circuit})
                elif isinstance(device, IBMDevice):
                    if cb.is_noisy:
                        from mpqp.execution.providers.ibm import (
                            generate_qiskit_noise_model,
                        )

                        original_noises = circuit.noises
                        circuit.noises = cb.noises if cb.noises is not None else []
                        nm, modified_circuit = generate_qiskit_noise_model(circuit)
                        circuit.noises = original_noises
                        if cb.transpiled_noise_model is None:
                            cb.transpiled_noise_model = (  # pyright: ignore[reportAttributeAccessIssue]
                                nm
                            )

                        translated_circuit = modified_circuit.to_other_device(
                            device,
                            skip_pre_measure=skip_pre_measure,
                            backend_sim=backend_sim,
                        )
                    else:
                        if device not in circuit.transpiled_circuit:
                            circuit.transpiled_circuit[device] = circuit.to_other_device(
                                device,
                                skip_pre_measure=skip_pre_measure,
                                backend_sim=backend_sim,
                            )
                        translated_circuit = circuit.transpiled_circuit[device]
                    result.update({id(circuit): translated_circuit})
                else:
                    raise NotImplementedError
    return result


def translate_binding_circuits(
    cb: CircuitBinding,
    device: AvailableDevice,
    skip_pre_measure: bool = False,
    backend_sim: Optional["AerSimulator"] = None,
) -> dict[int, Any]:
    """Returns a dictionary containing the transpiled circuits of a
    circuit binding with their memory address as keys.
    This functions' goal is to only translate circuits once and prevent altering the original object through transpiled_circuits.

    This is of use for runs not supported natively by the equivalent of circuit bindings.
    """
    return __translate_binding_circuits(cb, device, {}, skip_pre_measure, backend_sim)


def __translated_binding_measurements(
    cb: CircuitBinding, device: AvailableDevice, result: dict[int, Any]
) -> dict[int, Any]:
    for circuit in cb.circuits:
        if isinstance(circuit, CircuitBinding):
            __translated_binding_measurements(circuit, device, result)
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


def translate_binding_measurements(
    cb: CircuitBinding, device: AvailableDevice
) -> dict[int, Any]:
    """Returns a dictionary containing the transpiled measurements of a
    circuit binding with their memory address as keys.
    This functions' goal is to only translate measurements once and prevent altering the original object.

    This is of use for runs not supported natively by the equivalent of circuit bindings.
    """
    return __translated_binding_measurements(cb, device, {})


def merge_values(
    child_parameters: Optional[BindingParameters],
    parent_parameters: Optional[BindingParameters],
) -> Optional[BindingParameters]:
    """Merge child and parent values, giving precedence to the parent."""
    if child_parameters is None and parent_parameters is None:
        return None
    combined_parameters = dict(child_parameters) if child_parameters is not None else {}
    if parent_parameters is not None:
        combined_parameters.update(parent_parameters)
    return combined_parameters


def bind_same_parameters(
    child_parameters: Optional[BindingParameters],
    parent_parameters: Optional[BindingParameters],
) -> bool:
    """Return whether child and parent bind at least one common name."""
    if child_parameters is None or parent_parameters is None:
        return False
    child_names = {str(key) for key in child_parameters}
    parent_names = {str(key) for key in parent_parameters}
    return bool(child_names & parent_names)


def combine_zipped_options(
    child_parameters: Optional[BindingParameters],
    child_measurement: Optional[Measure],
    parent_parameters: Optional[BindingParameters],
    parent_measurement: Optional[Measure],
) -> list[BindingOptions]:
    """Combine one child option with one parent option in ZIP mode."""
    parameter_collision = bind_same_parameters(child_parameters, parent_parameters)
    measurement_collision = (
        child_measurement is not None and parent_measurement is not None
    )
    if not parameter_collision and not measurement_collision:
        return [
            (
                merge_values(child_parameters, parent_parameters),
                (
                    parent_measurement
                    if parent_measurement is not None
                    else child_measurement
                ),
            )
        ]
    if parameter_collision:
        if TYPE_CHECKING:
            assert parent_parameters is not None
            assert child_parameters is not None
        parent_execution_parameters = parent_parameters.copy()
        child_execution_parameters = child_parameters.copy()
    else:
        parent_execution_parameters = child_execution_parameters = merge_values(
            child_parameters, parent_parameters
        )

    return [
        (
            parent_execution_parameters,
            (
                parent_measurement
                if parent_measurement is not None
                else child_measurement
            ),
        ),
        (
            child_execution_parameters,
            (
                child_measurement
                if child_measurement is not None
                else parent_measurement
            ),
        ),
    ]


def parameter_set_key(
    parameter_set: Optional[BindingParameters],
) -> Optional[tuple[tuple[str, str]]]:
    """Build a stable, hashable key for a parameter mapping."""
    if parameter_set is None:
        return None
    return tuple(
        sorted((str(key), repr(value)) for key, value in parameter_set.items())
    )  # pyright: ignore[reportReturnType]


def execution_key(
    execution: BindingExecution,
) -> tuple[int, Optional[tuple[tuple[str, str]]], int]:
    """Build an identity key used to remove duplicate executions."""
    circuit, parameters, measurement = execution
    return id(circuit), parameter_set_key(parameters), id(measurement)


def product_values(
    child_parameters: Optional[BindingParameters],
    parent_values: list[Optional[BindingParameters]],
) -> list[Optional[BindingParameters]]:
    """Resolve and deduplicate parameter sets for PRODUCT mode."""
    unique_values: dict[
        Optional[tuple[tuple[str, str]]], Optional[BindingParameters]
    ] = {}
    keep_child_parameters = False

    for parent_parameters in parent_values:
        if child_parameters is None:
            parameter_set = (
                parent_parameters.copy() if parent_parameters is not None else None
            )
        elif parent_parameters is None:
            keep_child_parameters = True
            continue
        elif bind_same_parameters(child_parameters, parent_parameters):
            parameter_set = parent_parameters.copy()
            keep_child_parameters = True
        else:
            parameter_set = merge_values(child_parameters, parent_parameters)

        unique_values.setdefault(parameter_set_key(parameter_set), parameter_set)

    if keep_child_parameters:
        if TYPE_CHECKING:
            assert child_parameters is not None
        child_parameters = child_parameters.copy()
        unique_values.setdefault(parameter_set_key(child_parameters), child_parameters)

    return list(unique_values.values())


def product_measurements(
    child_measurement: Optional[Measure], parent_measurements: list[Measure | None]
) -> list[Optional[Measure]]:
    """Resolve and deduplicate measurements for PRODUCT mode."""
    unique_measurements: dict[int, Optional[Measure]] = {}
    keep_child_measurement = False

    for parent_measurement in parent_measurements:
        if child_measurement is None:
            unique_measurements.setdefault(id(parent_measurement), parent_measurement)
        elif parent_measurement is None:
            keep_child_measurement = True
        else:
            unique_measurements.setdefault(id(parent_measurement), parent_measurement)
            keep_child_measurement = True

    if keep_child_measurement:
        unique_measurements.setdefault(id(child_measurement), child_measurement)

    return list(unique_measurements.values())


"""Optional parameters and measurement combined while resolving a binding."""
_AxisElement = TypeVar("_AxisElement")


def expand_branch(
    branch: QCircuit | CircuitBinding,
) -> list[BindingExecution]:
    """Return the executions represented by a circuit or nested binding."""
    if isinstance(branch, CircuitBinding):
        return branch.unroll()
    return [(branch, None, None)]


def broadcast_axis(axis: list[_AxisElement], target_length: int) -> list[_AxisElement]:
    """Broadcast a singleton ZIP axis or validate its length."""
    if len(axis) == 1:
        return [axis[0]] * target_length
    if len(axis) != target_length:
        raise ValueError(
            f"In ZIP mode, lists must be length 1 or match the maximum list length ({target_length})."
        )
    return axis
