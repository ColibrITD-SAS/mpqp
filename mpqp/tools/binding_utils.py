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

:func:`create_binding_execution_circuit` prepares the MPQP circuit associated
with one resolved execution. It removes measurements already present in the
circuit, clears cached provider translations, and adds a copy of the measurement
selected by the binding. Copying the measurement keeps the original binding
unchanged and allows the same measurement to receive the correct targets for
each circuit size.

:func:`translate_binding_circuits` stores each translated circuit under the
:func:`id` of its original circuit. :func:`translate_binding_measurements` uses
both the measurement's :func:`id` and the number of qubits in the circuit where
it is used because a measurement with no targets applies to all qubits. Each
distinct input is translated once per call, so repeated executions can reuse
its translation. These keys identify objects within the current process, they
do not encode execution order.

Execution paths use :meth:`~mpqp.core.binding.CircuitBinding.unroll` to
associate the translations with individual parameter sets and measurements.
"""

from typing import TYPE_CHECKING, Any, Optional, TypeVar

from mpqp.core._types import BindingParameters
from mpqp.core.binding import (
    BindingExecution,
    BindingOptions,
    CircuitBinding,
)
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.execution.devices import AvailableDevice, AWSDevice, IBMDevice

if TYPE_CHECKING:
    from qiskit_aer import AerSimulator


def create_binding_execution_circuit(
    circuit: QCircuit, measurement: Optional[Measure]
) -> QCircuit:
    """Prepare the MPQP circuit used for one binding execution.

    The original measurements are removed, and a copy of the measurement
    selected by the binding is added. Copying it prevents MPQP from changing
    the original measurement when it fills in the targets for the current
    circuit. Cached provider translations are also cleared so this execution
    is translated from the prepared circuit.

    Args:
        circuit: Circuit selected for the execution.
        measurement: Measurement selected for the execution, if any.

    Returns:
        A circuit containing the selected measurement and no cached provider
        translation.
    """
    from copy import deepcopy

    execution_circuit = circuit.without_measurements(deep_copy=False)
    execution_circuit.transpiled_circuit = {}
    execution_circuit.transpiled_noise_model = None
    if measurement is not None:
        execution_circuit.add(deepcopy(measurement))
    return execution_circuit


def __translate_binding_circuits(
    cb: CircuitBinding,
    device: AvailableDevice,
    result: dict[int, Any] = {},
    skip_pre_measure: bool = False,
    backend_sim: Optional["AerSimulator"] = None,
):
    """Recursively translate each circuit in ``cb`` and store its translation in ``result``."""
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
                            circuit.transpiled_circuit[device] = (
                                circuit.to_other_device(
                                    device,
                                    skip_pre_measure=skip_pre_measure,
                                    backend_sim=backend_sim,
                                )
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
    """Translate the circuits contained in a circuit binding.

    This function translates each circuit only once and returns a dictionary
    containing the translated circuits, with their object IDs as keys. This
    prevents repeated translations without storing them in the original
    circuits through ``transpiled_circuit``.

    This is useful for execution paths that do not support circuit bindings
    natively.

    Args:
        cb: Circuit binding containing the circuits to translate.
        device: Device for which the circuits are translated.
        skip_pre_measure: If ``True``, translate without pre-measurement
            operations.
        backend_sim: Optional Aer simulator used during IBM translation.

    Returns:
        A dictionary mapping each original circuit's :func:`id` to its provider
        translation.
    """
    return __translate_binding_circuits(cb, device, {}, skip_pre_measure, backend_sim)


def translate_binding_measurements(
    cb: CircuitBinding, device: AvailableDevice
) -> dict[tuple[int, int], Any]:
    """Translate each binding measurement for the circuit where it is used.

    A measurement with no targets applies to all qubits. The same measurement
    must therefore be translated separately when it is used with circuits of
    different sizes. The original measurement is not modified.

    Args:
        cb: Circuit binding containing the measurements.
        device: Device for which the measurements are translated.

    Returns:
        A dictionary mapping each measurement's :func:`id` and the circuit's
        number of qubits to its provider translation.
    """
    translated: dict[tuple[int, int], Any] = {}
    for circuit, _, measurement in cb.unroll():
        if measurement is None:
            continue
        key = (id(measurement), circuit.nb_qubits)
        if key in translated:
            continue
        measurement_circuit = create_binding_execution_circuit(
            QCircuit(circuit.nb_qubits), measurement
        )
        translated[key] = measurement_circuit.to_other_device(device)
    return translated


def merge_values(
    child_parameters: Optional[BindingParameters],
    parent_parameters: Optional[BindingParameters],
) -> Optional[BindingParameters]:
    """Merge child and parent values, giving precedence to the parent."""
    if child_parameters is None and parent_parameters is None:
        return None
    combined_parameters = (
        {key: value for key, value in child_parameters.items()}
        if child_parameters is not None
        else {}
    )
    if parent_parameters is not None:
        for key, value in parent_parameters.items():
            combined_parameters[key] = value
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
        parent_execution_parameters = {
            key: value for key, value in parent_parameters.items()
        }
        child_execution_parameters = {
            key: value for key, value in child_parameters.items()
        }
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
                {key: value for key, value in parent_parameters.items()}
                if parent_parameters is not None
                else None
            )
        elif parent_parameters is None:
            keep_child_parameters = True
            continue
        elif bind_same_parameters(child_parameters, parent_parameters):
            parameter_set = {key: value for key, value in parent_parameters.items()}
            keep_child_parameters = True
        else:
            parameter_set = merge_values(child_parameters, parent_parameters)

        unique_values.setdefault(parameter_set_key(parameter_set), parameter_set)

    if keep_child_parameters:
        if TYPE_CHECKING:
            assert child_parameters is not None
        child_parameters = {key: value for key, value in child_parameters.items()}
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
