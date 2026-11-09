"""This module provides the utilities used to translate MPQP circuits and
circuit bindings to TKET for Quantinuum execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpqp.core.languages import Language
from mpqp.execution._execution_context import ExecutionContext
from mpqp.execution.devices import QUANTINUUMDevice
from mpqp.execution.job import Job, JobType

if TYPE_CHECKING:
    from pytket.circuit import Circuit as tket_Circuit

    from mpqp.core.binding import BindingParameters, CircuitBinding
    from mpqp.core.circuit import QCircuit


__all__ = ["mpqp_to_tket", "circuitbinding_to_tket"]


def mpqp_to_tket(
    circuit: "QCircuit",
    skip_pre_measure: bool = False,
    skip_measurements: bool = False,
    values: "BindingParameters | None" = None,
) -> "tket_Circuit":
    """Translate an MPQP circuit to its TKET equivalent.

    The circuit is first translated to Qiskit and then converted to TKET with
    ``pytket-qiskit``. Symbolic parameters are replaced before this conversion
    because the converter expects numerical values.

    Args:
        circuit: The original MPQP circuit to translate.
        skip_pre_measure: If ``True``, translate the circuit without its
            pre-measurement operations. See
            :meth:`QCircuit.to_other_language` for more information.
        skip_measurements: If ``True``, translate the circuit without its
            measurement instructions.
        values: Values used to replace the circuit's symbolic parameters before
            converting it to TKET.

    Returns:
        The translated TKET circuit.

    Raises:
        ValueError: If the circuit contains symbolic parameters for which no
            numerical value was provided.

    Examples:
        >>> circuit = QCircuit([H(0), CNOT(0, 1), BasisMeasure()])
        >>> tket_circuit = mpqp_to_tket(circuit)  # doctest: +QUANTINUUM
        >>> print(repr(tket_circuit))  # doctest: +ELLIPSIS, +QUANTINUUM
        [H q[0]; CX q[0], q[1]; Measure ...]
        >>> circuit_without_measurement = mpqp_to_tket(  # doctest: +QUANTINUUM
        ...     circuit,
        ...     skip_measurements=True,
        ... )
        >>> print(repr(circuit_without_measurement))  # doctest: +QUANTINUUM
        [H q[0]; CX q[0], q[1]; ]

    """
    qiskit_circuit = circuit.to_other_language(
        Language.QISKIT,
        skip_pre_measure=skip_pre_measure,
        skip_measurements=skip_measurements,
    )

    binding_values = {str(key): value for key, value in (values or {}).items()}
    unbound_parameters = [
        parameter.name
        for parameter in qiskit_circuit.parameters
        if parameter.name not in binding_values
    ]
    if unbound_parameters:
        raise ValueError(f"Missing values for circuit parameters: {unbound_parameters}")
    if qiskit_circuit.parameters:
        qiskit_circuit = qiskit_circuit.assign_parameters(
            {
                parameter: binding_values[parameter.name]
                for parameter in qiskit_circuit.parameters
            },
            inplace=False,
        )

    from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

    return qiskit_to_tk(qiskit_circuit)


def circuitbinding_to_tket(
    binding: "CircuitBinding", device: QUANTINUUMDevice
) -> list[tuple["tket_Circuit", ExecutionContext]]:
    """Translate all executions contained in a circuit binding to TKET.

    A circuit binding can contain several circuits, parameter values and
    measurements. It is first unrolled into individual executions. Each
    execution is then translated to one TKET circuit and paired with an
    :class:`ExecutionContext` containing the information needed to rebuild the
    corresponding MPQP result in the correct order.

    Args:
        binding: The original MPQP circuit binding to translate.
        device: The Quantinuum device on which the translated circuits will be
            executed.

    Returns:
        The TKET circuits and their MPQP execution contexts, in the same order
        as the unrolled binding.

    Examples:
        >>> binding = CircuitBinding(
        ...     [QCircuit([H(0)]), QCircuit([X(0)])],
        ...     measurements=BasisMeasure(shots=10),
        ... )
        >>> translated = circuitbinding_to_tket(  # doctest: +QUANTINUUM
        ...     binding,
        ...     QUANTINUUMDevice.TKET_AER_SIMULATOR,
        ... )
        >>> [  # doctest: +QUANTINUUM
        ...     [command.op.type.name for command in circuit.get_commands()]
        ...     for circuit, _ in translated
        ... ]
        [['H', 'Measure'], ['X', 'Measure']]
    """
    translated_executions: list[tuple[tket_Circuit, ExecutionContext]] = []

    from mpqp.tools.binding_utils import create_binding_execution_circuit

    for index, (circuit, values, measurement) in enumerate(binding.unroll()):
        context_circuit = create_binding_execution_circuit(circuit, measurement)
        tket_circuit = mpqp_to_tket(
            context_circuit,
            skip_measurements=binding.job_type == JobType.STATE_VECTOR,
            values=values,
        )

        translated_executions.append(
            (
                tket_circuit,
                ExecutionContext(
                    job=Job(
                        binding.job_type,
                        context_circuit,
                        device,
                        values=values,
                    ),
                    index=index,
                ),
            )
        )

    return translated_executions
