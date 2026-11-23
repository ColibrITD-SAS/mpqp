"""Translate MPQP circuits and circuit bindings to TKET for Quantinuum."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

from mpqp.core.languages import Language
from mpqp.execution._execution_context import (
    PreparedBindingExecutionGroup,
    TranslatedBindingExecutionGroup,
)
from mpqp.execution.devices import QUANTINUUMDevice
from mpqp.execution.job import JobType

if TYPE_CHECKING:
    from pytket.circuit import Circuit as tket_Circuit
    from pytket.circuit import OpType
    from qiskit import QuantumCircuit
    from qiskit._accelerate.circuit import ParameterExpression
    from sympy import Expr

    from mpqp.core._types import BindingParameters
    from mpqp.core.binding import CircuitBinding
    from mpqp.core.circuit import QCircuit


__all__ = ["mpqp_to_tket", "circuitbinding_to_tket"]


def mpqp_to_tket(
    circuit: "QCircuit",
    skip_pre_measure: bool = False,
    skip_measurements: bool = False,
) -> "tket_Circuit":
    """Translate an MPQP circuit to its TKET equivalent.

    MPQP uses Qiskit as an intermediate representation for this translation.
    Parameters remain symbolic in the TKET circuit, so the same translated
    circuit can later be used with different values.

    Args:
        circuit: The original MPQP circuit to translate.
        skip_pre_measure: If ``True``, translate the circuit without its
            pre-measurement operations. See
            :meth:`QCircuit.to_other_language` for more information.
        skip_measurements: If ``True``, translate the circuit without its
            measurement instructions.

    Returns:
        The translated TKET circuit.

    Examples:
        >>> circuit = QCircuit([H(0), CNOT(0, 1), BasisMeasure()])
        >>> tket_circuit = mpqp_to_tket(circuit)  # doctest: +QUANTINUUM
        >>> print(tket_circuit)  # doctest: +QUANTINUUM
        <tket::Circuit, qubits=2, gates=4>
        >>> print(repr(tket_circuit))  # doctest: +QUANTINUUM
        [H q[0]; CX q[0], q[1]; Measure q[0] --> c[0]; Measure q[1] --> c[1]; ]
        >>> circuit_without_measurement = mpqp_to_tket(  # doctest: +QUANTINUUM
        ...     circuit,
        ...     skip_measurements=True,
        ... )
        >>> print(repr(circuit_without_measurement))  # doctest: +QUANTINUUM
        [H q[0]; CX q[0], q[1]; ]

    Note:
        This function does not assign parameter values. Quantinuum execution
        assigns them later, after compiling the symbolic circuit.

    """
    qiskit_circuit = circuit.to_other_language(
        Language.QISKIT,
        skip_pre_measure=skip_pre_measure,
        skip_measurements=skip_measurements,
    )

    return _qiskit_to_tket(qiskit_circuit)


def _get_symbolic_qiskit_to_tket_gate_map() -> dict[str, "OpType"]:
    """Match each supported symbolic Qiskit gate with its TKET gate type.

    ``pytket-qiskit`` can convert a gate when its parameter value is already
    known, such as ``Ry(0.5)``. It cannot convert the same gate while its
    parameter is still symbolic, such as ``Ry(theta)``. This mapping creates
    the equivalent TKET gate while keeping the parameter symbolic.
    """
    from pytket.circuit import OpType

    return {
        "p": OpType.U1,
        "r": OpType.PhasedX,
        "rx": OpType.Rx,
        "ry": OpType.Ry,
        "rz": OpType.Rz,
        "u1": OpType.U1,
        "u2": OpType.U2,
        "u3": OpType.U3,
        "u": OpType.U3,
        "cp": OpType.CU1,
        "crx": OpType.CRx,
        "cry": OpType.CRy,
        "crz": OpType.CRz,
        "cu": OpType.CU3,
        "cu1": OpType.CU1,
        "cu3": OpType.CU3,
        "rxx": OpType.XXPhase,
        "ryy": OpType.YYPhase,
        "rzz": OpType.ZZPhase,
    }


def _qiskit_parameter_to_tket(parameter: object) -> "Expr | float":
    """Convert a Qiskit angle from radians to TKET half-turns.

    Args:
        parameter: Numerical or symbolic Qiskit angle.

    Returns:
        The equivalent TKET angle.

    Raises:
        TypeError: If the angle is neither numerical nor symbolic.
    """
    from qiskit.circuit import ParameterExpression
    from sympy import pi

    if isinstance(parameter, ParameterExpression):
        return _qiskit_expression_to_sympy(parameter) / pi
    if isinstance(parameter, Real):
        return float(parameter) / float(pi)
    raise TypeError(
        "A Qiskit gate parameter must be numerical or symbolic, "
        f"not {type(parameter).__name__}."
    )


def _qiskit_expression_to_sympy(parameter: "ParameterExpression") -> "Expr":
    """Convert a symbolic Qiskit expression to SymPy.

    Qiskit's conversion from ``ParameterExpression`` to SymPy can fail for
    some symbolic power expressions.

    When this happens, the parameters are given unique temporary names while
    SymPy reads the expression. They are then restored as SymPy symbols with
    their original names.
    """
    from qiskit.circuit import Parameter
    from qiskit.circuit.parameterexpression import sympify as qiskit_sympify
    from sympy import Expr, Symbol
    from sympy.parsing.sympy_parser import parse_expr

    try:
        converted_expression = qiskit_sympify(parameter)
        if isinstance(converted_expression, Expr):
            return converted_expression
    except (AttributeError, TypeError):
        pass

    renamed_parameters = {
        qiskit_parameter: Parameter(f"mpqp_parameter_{qiskit_parameter.uuid.hex}")
        for qiskit_parameter in parameter.parameters
    }
    renamed_expression = parameter.subs(renamed_parameters)
    sympy_symbols = {
        str(renamed_parameter.name): Symbol(qiskit_parameter.name)
        for qiskit_parameter, renamed_parameter in renamed_parameters.items()
    }
    converted_expression = parse_expr(str(renamed_expression), local_dict=sympy_symbols)
    if not isinstance(converted_expression, Expr):
        raise TypeError(
            "The symbolic Qiskit parameter could not be converted to SymPy."
        )
    return converted_expression


def _qiskit_to_tket(qiskit_circuit: "QuantumCircuit") -> "tket_Circuit":
    """Convert a Qiskit circuit to TKET while preserving symbolic parameters.

    Instructions without symbols are converted with ``pytket-qiskit``.
    Symbolic gates are added directly to the TKET circuit because the adapter
    cannot currently convert them. Processing the instructions in their
    original order preserves the gate order.
    """
    from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk
    from qiskit.circuit import ParameterExpression

    if not qiskit_circuit.parameters and not isinstance(
        qiskit_circuit.global_phase,
        ParameterExpression,
    ):
        return qiskit_to_tk(qiskit_circuit)

    symbolic_gate_map = _get_symbolic_qiskit_to_tket_gate_map()

    empty_qiskit_circuit = qiskit_circuit.copy_empty_like()
    empty_qiskit_circuit.global_phase = 0
    tket_circuit = qiskit_to_tk(empty_qiskit_circuit)
    qiskit_circuit_without_symbols = qiskit_circuit.copy_empty_like()
    qiskit_circuit_without_symbols.global_phase = 0

    for instruction in qiskit_circuit.data:
        operation = instruction.operation
        if not any(
            isinstance(parameter, ParameterExpression) for parameter in operation.params
        ):
            qiskit_circuit_without_symbols.append(
                operation,
                instruction.qubits,
                instruction.clbits,
            )
            continue

        if qiskit_circuit_without_symbols.data:
            tket_circuit.append(qiskit_to_tk(qiskit_circuit_without_symbols))
            qiskit_circuit_without_symbols.clear()
        gate_type = symbolic_gate_map.get(operation.name)
        if gate_type is None:
            raise NotImplementedError(
                "Symbolic Qiskit gate "
                f"'{operation.name}' is not supported by the TKET translator."
            )
        if operation.name == "cu" and operation.params[-1] != 0:
            raise NotImplementedError(
                "Symbolic Qiskit CU gates with a nonzero global phase are not "
                "supported by the TKET translator."
            )

        parameters = operation.params
        if operation.name == "cu":
            parameters = parameters[:-1]
        tket_parameters = [
            _qiskit_parameter_to_tket(parameter) for parameter in parameters
        ]
        qubits = [
            tket_circuit.qubits[qiskit_circuit.find_bit(qubit).index]
            for qubit in instruction.qubits
        ]
        tket_circuit.add_gate(gate_type, tket_parameters, qubits)

    if qiskit_circuit_without_symbols.data:
        tket_circuit.append(qiskit_to_tk(qiskit_circuit_without_symbols))
    tket_circuit.add_phase(_qiskit_parameter_to_tket(qiskit_circuit.global_phase))
    return tket_circuit


def bind_tket_parameters(
    circuit: "tket_Circuit",
    values: "BindingParameters | None",
) -> "tket_Circuit":
    """Return a copy of a TKET circuit with its parameter values assigned.

    The original circuit is not changed, so it can be reused with another set
    of values.

    Args:
        circuit: The TKET circuit to bind.
        values: The value of each parameter, keyed by its expression or name.

    Returns:
        A new TKET circuit containing the assigned values.

    Raises:
        ValueError: If a value is missing for any circuit symbol.
        TypeError: If a parameter value is not a real number.
    """
    bound_circuit = circuit.copy()
    symbols = {str(symbol): symbol for symbol in bound_circuit.free_symbols()}
    binding_values: dict[str, float] = {}
    for key, value in (values or {}).items():
        if not isinstance(value, Real):
            raise TypeError(
                "TKET circuit parameters must be real numbers, but parameter "
                f"{key} has value {value}."
            )
        binding_values[str(key)] = float(value)
    missing = [name for name in symbols if name not in binding_values]
    if missing:
        raise ValueError(f"Missing values for circuit parameters: {missing}")
    if symbols:
        bound_circuit.symbol_substitution(
            {symbol: binding_values[name] for name, symbol in symbols.items()}
        )
    return bound_circuit


def circuitbinding_to_tket(
    binding: "CircuitBinding", device: QUANTINUUMDevice
) -> list[TranslatedBindingExecutionGroup]:
    """Translate the executions in a circuit binding to TKET.

    Executions that use the same circuit and measurement share one symbolic
    TKET circuit. Their parameter values and original positions are kept in
    separate execution contexts. This allows Quantinuum to compile the shared
    circuit once and then run it with each set of values.

    Args:
        binding: The original MPQP circuit binding to translate.
        device: The Quantinuum device on which the translated circuits will be
            executed.

    Returns:
        Each symbolic TKET circuit with the executions that use it.

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

    Note:
        Parameter values are assigned after the TKET circuit is compiled for
        the selected backend.
    """
    from mpqp.tools.binding_utils import prepare_binding_execution_groups

    execution_groups = prepare_binding_execution_groups(binding, device)
    return translate_binding_execution_groups(
        execution_groups,
        device,
        binding.job_type,
    )


def translate_binding_execution_groups(
    execution_groups: list[PreparedBindingExecutionGroup],
    device: QUANTINUUMDevice,
    job_type: JobType,
) -> list[TranslatedBindingExecutionGroup]:
    """Translate prepared binding executions to TKET.

    Args:
        execution_groups: Binding executions prepared before provider
            translation.
        device: Quantinuum device for which the circuits are translated.
        job_type: Type of job represented by the binding.

    Returns:
        Each symbolic TKET circuit with the executions that share it.
    """
    translated_executions = []
    for execution_group in execution_groups:
        if execution_group.execution_circuit.measurements:
            tket_circuit = mpqp_to_tket(
                execution_group.execution_circuit,
                skip_measurements=job_type == JobType.STATE_VECTOR,
            )
        else:
            tket_circuit = execution_group.original_circuit.transpiled_for_device(
                device
            )
            if TYPE_CHECKING:
                assert isinstance(tket_circuit, tket_Circuit)
        translated_executions.append(
            TranslatedBindingExecutionGroup(
                circuit=tket_circuit,
                contexts=execution_group.contexts,
            )
        )

    return translated_executions
