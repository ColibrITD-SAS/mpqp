def parse_custom_gates(qasm_code: str) -> tuple[dict[str, str], str]:
    """
    Parses custom gate definitions from QASM code.

    Args:
        qasm_code: The QASM code containing custom gate definitions.

    Returns:
        tuple[dict[str, str], str]: A tuple containing a dictionary of custom gate definitions
        and the QASM code with custom gate definitions removed.

    Exemple:
        >>> qasm_str = '''gate rzz(theta) a,b {
        ...     cx a,b;
        ...     u1(theta) b;
        ...     cx a,b;
        ... }
        ... qubit[3] q;
        ... bit[1] c0;
        ... bit[1] c1;
        ... rzz(0.2) q[1], q[2];
        ... c2[0] = measure q[2];'''
        >>> print(parse_custom_gates(qasm_str))
        ({'rzz': [['theta'], ['a', 'b'], 'cx a,b;', 'u1(theta) b;', 'cx a,b;']}, 'qubit[3] q;\nbit[1] c0;\nbit[1] c1;\nrzz(0.2) q[1], q[2];\nc2[0] = measure q[2];')

    """
    custom_gates = {}

    replaced_code = qasm_code

    lines = qasm_code.split("\n")

    in_custom_gate = False
    current_gate_name = ""
    current_gate_definition = []

    for line in lines:
        if line.strip().startswith("gate"):
            in_custom_gate = True
            current_gate_name_parametres = line.split()[1].split("(")
            current_gate_name = current_gate_name_parametres[0]
            current_gate_parametres = (
                current_gate_name_parametres[1][:-1].split(",")
                if len(current_gate_name_parametres) > 1
                else ""
            )
            current_gate_definition = []
            current_gate_qubits = ""
            for elem in line.split()[2:]:
                current_gate_qubits += elem
            current_gate_qubits = current_gate_qubits.replace("{", "").split(",")

            current_gate_definition.append(current_gate_parametres)
            current_gate_definition.append(current_gate_qubits)
            replaced_code = replaced_code.replace(line + "\n", "")
        elif in_custom_gate:
            if line.strip().endswith("}"):
                custom_gates[current_gate_name] = current_gate_definition
                in_custom_gate = False
            else:
                current_gate_definition.append(line.strip())
            replaced_code = replaced_code.replace(line + "\n", "")

    return custom_gates, replaced_code


def replace_custom_gates(qasm_code: str) -> str:
    """
    Replaces instances of custom gates with their definitions in QASM code.

    Args:
        qasm_code : The QASM code containing custom gate calls.

    Returns:
        str: The QASM code with custom gate calls replaced by their definitions.

    Exemple:
        >>> qasm_str = '''gate MyGate a, b {
        ...     h a;
        ...     cx a, b;
        ... }
        ... qreg q[3];
        ... creg c[2];
        ... MyGate q[0], q[1];
        ... measure q -> c;'''
        >>> print(replace_custom_gates(qasm_str))
        qreg q[3];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;

    """
    replaced_code = qasm_code
    custom_gates, replaced_code = parse_custom_gates(qasm_code)

    for gate_name in custom_gates:

        lines = qasm_code.split("\n")
        for line in lines:
            if line.strip().startswith(gate_name + " ") or line.strip().startswith(
                gate_name + "("
            ):
                current_gate_qubits = [
                    elem.replace(",", "").replace(";", "") for elem in line.split()[1:]
                ]

                current_gate_parameters = line.split()[0].split("(")
                current_gate_parameters = (
                    current_gate_parameters[1][:-1].split(",")
                    if len(current_gate_parameters) > 1
                    else []
                )

                all_gate = ""
                for gate in custom_gates[gate_name][2:]:
                    all_gate += gate + "\n"

                for i, parameter in enumerate(custom_gates[gate_name][1]):
                    all_gate = (
                        all_gate.replace(
                            "," + parameter + ",", ", " + current_gate_qubits[i] + ","
                        )
                        .replace(
                            " " + parameter + ",", " " + current_gate_qubits[i] + ","
                        )
                        .replace(parameter + ";", current_gate_qubits[i] + ";")
                        .replace(parameter + " ;", current_gate_qubits[i] + " ;")
                    )

                for i, parameter in enumerate(custom_gates[gate_name][0]):
                    all_gate = all_gate.replace(parameter, current_gate_parameters[i])

                replaced_code = replaced_code.replace(line + "\n", all_gate)

    return replaced_code
