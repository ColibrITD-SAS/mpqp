"""
The latest version of OpenQASM (3.0, started in 2020) has been released by a
collaborative group from IBM Quantum, AWS Quantum Computing, Zapata Computing,
Zurich Instruments, and the University of Oxford. This version extends OpenQASM
2.0, adding advanced features and modifying parts of the syntax and grammar.
Some aspects of OpenQASM 2.0 are not fully backward compatible, hence the need
to keep track of instructions requiring custom definitions in :class:`Instr`.

To aid in the transition, this module provides conversion functions for moving
between OpenQASM 2.0 and 3.0, as well as managing user-defined gates and handling
code transformations. Key functionalities include:

1. **OpenQASM 2.0 to 3.0 Conversion**:
    - :func:`convert_instruction_2_to_3`: Converts individual instructions from QASM 2.0
      syntax to 3.0, handling specific syntax adjustments.
    - :func:`parse_openqasm_2_file`: Splits OpenQASM 2.0 code into individual instructions,
      preserving gate declarations to ensure proper handling during conversion.
    - :func:`open_qasm_2_to_3`: Main function for converting OpenQASM 2.0 code to 3.0. 
      It adds necessary library includes.
    - :func:`open_qasm_file_conversion_2_to_3`: Reads from the specified file, and outputs
      the converted file in QASM 2.0 syntax.

2. **OpenQASM 3.0 to 2.0 Conversion**:
    - :func:`convert_instruction_3_to_2`: Converts individual instructions from QASM 3.0
      syntax to 2.0, handling specific syntax adjustments.
    - :func:`parse_openqasm_3_file`: Splits OpenQASM 3.0 code into individual instructions,
      preserving gate declarations to ensure proper handling during conversion.
    - :func:`open_qasm_3_to_2`: Main function for converting OpenQASM 3.0 code to 2.0. 
      It adds necessary library includes, tracks cumulative global phases. 
    - :func:`open_qasm_file_conversion_3_to_2`: Reads from the specified file, and outputs
      the converted file in QASM 2.0 syntax.

3. **User-Defined Gate Handling**:
    - **UserGate Class**: Represents user-defined gates in OpenQASM. Each ``UserGate`` instance
      stores the gate's name, parameters, qubits, and instruction sequence.
    - :func:`parse_user_gates`: Extracts and stores user-defined gate definitions 
      from OpenQASM code, removing them from the main code to allow separate handling. 
      Custom gates are identified using the ``GATE_PATTERN`` regex and stored as ``UserGate`` instances.
    - :func:`remove_user_gates`: Replaces calls to user-defined gates in OpenQASM code with
      their expanded definitions. This function relies on ``parse_user_gates`` to retrieve 
      gate definitions, and it substitutes parameter and qubit values within each gate's body 
      instructions for accurate expansion.
  
4. **Supporting Functions**:
    - :func:`open_qasm_hard_includes`: Combines multiple OpenQASM files into a single file 
      with resolved includes, simplifying code management for projects with multiple source files.

"""

import os
import re
from enum import Enum, auto
from os.path import splitext
from pathlib import Path
from typing import Optional
from warnings import warn

from anytree import Node, PreOrderIter
from typeguard import typechecked

from mpqp.tools.errors import InstructionParsingError, OpenQASMTranslationWarning


class Instr(Enum):
    """Special instruction for which the definition needs to included in the
    file."""

    STD_LIB = auto()
    QE_LIB = auto()
    CSX = auto()
    U0 = auto()
    CU3 = auto()
    SXDG = auto()
    RZZ = auto()
    RXX = auto()
    RCCX = auto()
    RC3X = auto()
    C3X = auto()
    C4X = auto()
    C3SQRTX = auto()
    OQASM2_ALL_STDGATES = auto()
    OQASM3_ALL_STDGATES = auto()
    BRAKET_CUSTOM_INCLUDE = auto()


std_gates_2 = [
    "cu3",
    "csx",
    "sxdg",
    "u0",
    "rxx",
    "rzz",
    "rccx",
    "rc3x",
    "c3x",
    "c3sqrtx",
    "c4x",
]
std_gates_2_3 = ["u", "swap", "cswap", "cp"]
qelib1_gates = [
    "u3",
    "u2",
    "u1",
    "u0",
    "cx",
    "id",
    "u",
    "p",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "sxdg",
    "rx",
    "ry",
    "rz",
    "swap",
    "cz",
    "cy",
    "ch",
    "ccx",
    "crz",
    "cu1",
    "cu3",
    "cswap",
    "crx",
    "cry",
    "cp",
    "cu",
]
std_gates_3 = [
    "u1",
    "u2",
    "u3",
    "cx",
    "CX",
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "cz",
    "cy",
    "ch",
    "ccx",
    "crx",
    "cry",
    "crz",
    "cu",
    "p",
    "cphase",
    "phase",
    "sx",
]
std_gates_3_to_2_map = {
    "U": "u",
    "phase": "u1",
    "cphase": "cu1",
}


@typechecked
def qasm_code(instr: Instr) -> str:
    """Return the string corresponding of the declaration of the instruction in
    parameter. It is also used to return the whole standard library string when
    we hard include it.

    Args:
        instr: Instr for which we want the corresponding OpenQASM code.

    Returns:
        OpenQASM definition of ``instr``.
    """

    # FIXME: if run from outside the mpqp folder, the following line will fuck
    # things up
    headers_folder = os.path.dirname(__file__) + "/header_codes/"
    special_file_names = {
        Instr.OQASM2_ALL_STDGATES: "qelib1.inc",
        Instr.OQASM3_ALL_STDGATES: "stdgates.inc",
        Instr.BRAKET_CUSTOM_INCLUDE: "braket_custom_include.inc",
    }

    if instr in special_file_names:
        file_name = special_file_names[instr]
    else:
        file_name = instr.name.lower() + ".qasm"

    with open(headers_folder + file_name, "r") as f:
        return f.read()


@typechecked
def parse_openqasm_2_file(code: str) -> list[str]:
    """Splits a complete OpenQASM2 program into individual instructions.

    Args:
        code: The complete OpenQASM 2.0 program.

    Returns:
        List of instructions.

    Note:
        we do not check for correct syntax, it is assumed that the code is well
        formed.
    """
    # 3M-TODO: deal with comments, for the moment we remove them all

    # removing comment
    cleaned_code = "\n".join(line.lstrip() for line in code.splitlines())

    cleaned_code = "".join([loc.split("//")[0] for loc in cleaned_code.split("\n")])

    cleaned_code = cleaned_code.replace("\t", " ").strip()

    gate_matches = list(re.finditer(r"gate .*?}", cleaned_code))
    sanitized_start = (
        cleaned_code[: gate_matches[0].span()[0]] if gate_matches else cleaned_code
    )
    instructions = sanitized_start.split(";")

    for i in range(len(gate_matches)):
        # gates definition are added as a single instruction
        instructions.append(cleaned_code[slice(*(gate_matches[i].span()))])
        # all instructions between two gate definitions are added individually
        instructions.extend(
            cleaned_code[
                gate_matches[i].span()[1] : (
                    None
                    if i == len(gate_matches) - 1
                    else gate_matches[i + 1].span()[0]
                )
            ].split(";")
        )

    return list(filter(lambda i: i != "", instructions))


@typechecked
def convert_instruction_2_to_3(
    instr: str,
    included_instr: set[Instr],
    included_tree_current: Node,
    defined_gates: set[str],
    path_to_main: Optional[str] = None,
    translation_warning: bool = True,
) -> tuple[str, str]:
    """Some instructions changed name from QASM 2 to QASM 3, also the way to
    import files changed slightly. This function operates those changes on a
    single instruction.

    Args:
        instr: Instruction to be upgraded.
        included_instr: Some instructions need new imports, in order to keep
            track of which instruction are already.
        imported in the overall scope, a dictionary of already included
            instructions is passed and modified along.
        included_tree_current: Current Node in the file inclusion tree.
        defined_gates: Set of custom gates already defined.
        path_to_main: Path to the main folder from which include paths are
            described.

    Returns:
        The upgraded instruction and the potential code to add in the header as
        the second element.
    """
    if path_to_main is None:
        path_to_main = "."

    def add_std_lib():
        """Add the instruction of including the standard library of OpenQASM3
        code if it is not already done"""
        if Instr.STD_LIB not in included_instr:
            included_instr.add(Instr.STD_LIB)
            to_add = qasm_code(Instr.STD_LIB)
        else:
            to_add = ""
        return to_add

    header_code = ""
    instructions_code = ""

    instr_name = instr.split(" ")[0].split("(")[0]
    # If the line is the OpenQASM header
    if instr.startswith("OPENQASM 2.0"):
        header_code += "OPENQASM 3.0;\n"
    elif instr_name == "include":
        path = instr.split(" ")[-1].strip("'\"")
        if path != "qelib1.inc":
            if any(path in node.name for node in included_tree_current.ancestors):
                raise RuntimeError("Circular dependency detected.")
            # Convert the file included, add it to the inclusion tree,
            # and create a new file and include it in the converted code
            if not any(
                path in node.name for node in PreOrderIter(included_tree_current.root)
            ):  # checks in the path is not already included
                with open(f"{path_to_main}/{path}", "r") as f:
                    child = Node(path, parent=included_tree_current)
                    converted_content = open_qasm_2_to_3(
                        f.read(),
                        child,
                        path_to_main,
                        defined_gates,
                        translation_warning,
                    )
                new_path = splitext(path)[0] + "_converted" + splitext(path)[1]
                with open(f"{path_to_main}/{new_path}", "w") as f:
                    f.write(converted_content)
                header_code += f"include '{new_path}';\n"
    elif instr_name in {"qreg", "creg"}:
        # classical and quantum bits have the same structure
        # `qreg <name>[<size>];` -> `qubit[<size>] <name>;`
        m = re.match(r"(\w)reg\s+(.+?)\[(\d+)\]", instr)
        if m is None:
            raise InstructionParsingError("On instruction: " + instr)
        bit_type_prefix = "qu" if m.group(1) == "q" else ""
        instructions_code += f"{bit_type_prefix}bit[{m.group(3)}] {m.group(2)};\n"
    elif instr_name == "measure":
        # `measure <q_name[+reg]?> -> <c_name[+reg]?>;` -> `<c_name[+reg]?> = measure <q_name[+reg]?>;`
        m = re.match(r"measure\s+(.+?)\s+->\s+(.+)", instr)
        if m is None:
            raise InstructionParsingError("On instruction: " + instr)
        instructions_code += f"{m.group(2)} = measure {m.group(1)};\n"
    elif instr_name in {"reset", "barrier"}:
        instructions_code += instr + ";\n"
    elif instr_name.lower() == "u":
        if translation_warning:
            warn(
                """
There is a phase e^(i(a+c)/2) difference between U(a,b,c) gate in 2.0 and 3.0.
We handled that for you by adding the extra phase at the right place. 
Be careful if you want to create a control gate from this circuit/gate, the
phase can become non-global.""",
                OpenQASMTranslationWarning,
            )
        header_code += add_std_lib()
        instructions_code += "u3" + instr[1:] + ";\n"
    elif instr_name == "cu1":
        header_code += add_std_lib()
        instructions_code += "cp" + instr[3:] + ";\n"
    elif instr_name in std_gates_3 + std_gates_2_3 + std_gates_2:
        instructions_code += instr + ";\n"
        header_code += add_std_lib()
        new_instr = (
            [instruc for instruc in Instr if instr_name == instruc.name.lower()][0]
            if instr_name in std_gates_2
            else None
        )
        if new_instr is not None and new_instr not in included_instr:
            included_instr.add(new_instr)
    elif instr_name == "gate":
        defined_gates.add(instr.split()[1])
        g_string = instr.split("{")[0] + "{\n"
        g_instructions = filter(
            lambda i: not re.fullmatch(r"\s*", i),
            instr.split("{")[1].split("}")[0].split(";"),
        )
        for instruction in g_instructions:
            instruction = instruction.strip()
            i_code, h_code = convert_instruction_2_to_3(
                instruction,
                included_instr,
                included_tree_current,
                defined_gates,
                path_to_main,
                translation_warning,
            )
            g_string += " " * 4 + i_code
            header_code += h_code
        instructions_code += g_string + "}\n"
    elif instr_name == "if":
        if_statement = instr.split(")")[0] + ")"
        nested_instr = ")".join(instr.split(")")[1:])
        i_code, h_code = convert_instruction_2_to_3(
            nested_instr,
            included_instr,
            included_tree_current,
            defined_gates,
            path_to_main,
            translation_warning,
        )
        instructions_code += if_statement + i_code
        header_code += h_code
    elif instr_name == "opaque":
        raise NotImplementedError("opaque exports not handled yet")
    else:
        gate = instr.split()[0]
        if gate == "ctrl":
            gate = instr.split()[2]
        if gate not in defined_gates:
            raise ValueError(f"Gates undefined at the time of usage: {gate}")
        if len(instr) != 0:
            instructions_code += instr + ";\n"

    return instructions_code, header_code


@typechecked
def open_qasm_2_to_3(
    code: str,
    included_tree_current_node: Optional[Node] = None,
    path_to_file: Optional[str] = None,
    defined_gates: Optional[set[str]] = None,
    translation_warning: bool = True,
) -> str:
    """Converts an OpenQASM code from version 2.0 and 3.0.

    This function will also recursively go through the imported files to
    translate them too. It is a partial conversion (the ``opaque`` keyword is
    not handled and comments are stripped) for helping building temporary
    bridges between different platforms that use different versions.

    Args:
        code: String containing the OpenQASM 2.0 code and instructions.
        included_tree_current_node: Current Node in the file inclusion tree.
        path_to_file: Path to the location of the file from which the code is
            coming (useful for locating imports).
        defined_gates: Set of custom gates already defined.

    Returns:
        Converted OpenQASM code in the 3.0 version.

    Example:
        >>> qasm2_str = '''OPENQASM 2.0;
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0],q[1];
        ... measure q[0] -> c[0];
        ... measure q[1] -> c[1];
        ... '''
        >>> print(open_qasm_2_to_3(qasm2_str)) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0],q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];


    """
    if included_tree_current_node is None:
        included_tree_current_node = Node("initial_code")
    if path_to_file is None:
        path_to_file = "."
    if defined_gates is None:
        defined_gates = set()

    header_code = ""
    instructions_code = ""

    instructions = parse_openqasm_2_file(code)

    included_instructions = set()

    defined_gates.update(std_gates_2_3 + std_gates_3)

    for instr in instructions:
        i_code, h_code = convert_instruction_2_to_3(
            instr,
            included_instructions,
            included_tree_current_node,
            defined_gates,
            path_to_file,
            translation_warning,
        )
        header_code += h_code
        instructions_code += i_code

    target_code = header_code + "\n" + instructions_code

    return target_code


@typechecked
def open_qasm_file_conversion_2_to_3(
    path: str, translation_warning: bool = True
) -> str:
    """Converts an OpenQASM code in a file from version 2.0 and 3.0.

    This function is a shorthand to initialize :func:`open_qasm_2_to_3` with the
    correct values.

    Args:
        path: Path to the file containing the OpenQASM 2.0 code, and eventual
            imports.

    Returns:
        Converted OpenQASM code in the 3.0 version.

    Examples:
        >>> example_dir = "examples/scripts/qasm_files/"
        >>> with open(example_dir + "main.qasm", "r") as f:
        ...     print(f.read()) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 2.0;
        include "include1.qasm";
        include "include2.qasm";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        gate2 q[0];
        gate3 q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        >>> print(open_qasm_file_conversion_2_to_3(example_dir + "main.qasm")) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 3.0;
        include 'include1_converted.qasm';
        include 'include2_converted.qasm';
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0],q[1];
        gate2 q[0];
        gate3 q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        >>> with open(example_dir + "include1_converted.qasm", "r") as f:
        ...     print(f.read()) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 3.0;
        include "stdgates.inc";
        gate gate2 a {
            u3(pi, -pi/2, pi/2) a;
        }
        >>> with open(example_dir + "include2_converted.qasm", "r") as f:
        ...     print(f.read()) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 3.0;
        include "stdgates.inc";
        gate gate3 a, b {
            u3(0, -pi/2, pi/3) a;
            cz a, b;
        }

    """

    with open(path, "r") as f:
        code = f.read()
        return open_qasm_2_to_3(
            code,
            Node(path),
            str(Path(path).parent),
            translation_warning=translation_warning,
        )


@typechecked
def open_qasm_hard_includes(
    code: str,
    included_files: set[str],
    path_to_file: Optional[str] = None,
    is_openqasm_header_included: bool = False,
    remove_included: bool = True,
) -> str:
    r"""Converts an OpenQASM code (2.0 and 3.0) to use no includes, but writes
    every instruction in previously included files, directly in the code
    returned.

    Args:
        code: String containing the OpenQASM code and instructions.
        included_files: The set of files already included, used to avoid
            duplicate imports and circular dependencies. This set should be
            initialized with the name of the root file you started with.
        path_to_file: Path used to localize files that are included.
        is_openqasm_header_included: Boolean used to only include once the
            OpenQASM header.

    Returns:
        Include-less OpenQASM code.

    Example:
        >>> examples_folder = "tests/qasm/qasm_examples"
        >>> filename = examples_folder + "/with_include.qasm"
        >>> with open(filename) as f:
        ...     print(open_qasm_hard_includes(f.read(), {filename}).strip("\n"))
        gate csx a, b {
            ctrl @ sx a, b;
        }

    """
    if path_to_file is None:
        path_to_file = "./"

    lines = code.split("\n")
    converted_code = []

    for line in lines:
        if "include" in line:
            line_array = line.split()
            if not remove_included:
                converted_code.append(line)

            file_name = line_array[line_array.index("include") + 1].strip(";'\"")
            if file_name not in included_files:
                included_files.add(file_name)
                if file_name in {"qelib1.inc"}:
                    converted_code.append(qasm_code(Instr.OQASM2_ALL_STDGATES))
                elif file_name in {"stdgates.inc"}:
                    converted_code.append(qasm_code(Instr.OQASM3_ALL_STDGATES))
                elif file_name in {"braket_custom_include.inc"}:
                    converted_code.append(qasm_code(Instr.BRAKET_CUSTOM_INCLUDE))
                else:
                    with open(path_to_file + file_name, "r") as f:
                        converted_code.append(
                            open_qasm_hard_includes(
                                f.read(),
                                included_files,
                                path_to_file,
                                is_openqasm_header_included,
                            )
                        )
        elif line.startswith("OPENQASM "):
            if not is_openqasm_header_included:
                converted_code.append(line)
                is_openqasm_header_included = True
        else:
            if not line.startswith("//"):
                converted_code.append(line)

    return "\n".join(converted_code)


class UserGate:
    """Represents a custom user-defined quantum gate with specified parameters, qubits, and instructions.
    This class serves as a template for custom gates that can be used in a quantum circuit.

    Args:
        name: The name of the user-defined gate.
        parameters: A list of parameter names that the gate requires (e.g., angles, coefficients).
        qubits: A list of qubit identifiers that the gate operates on.
        instructions: A list of instructions (quantum operations) that define the gate's behavior.

    """

    def __init__(
        self,
        name: str,
        parameters: list[str],
        qubits: list[str],
        instructions: list[str],
    ):
        self.name = name
        self.parameters = parameters
        self.qubits = qubits
        self.instructions = instructions

    def __repr__(self):
        return f"UserGate(name={self.name}, parameters={self.parameters}, qubits={self.qubits}, instructions={self.instructions})"

    def __str__(self):
        return (
            f"gate {self.name}({', '.join(self.parameters)}) {', '.join(self.qubits)} "
            + "{\n"
            + '\n'.join(self.instructions)
            + "\n}"
        )

    def dict(self):
        return {
            "name": self.name,
            "parameters": self.parameters,
            "qubits": self.qubits,
            "instructions": self.instructions,
        }


# example of custom gate declaration:
# gate rotation (theta) q1, q2 { rx (theta) q1; cnot q1, q2; }
#      --------  -----  ------  -----------------------------
#        ^         ^      ^                  ^
#     gate_name  param? qubits         instructions
GATE_PATTERN = re.compile(
    r"gate\s+(?P<name>\w+)\s*(\((?P<param>[^)]+)\))?\s*(?P<qubits>\w+\s*(?:,\s*\w+)*)\s*{(?P<instructions>[^}]*)}",
    re.MULTILINE | re.DOTALL,
)
# example of gate call:
# rotation (theta) q1, q2;
# --------  -----  ------
#    ^        ^       ^
# gate_name  param? qubits
GATE_CALL_PATTERN = re.compile(
    r"(?P<gate>\w+)\s*(\((?P<params>[^)]*)\))?\s*(?P<qubits>[^;]*);",
    re.MULTILINE | re.DOTALL,
)


def parse_user_gates(
    qasm_code: str, skip_qelib1: bool = False
) -> tuple[list[UserGate], str]:
    r"""Parses user gate definitions from QASM code.

    Args:
        qasm_code: The QASM code containing user gate definitions.

    Returns:
        A tuple containing a dictionary of user gate definitions
        and the QASM string stripped of it's user gate definitions.

    Example:
        >>> qasm_str = '''gate rzz(theta) a,b {
        ...     cx a,b;
        ...     u1(theta) b;
        ...     cx a,b;
        ... }
        ... qubit[3] q;
        ... creg c[2];
        ... rzz(0.2) q[1], q[2];
        ... c2[0] = measure q[2];'''
        >>> user_gates, qasm_code = parse_user_gates(qasm_str)
        >>> print(user_gates)
        [UserGate(name=rzz, parameters=['theta'], qubits=['a', 'b'], instructions=['cx a,b;', 'u1(theta) b;', 'cx a,b;'])]
        >>> print(qasm_code)
        qubit[3] q;
        creg c[2];
        rzz(0.2) q[1], q[2];
        c2[0] = measure q[2];

    """
    copy_qasm_code = "\n".join(
        [line.lstrip() for line in qasm_code.splitlines() if line.strip()]
    )
    included_files = set()
    if skip_qelib1:
        included_files.add("qelib1.inc")
    qasm_code_include = open_qasm_hard_includes(
        copy_qasm_code, included_files, remove_included=False
    )
    matches = list(GATE_PATTERN.finditer(qasm_code_include))
    user_gates = []
    for match in matches:
        parameters = (
            [p.strip() for p in match.group("param").split(',')]
            if match.group("param")
            else []
        )
        qubits = [q.strip() for q in match.group("qubits").split(',')]
        instructions = [
            line.strip() + ";"
            for line in match.group("instructions").split(';')
            if line.strip()
        ]
        user_gate = UserGate(
            name=match.group("name"),
            parameters=parameters,
            qubits=qubits,
            instructions=instructions,
        )
        user_gates.append(user_gate)
        copy_qasm_code = copy_qasm_code.replace(match.group(0), "")

    return user_gates, copy_qasm_code.strip()


def remove_user_gates(qasm_code: str, skip_qelib1: bool = False) -> str:
    """Replaces instances of user gates with their definitions in the given QASM
    code. This uses :func:`parse_user_gates` to separate the gate definitions
    from the rest of the code.

    Args:
        qasm_code: The QASM code containing user gate calls.

    Returns:
        The QASM code with user gate calls replaced by their definitions.

    Example:
        >>> qasm_str = '''gate MyGate a, b {
        ...      h a;
        ...      cx a, b;
        ... }
        ... qreg q[3];
        ... creg c[2];
        ... MyGate q[0], q[1];
        ... measure q -> c;'''
        >>> print(remove_user_gates(qasm_str))
        qreg q[3];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;

    """
    user_gates, qasm_code = parse_user_gates(qasm_code, skip_qelib1)
    previous_qasm_body = None
    while previous_qasm_body != qasm_code:
        previous_qasm_body = qasm_code
        for gate in user_gates:
            for match in GATE_CALL_PATTERN.finditer(qasm_code):
                if match.group("gate") == gate.name:
                    param_values = (
                        [p.strip() for p in match.group("params").split(',')]
                        if match.group("params")
                        else []
                    )
                    qubit_values = [q.strip() for q in match.group("qubits").split(',')]

                    expanded = []
                    replacements = {}
                    for qubit, value in zip(gate.qubits, qubit_values):
                        replacements[qubit] = value
                    for param, value in zip(gate.parameters, param_values):
                        replacements[param] = value

                    def replace(match: re.Match[str]):
                        return replacements[match.group(0)]

                    for instruction in gate.instructions:
                        expanded.append(
                            re.sub(
                                '|'.join(
                                    r'\b%s\b' % re.escape(s) for s in replacements
                                ),
                                replace,
                                instruction,
                            )
                        )

                    expanded_instructions = "\n".join(expanded)
                    qasm_code = qasm_code.replace(match.group(0), expanded_instructions)

    return qasm_code


def remove_include_and_comment(qasm_code: str) -> str:
    r"""
    Removes lines that start with 'include' or comments (starting with '\\')
    from a given OpenQASM code string.

    Args:
        qasm_code: The input QASM code as a string.

    Returns:
        The modified QASM code with 'include' lines and comments removed.

    Example:
        >>> qasm_code = '''include "stdgates.inc";
        ... qreg q[2];
        ... // This is a comment
        ... H q[0];'''
        >>> print(remove_include_and_comment(qasm_code))
        qreg q[2];
        H q[0];

    """
    replaced_code = []
    for line in qasm_code.split("\n"):
        line = line.lstrip()
        if line.startswith("include") or line.startswith("//"):
            pass
        else:
            replaced_code.append(line)
    return "\n".join(replaced_code)


@typechecked
def convert_instruction_3_to_2(
    instr: str,
    included_instr: set[Instr],
    included_tree_current: Node,
    defined_gates: set[str],
    path_to_main: Optional[str] = None,
    gphase: float = 0.0,
) -> tuple[str, str, float]:
    r"""Some instructions changed name from QASM 2 to QASM 3, also the way to
    import files changed slightly. This function operates those changes on a
    single instruction.

    Args:
        instr: Instruction to be upgraded.
        included_instr: Some instructions need new imports, in order to keep
            track of which instruction are already.
        imported in the overall scope, a dictionary of already included
            instructions is passed and modified along.
        included_tree_current: Current Node in the file inclusion tree.
        defined_gates: Set of custom gates already defined.
        path_to_main: Path to the main folder from which include paths are
            described.
        gphase: The global phase of a circuit, which is not handled in OpenQASM2.

    Returns:
        The upgraded instruction, the potential code to add in the header as
        the second element and the global phase of the circuit.

    Example:
        >>> convert_instruction_3_to_2("phase(0.3) q1[0];",set(),Node(""),set())
        ('u1(0.3) q1[0];;\n', '', 0.0)

    """
    # 6M-TODO: not handled for loop, or a switch case, or pulse and low level quantum operations, etc.
    if path_to_main is None:
        path_to_main = "."

    def add_qe_lib():
        """Add the instruction of including the standard library of OpenQASM2
        code if it is not already done"""
        if Instr.QE_LIB not in included_instr:
            included_instr.add(Instr.QE_LIB)
            to_add = qasm_code(Instr.QE_LIB)
        else:
            to_add = ""
        return to_add

    header_code = ""
    instructions_code = ""

    instr_match = re.match(r"\s*(\w+)\s*", instr)
    if instr_match:
        instr_name = instr_match.group(1)
    else:
        raise ValueError(f"Could not parse instruction: {instr}")

    if instr_name == "OPENQASM":
        instr_match = re.match(r"OPENQASM\s*(\d+.\d+)\s*", instr)
        if instr_match:
            version = float(instr_match.group(1))
            if version != 3.0:
                raise ValueError(
                    f"Only OPENQASM 3.0 is supported. OPENQASM {version} is not valid: {instr}"
                )
            else:
                header_code += "OPENQASM 2.0;\n"
        else:
            raise ValueError(f"Could not parse OPENQASM (version): {instr}")
    elif instr_name == "include":
        m = re.match(r'\s*include\s+["\']([^"\']+)["\']', instr)
        if m:
            path = m.group(1).strip("'\"")
            if path != "stdgates.inc":
                if any(path in node.name for node in included_tree_current.ancestors):
                    raise RuntimeError("Circular dependency detected.")
                if not any(
                    path in node.name
                    for node in PreOrderIter(included_tree_current.root)
                ):
                    with open(f"{path_to_main}/{path}", "r") as f:
                        child = Node(path, parent=included_tree_current)
                        converted_content, gphase = open_qasm_3_to_2(
                            f.read(), child, path_to_main, defined_gates, gphase
                        )
                    new_path = splitext(path)[0] + "_converted" + splitext(path)[1]
                    with open(f"{path_to_main}/{new_path}", "w") as f:
                        f.write(converted_content)
                    header_code += f"include '{new_path}';\n"

    elif instr_name in {"qubit", "bit"}:
        m = re.match(r"\s*(qu)?bit\s*\[\s*(\d+)\s*\]\s*([\w\d_]+)\s*", instr)
        if m:
            bit_type = "q" if m.group(1) else "c"
            instructions_code += f"{bit_type}reg {m.group(3)}[{m.group(2)}];\n"

    elif re.match(
        r"\s*([\w\d_]+)(\[.*?\])?\s*=\s*measure\s*([\w\d_]+)(\[.*?\])?\s*", instr
    ):
        m = re.match(
            r"\s*([\w\d_]+)(\[.*?\])?\s*=\s*measure\s*([\w\d_]+)(\[.*?\])?\s*", instr
        )
        if m:
            c, nb_c, q, nb_q = m.groups()
            if nb_c and nb_q:
                instructions_code += f"measure {q}{nb_q} -> {c}{nb_c};\n"
            else:
                instructions_code += f"measure {q} -> {c};\n"
    elif instr_name in qelib1_gates:
        header_code += add_qe_lib()
        instructions_code += instr + ";\n"
    elif instr_name in std_gates_3_to_2_map:
        converted_instr_name = std_gates_3_to_2_map[instr_name]
        instructions_code += (
            re.sub(r"\b" + instr_name + r"\b", converted_instr_name, instr) + ";\n"
        )
    # elif instr_name in std_gates_3:
    #    m = re.match(r"\s*(.*)", instr)
    #    if m:
    #        instructions_code += m.group(1) + ";\n"
    elif instr_name == "gate":
        m = re.match(
            r"\s*gate\s+(\w+)\s*(\(([^)]*)\))?\s*([\w\s,]*)\s*{\s*([^}]*)\s*}",
            instr,
            re.DOTALL,
        )
        if m:
            gate_name = m.group(1)
            params = f"({m.group(3)})" if m.group(3) else ""
            qubits = m.group(4).strip()
            body = m.group(5)

            defined_gates.add(gate_name)
            g_string = f"gate {gate_name}{params} {qubits} {{\n"
            g_instructions = filter(
                lambda i: not re.fullmatch(r"\s*", i), body.split(";")
            )
            for instruction in g_instructions:
                instruction = instruction.strip()
                i_code, h_code, gphase = convert_instruction_3_to_2(
                    instruction,
                    included_instr,
                    included_tree_current,
                    defined_gates,
                    path_to_main,
                    gphase,
                )
                g_string += " " * 4 + i_code  # Add indentation to body instructions
                header_code += h_code

            # Finalize the gate and add it to instructions_code
            instructions_code += g_string + "}\n"

    elif re.match(r"\s*if\s*\(.*?\)\s*.+", instr):
        m = re.match(r"(\s*if\s*\(.*?\))\s*(.+)", instr)
        if m:
            if_statement = m.group(1)
            nested_instr = m.group(2)
            i_code, h_code, gphase = convert_instruction_3_to_2(
                nested_instr,
                included_instr,
                included_tree_current,
                defined_gates,
                path_to_main,
                gphase,
            )
            instructions_code += if_statement + " " + i_code
            header_code += h_code
    elif instr_name in {"reset", "barrier"}:
        instructions_code += instr + ";\n"
    elif instr_name == "gphase":
        instr_match = re.match(r"gphase\((.*)\)\s*", instr)
        if instr_match:
            try:
                phase = float(instr_match.group(1))
            except ValueError:
                raise ValueError(
                    f"gphase can not be converted to float: {instr_match.group(1)}, {instr}"
                )
            gphase += phase
    else:
        gate = instr.split()[0].split("(")[0]
        if gate not in defined_gates:
            raise ValueError(
                f"Gates not defined/handled at the time of usage: {gate}, {instr_name}"
            )
        m = re.match(r"\s*(.*)", instr)
        if m:
            instructions_code += m.group(1) + ";\n"

    return instructions_code, header_code, gphase


@typechecked
def open_qasm_3_to_2(
    code: str,
    included_tree_current_node: Optional[Node] = None,
    path_to_file: Optional[str] = None,
    defined_gates: Optional[set[str]] = None,
    gphase: float = 0.0,
) -> tuple[str, float]:
    """Converts an OpenQASM 3.0 code back to OpenQASM 2.0.

    This function will also recursively go through the imported files to
    translate them too. It is a partial conversion (the ``opaque``, ``for``,
    ``switch``, and many others keywords are not handled) for helping
    building temporary bridges between different platforms using different
    versions.

    Args:
        code: String containing the OpenQASM 3.0 code.
        included_tree_current_node: Current Node in the file inclusion tree.
        path_to_file: Path to the location of the file from which the code is coming (useful for locating imports).
        defined_gates: Set of custom gates already defined.
        gphase: The global phase of a circuit, which is not handled in OpenQASM2.

    Returns:
        Converted OpenQASM code in the 2.0 version.

    Example:
        >>> qasm3_str = '''OPENQASM 3.0;
        ... qubit[2] q;
        ... bit[2] c;
        ... h q[0];
        ... cx q[0],q[1];
        ... c[0] = measure q[0];
        ... c[1] = measure q[1];
        ... '''
        >>> qasm_2, gphase = open_qasm_3_to_2(qasm3_str)
        >>> print(qasm_2)  # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];

    """
    if included_tree_current_node is None:
        included_tree_current_node = Node("initial_code")
    if path_to_file is None:
        path_to_file = "."
    if defined_gates is None:
        defined_gates = set()

    header_code = ""
    instructions_code = ""

    instructions = parse_openqasm_3_file(code)

    included_instructions = set()
    defined_gates.update(std_gates_3)

    for instr in instructions:
        i_code, h_code, gphase = convert_instruction_3_to_2(
            instr,
            included_instructions,
            included_tree_current_node,
            defined_gates,
            path_to_file,
            gphase,
        )
        header_code += h_code
        instructions_code += i_code
    gphase_code = f"// gphase {gphase}\n" if gphase != 0 else ""
    target_code = header_code + gphase_code + instructions_code

    return target_code, gphase


@typechecked
def parse_openqasm_3_file(code: str) -> list[str]:
    """Splits a complete OpenQASM 3 program into individual instructions.

    Args:
        code: The complete OpenQASM 3.0 program.

    Returns:
        List of instructions.

    Note:
        We do not check for correct syntax; it is assumed that the code is well-formed.
    """
    cleaned_code = re.sub(r"//.*?$|/\*.*?\*/", "", code, flags=re.DOTALL | re.MULTILINE)

    cleaned_code = cleaned_code.replace("\t", " ").strip()

    gate_matches = list(re.finditer(r"gate .*?}", cleaned_code, re.DOTALL))

    sanitized_start = (
        cleaned_code[: gate_matches[0].span()[0]] if gate_matches else cleaned_code
    )

    instructions = sanitized_start.split(";")

    for i in range(len(gate_matches)):
        instructions.append(cleaned_code[slice(*(gate_matches[i].span()))])
        instructions.extend(
            cleaned_code[
                gate_matches[i].span()[1] : (
                    None
                    if i == len(gate_matches) - 1
                    else gate_matches[i + 1].span()[0]
                )
            ].split(";")
        )

    instructions = [i.lstrip() for i in instructions]
    return list(filter(lambda i: i.strip() != "", instructions))


@typechecked
def open_qasm_file_conversion_3_to_2(path: str) -> tuple[str, float]:
    """Converts an OpenQASM code in a file from version 3.0 and 2.0.

    This function is a shorthand to initialize :func:`open_qasm_3_to_2` with the
    correct values.

    Args:
        path: Path to the file containing the OpenQASM 3.0 code, and eventual
            imports.

    Returns:
        Converted OpenQASM code in the 2.0 version.

    Examples:
        >>> example_dir = "examples/scripts/qasm_files/"
        >>> with open(example_dir + "main_converted.qasm", "r") as f:
        ...     print(f.read()) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 3.0;
        include 'include1_converted.qasm';
        include 'include2_converted.qasm';
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0],q[1];
        gate2 q[0];
        gate3 q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        >>> qasm_2, gphase = open_qasm_file_conversion_3_to_2(example_dir + "main_converted.qasm")
        >>> print(qasm_2) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 2.0;
        include 'include1_converted_converted.qasm';
        include 'include2_converted_converted.qasm';
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        gate2 q[0];
        gate3 q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        >>> with open(example_dir + "include1_converted_converted.qasm", "r") as f:
        ...     print(f.read()) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 2.0;
        include "qelib1.inc";
        gate gate2 a {
            u3(pi, -pi/2, pi/2) a;
        }
        >>> with open(example_dir + "include2_converted_converted.qasm", "r") as f:
        ...     print(f.read()) # doctest: +NORMALIZE_WHITESPACE
        OPENQASM 2.0;
        include "qelib1.inc";
        gate gate3 a, b {
            u3(0, -pi/2, pi/3) a;
            cz a, b;
        }

    """

    with open(path, "r") as f:
        code = f.read()
        return open_qasm_3_to_2(code, Node(path), str(Path(path).parent))
