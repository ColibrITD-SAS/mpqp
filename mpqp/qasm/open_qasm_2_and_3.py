import os
from pathlib import Path
import re
from os.path import splitext
from enum import Enum

from warnings import warn
from textwrap import dedent
from anytree import Node, PreOrderIter
from typeguard import typechecked

from mpqp.tools.errors import InstructionParsingError


class Instr(Enum):
    """Special instruction of which the definition needs to included in the
    header of the file."""

    STD_LIB = 0
    CSX = 1
    U0 = 2
    CU3 = 3
    SXDG = 4
    RZZ = 5
    RXX = 6
    RCCX = 7
    RC3X = 8
    C3X = 9
    C4X = 10
    C3SQRTX = 11
    OQASM2_ALL_STDGATES = 12
    OQASM3_ALL_STDGATES = 13
    BRAKET_CUSTOM_INCLUDE = 14


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

    # 3M-TODO: if run from outside the mpqp folder, the following line will fuck things up, fix it
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
        code: The complete OpenQASM 2.0 program, we do not check for correct syntax, it is assumed that the code is
            well formed.

    Returns:
        List of instructions.
    """
    # 6M-TODO: deal with comments, for the moment we remove them all

    # removing comment
    cleaned_code = "".join([loc.split("//")[0] for loc in code.split("\n")])

    cleaned_code = cleaned_code.replace("\t", " ").strip()

    # gate must be registered as a single instruction
    gate_matches = list(re.finditer(r"gate .*?}", cleaned_code))
    # we start with adding all the instructions up to the first gate definition
    sanitized_start = (
        cleaned_code[: gate_matches[0].span()[0]] if gate_matches else cleaned_code
    )
    instructions = sanitized_start.split(";")

    # we then add turn by turn the following gate definition, all the
    # instructions between it and the next one, and so on...
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

    return list(filter(lambda i: i != "", instructions))


@typechecked
def convert_instruction_2_to_3(
    instr: str,
    included_instr: set[Instr],
    included_tree_current_node: Node,
    defined_gates: set[str],
    path_to_main: str = ".",
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
        included_tree_current_node: Current Node in the file inclusion tree.
        defined_gates: Set of custom gates already defined.
        path_to_main: Path to the main folder from which include paths are
            described.

    Returns:
        The upgraded instruction and the potential code to add in the header as
        the second element.
    """

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
            if is_path_in_ancestors(path, included_tree_current_node):
                raise RuntimeError("Circular dependency detected.")
            # Convert the file included, add it to the inclusion tree,
            # and create a new file and include it in the converted code
            if not is_path_in_tree(path, included_tree_current_node):
                with open(f"{path_to_main}/{path}", "r") as f:
                    child = Node(path, parent=included_tree_current_node)
                    converted_content = open_qasm_2_to_3(
                        f.read(), child, path_to_main, defined_gates
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
        warn(
            dedent(
                """OpenQASMTranslationWarning: 
                There is a phase e^(i(a+c)/2) difference between U(a,b,c) gate in 2.0 and 3.0.
                We handled that for you by adding the extra phase at the right place. 
                Be careful if you want to create a control gate from this circuit/gate, the
                phase can become non-global."""
            )
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
            header_code += qasm_code(new_instr)
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
                included_tree_current_node,
                defined_gates,
                path_to_main,
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
            included_tree_current_node,
            defined_gates,
            path_to_main,
        )
        instructions_code += if_statement + i_code + ";\n"
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
    included_tree_current_node: Node = Node("initial_code"),
    path_to_file: str = ".",
    defined_gates: set[str] = set(),
) -> str:
    """Converts an OpenQASM code from version 2.0 and 3.0.

    It is a partial conversion (mainly circuit structure) for helping building
    temporary bridges between different platforms that use different versions.

    Args:
        code: String containing the OpenQASM 2.0 code and instructions.
        included_tree_current_node: Current Node in the file inclusion tree.
        path_to_file: Path to the location of the file from which the code is
            coming (useful for locating imports).
        defined_gates: Set of custom gates already defined.

    Returns:
        Converted OpenQASM code in the 3.0 version.

    Example:
        >>> qasm2_str = '''\\
        ... OPENQASM 2.0;
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0],q[1];
        ... measure q[0] -> c[0];
        ... measure q[1] -> c[1];
        ... '''
        >>> open_qasm_2_to_3(qasm2_str)
        '''OPENQASM 3.0;
        include 'stdgates.inc';
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0],q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        '''
    """

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
        )
        header_code += h_code
        instructions_code += i_code

    target_code = header_code + "\n" + instructions_code

    return target_code


@typechecked
def open_qasm_file_conversion_2_to_3(path: str) -> str:
    """Converts an OpenQASM code in a file from version 2.0 and 3.0.

    It is a partial conversion (mainly circuit structure) for helping building
    temporary bridges between different platforms that use different versions.

    Args:
        path: Path to the file containing the OpenQASM 2.0 code, and eventual
            imports.

    Returns:
        Converted OpenQASM code in the 3.0 version.

    Example:
        >>> example_dir = "example/qasm_files/"
        >>> with open(example_dir + "main.qasm", "r") as f:
        ...     print(f.read())
        '''OPENQASM 2.0;
        include "include1.qasm";
        include "include2.qasm";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        gate2 q[0];
        gate3 q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];'''
        >>> print(open_qasm_file_conversion_2_to_3(example_dir + "main.qasm"))
        '''OPENQASM 3.0;
        include 'include1_converted.qasm';
        include 'include2_converted.qasm';
        include 'stdgates.inc';
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0],q[1];
        gate2 q[0];
        gate3 q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];'''
        >>> with open(example_dir + "include1_converted.qasm", "r") as f:
        ...     print(f.read())
        '''OPENQASM 3.0;
        include 'stdgates.inc';
        gate gate2 a {
            u3(pi, -pi/2, pi/2) a;
        }'''
        >>> with open(example_dir + "include2_converted.qasm", "r") as f:
        ...     print(f.read())
        '''OPENQASM 3.0;
        include 'stdgates.inc';
        gate gate3 a, b {
            u3(0, -pi/2, pi/3) a;
            cz a, b;
        }'''
    """

    with open(path, "r") as f:
        code = f.read()
        return open_qasm_2_to_3(code, Node(path), str(Path(path).parent))


@typechecked
def open_qasm_hard_includes(
    code: str,
    included_files: set[str],
    path_to_file: str = "./",
    is_openqasm_header_included: bool = False,
) -> str:
    r"""Converts an OpenQASM code (2.0 and 3.0) to use no includes, but writes
    every instruction in previously included files, directly in the code
    returned.

    Example:
        >>> examples_folder = "tests/qasm/qasm_examples"
        >>> filename = examples_folder + "/with_include.qasm"
        >>> with open(filename) as f:
        ...     print(open_qasm_hard_includes(f.read(), {filename}).strip("\n"))
        '''gate csx a, b {
            ctrl @ sx a, b;
        }'''

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
    """
    lines = code.split("\n")
    converted_code = []

    for line in lines:
        if "include" in line:
            line_array = line.split()

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


@typechecked
def is_path_in_tree(path: str, any_node: Node):
    """Checks if a path is already present in a node of the tree represented by
    one of his nodes (in parameter).

    Args:
        path: Element to search in the tree.
        any_node: One node of the tree on which we want to search.

    Returns:
        ``True`` if the path is the name of one of the nodes of the tree.
    """
    return any([path in node.name for node in PreOrderIter(any_node.root)])


@typechecked
def is_path_in_ancestors(path: str, node: Node):
    """Checks if a path is already present in an ancestor of the node in
    parameter.

    Args:
        path: Element to search in the tree.
        node: The node for which we want to search in his ancestors.

    Returns:
        ``True`` if the path is the name of one of the ancestors of the node/
    """
    return any([path in node.name for node in node.ancestors])
