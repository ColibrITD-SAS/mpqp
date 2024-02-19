QASM converter
==================

.. code-block:: python

    from mpqp.qasm import *

To allow interoperability with different providers, we leverage the OpenQASM
standard, mainly adopted by the community and SDKs. We describe here how to
convert OpenQASM from a version to another, and how to generate supported
providers' circuits from OpenQASM code.

.. note::
    To learn more about how we generate OpenQASM code from a :class:`QCircuit`, 
    have a look at the :meth:`to_qasm2<mpqp.core.circuit.QCircuit.to_qasm2>` and 
    :meth:`to_qasm3<mpqp.core.circuit.QCircuit.to_qasm3>` methods.

From OpenQASM2.0 to OpenQASM3.0
-------------------------------

Recently, a new version of OpenQASM (3.0) has been released by conjoint members
of IBM Quantum, AWS Quantum Computing, Zapata Computing, Zurich Instruments and
University of Oxford. This version extends the 2.0 one, adding some advanced
features, and modifying parts of syntax and grammar, making some part of
OpenQASM2.0 not fully retro-compatible.

Therefore, we developed an algorithm able to translate an OpenQASM 2.0 code into
a OpenQASM 3.0, translating also recursively the included files in the code. One
can use the function :func:`open_qasm_2_to_3` to translate a code given in
parameter as a string. The other facultative parameters are used for recursive
calls of the function (when having to translate included files for instance),
but are not relevant from a user point of view, expect the parameter
``path_to_file``, useful for locating imports. The translator converts also
imported files, and includes the new ones in the converted code.

.. autofunction:: mpqp.qasm.open_qasm_2_and_3.open_qasm_2_to_3

During the translation, other functions are called to make the implementation
more modular. The first step is to parse the OpenQASM 2.0 code, in order to
extract the header and the instructions, and this is done by the function
:func:`parse_openqasm_2_file<mpqp.qasm.open_qasm_2_and_3.parse_openqasm_2_file>`.

.. autofunction:: mpqp.qasm.open_qasm_2_and_3.parse_openqasm_2_file

Once the file is parsed and divided in list of independent instructions, we
convert each instruction into the OpenQASM 3.0 equivalent. When needed
additional imports or definitions are added to a common header.

.. autofunction:: mpqp.qasm.open_qasm_2_and_3.convert_instruction_2_to_3

We can also give the path of the file instead of the string, and it will create
a new file with the converted code. The imported files will also be converted
and corresponding new files will be created and imported in the new code.

.. autofunction:: mpqp.qasm.open_qasm_2_and_3.open_qasm_file_conversion_2_to_3

.. _hard-include:

We also give the possibility to directly write in the OpenQASM code all the
content of included files. This is helpful when a parser or provider does not
support the ``include`` statement.

.. autofunction:: mpqp.qasm.open_qasm_2_and_3.open_qasm_hard_includes


From OpenQASM to Qiskit
-----------------------

The main object used to perform quantum computations in Qiskit is the
``QuantumCircuit``. Qiskit naturally supports OpenQASM 2.0 to instantiate a
circuit. One can remark that few remote devices also support OpenQASM 3.0 code,
this is not generalized yet to the whole library and device. We call the
function
:func:`qasm2_to_QuantumCircuit<mpqp.qasm.qasm_to_qiskit.qasm2_to_QuantumCircuit>`
to generate the circuit from the qasm code.

.. autofunction:: mpqp.qasm.qasm_to_qiskit.qasm2_to_QuantumCircuit

From OpenQASM to myQLM
----------------------

The myQLM library allows the user to instantiate a myQLM ``Circuit`` from an
OpenQASM 2.0 code. MyQLM is able to parse most of the standard gates, and allows
us to complete the missing gates by linking them to already defined ones. We
call the function
:func:`qasm2_to_Circuit<mpqp.qasm.qasm_to_myqlm.qasm2_to_Circuit>` to generate
the circuit from the qasm code.

.. autofunction:: mpqp.qasm.qasm_to_myqlm.qasm2_to_myqlm_Circuit

From OpenQASM to Braket
-----------------------

Amazon Braket made the choice to directly support a subset of OpenQASM 3.0 for
gate-based devices and simulators. In fact, Braket supports a set of data types,
statements and pragmas (specific to Braket) for OpenQASM 3.0, sometimes with a
different syntax.

Braket Circuit parser does not support for the moment the OpenQASM 3.0 native
operations (``U`` and ``gphase``) but allows to define custom gates using a
combination of supported standard gates (``rx``, ``ry``, ``rz``, ``cnot``,
``phaseshift`` for instance). Besides, the inclusion of files is not yet handled
by Braket library. Therefore, we temporarily created a custom file to *hard-*
include (see :ref:`above<hard-include>`) directly in the OpenQASM 3.0 code, to
be sure the parser and interpreter have all definitions in there. We also
hard-include all included files in the OpenQASM 3.0 code inputted for
conversion.

.. note::
    In the custom hard-imported file for native and standard gate redefinitions, 
    we use ``ggphase`` to define the global phase, instead of the OpenQASM 3.0 
    keyword ``gphase``, which is already used and protected by Braket.

.. autofunction:: mpqp.qasm.qasm_to_braket.qasm3_to_braket_Circuit

If needed, one can also generate a Braket ``Program`` from an OpenQASM 3.0 input
string using the function below. However, in this case, the program parser does
not need to redefine the native gates, and thus only performing a hard import of
standard gates and other included file is sufficient. However, note that a
``Program`` cannot be used to retrieve the statevector and expectation value in
Braket.

.. autofunction:: mpqp.qasm.qasm_to_braket.qasm3_to_braket_Program
