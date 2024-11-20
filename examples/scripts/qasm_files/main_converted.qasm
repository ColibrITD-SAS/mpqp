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