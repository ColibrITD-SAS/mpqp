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