"""Examples of OpenQASM conversion from 2.0 to 3.0"""

from mpqp.qasm import open_qasm_2_to_3

print("-------------------------")
print("-------------------------")

qasm2_1 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];"""

print(open_qasm_2_to_3(qasm2_1))

print("-------------------------")
print("-------------------------")

qasm2_2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c0[1];
creg c1[1];
creg c2[1];

u3(0.3,0.2,0.1) q[0];
h q[1];
cx q[1],q[2];
barrier q;
cx q[0],q[1];
h q[0];
measure q[0] -> c0[0];
measure q[1] -> c1[0];
rxx(0.4) q[0], q[1];
rzz(0.2) q[1], q[2];
measure q[2] -> c2[0];"""

print(open_qasm_2_to_3(qasm2_2))

print("-------------------------")
print("-------------------------")

qasm2_3 = """OPENQASM 2.0;
include "qasm_files/test_qasm.inc";
u3(0.3,0.2,0.1) q[0];
h q[1];
cx q[1],q[0];

barrier q;"""

print(open_qasm_2_to_3(qasm2_3, path_to_file="mpqp/example"))

print("-------------------------")
print("-------------------------")

qasm2_4 = """OPENQASM 2.0;
include "qelib1.inc";
gate hello(theta) a, b {
    h b;
    cu1(theta) a, b;
    u(-theta) b;
    cx a, b;
    h b;
    u2(-pi, pi-theta) a;
}
u3(0.3,0.2,0.1) q[0];
h q[1];
cx q[1],q[0];

barrier q;"""

print(open_qasm_2_to_3(qasm2_4))
