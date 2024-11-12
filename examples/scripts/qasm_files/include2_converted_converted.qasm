OPENQASM 2.0;
include "qelib1.inc";
gate gate3 a, b {
    u3(0, -pi/2, pi/3) a;
    cz a, b;
}
