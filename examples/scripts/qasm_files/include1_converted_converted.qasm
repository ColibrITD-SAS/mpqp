OPENQASM 2.0;
include "qelib1.inc";
gate gate2 a {
    u3(pi, -pi/2, pi/2) a;
}
