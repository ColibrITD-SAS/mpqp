OPENQASM 3.0;
include "stdgates.inc";

gate gate3 a, b {
    u3(0, -pi/2, pi/3) a;
    cz a, b;
}
