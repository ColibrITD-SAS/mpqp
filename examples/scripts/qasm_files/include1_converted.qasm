OPENQASM 3.0;
include "stdgates.inc";

gate gate2 a {
    u3(pi, -pi/2, pi/2) a;
}
