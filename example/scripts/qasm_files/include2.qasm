OPENQASM 2.0;
gate gate3 a, b {
    U(0, -pi/2, pi/3) a;
    cz a, b;
}