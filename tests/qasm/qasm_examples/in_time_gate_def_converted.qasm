
qubit[2] q;
gate my_csx a, b {
    ctrl @ s a, b;
}
my_csx q[0], q[1];
