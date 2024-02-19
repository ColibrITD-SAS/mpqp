qreg q[2];

my_csx q[0], q[1];

gate my_csx a, b { 
    ctrl @ s a, b; 
}