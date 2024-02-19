qreg q[2];

gate my_csx a, b { 
    ctrl @ s a, b; 
}

my_csx q[0], q[1];