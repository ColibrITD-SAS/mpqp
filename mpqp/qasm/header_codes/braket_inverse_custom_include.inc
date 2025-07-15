// Translation of Braket SDK gates to classic instructions on OpenQASM2
gate cnot a, b { cx a, b; }

gate ccnot a, b, c { ccx a, b, c; }

gate phaseshift(a) b { p(a) b; }

gate cphaseshift(a) b, c { cp(a) b, c; }

gate cphaseshift00(a) b, c { 
    x b;
    x c;
    cp(a) b, c;
    x b;
    x c; 
}

gate cphaseshift01(a) b, c {
    x b; 
    cp(a) b, c; 
    x b; 
}

gate cphaseshift10(a) b, c {
    x c; 
    cp(a) b, c; 
    x c; 
}
gate cswap a, b, c 
{ 
    ccx a, b, c;
    ccx c, a, b;
    ccx a, b, c;
}

gate cy a, b 
{ 
    p(pi*0.5) a; 
    cx a, b; 
    s a;    
}

gate ecr a, b {
    h b;
    cx a, b;
    rz(pi/4) b;
    cx a, b;
    h b;
    x a;
    h b;
    cx a, b;
    rz(- pi/4) b;
    cx a, b;
    h b;
}

gate iswap a, b 
{
    s a;
    s b;
    h a;
    cx a, b;
    cx b, a;
    h b;
}

// pswap

gate si a { p(- pi*0.5) a; }

gate ti a { p(- pi*0.25) a; }

gate v a { 
    rx(pi/2) a; 
    gphase(pi/4);
}

gate vi a {
    rx(- pi/2) a;
    gphase(- pi/4);
}

gate xx(a) b, c {
    h b;
    h c;
    cx b, c;
    rz(a) c;
    cx b, c;
    h c;
    h b;
}

gate xy(a) c, d {
    rz(-a) c;
    p(- pi/2) d;
    s c;
    rx(pi/2) d;
    gphase(pi/4);
    s d;
    cx d, c;
    ry(-a/2) c;
    ry(-a/2) d;
    cx d, c;
    p(- pi/2) c;
    p(- pi/2) d;
    rz(a) c;
    rx(-pi/2) d;
    gphase(-pi/4);
    s d;
}

gate yy(a) b, c {
    rx(pi/2) b;
    rx(pi/2) c;
    cx b, c;
    rz(a) c;
    cx b, c;
    rx(-pi/2) b;
    rx(-pi/2) c;
}

gate zz(a) b, c {
    cx b, c;
    rz(a) c;
    cx b, c;
}

gate gpi(a) b {
    rz(a) b;
    x b;
    rz(-a) b;
}

gate gpi2(a) b {
    rz(-a) b;
    rx(pi/2) b;
    rz(a) b;
}

gate ms(a, b, c) q0, q1 {
    rz(a) q0;
    rz(b) q1;
    h q0;
    h q1;
    cx q0, q1;
    rz(c) q1;
    cx q0, q1;
    h q1;
    h q0;
} 

gate i a { id a; }