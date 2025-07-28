// Conversion of Qiskit specific gates to OpenQASM2 instructions

gate ch a,b {
    s b;
    h b;
    t b;
    cx a, b;
    p(- pi*0.25) b;
    h b;
    p(- pi*0.5) b;
}

gate sx a {
    rx(pi/2) a; 
    gphase(pi/4);
}

gate sy a {
    ry(pi/2) a;
    gphase(pi/4);
}

gate sz a {
    rz(pi/2) a;
    gphase(pi/4);
}

gate sxdg a {
    s a;
    h a;
    s a;
    gphase(-pi/4);
}

gate csx a, b {
    h b; 
    u3(0, 0, pi/4) a; 
    cx a,b;
    u3(0, 0, -pi/4) b; 
    cx a,b;
    u3(0, 0, pi/4) b;
    h b;
}

gate cy a, b {
    p(- pi*0.5) b; 
    cx a, b; 
    s b; 
}

gate tdg a {
    p(- pi*0.25) a;
}

gate crx(theta) a, b {
    u3(0, 0, pi/2) b;
    cx a,b;
    u3(-theta/2,0,0) b;
    cx a,b;
    u3(theta/2,-pi/2,0) b;
}

gate cry(lambda) a, b {
    u3(lambda/2,0,0) b; 
    cx a,b;
    u3(-lambda/2,0,0) b; 
    cx a,b;
}

gate crz(lambda) a, b {
    rz(lambda/2) b;
    cx a,b;
    rz(-lambda/2) b; 
    cx a,b;
}

gate u1(theta) a {
    u3(0, 0, theta) a;
}

gate u2(theta, phi) a {
    u3(pi/2, theta, phi) a;
}
 
gate cu(theta,phi,lambda,gamma) c, t {
    u(0, 0, gamma) c;
    u(0, 0, lambda/2 + phi/2) c;
    u(0, 0, lambda/2 - phi/2) t;
    cx c,t;
    u(-theta/2,0,-phi/2 - lambda/2) t;
    cx c,t;
    u(theta/2,phi,0) t;
}

gate cu1(lambda) a, b {
    u1(lambda/2) a; 
    cx a,b;
    u1(-lambda/2) b; 
    cx a,b;
    u1(lambda/2) b;
}

gate cswap a, b, c {
    cx c,b;
    ccx a,b,c;
    cx c,b;
}

gate rxx(a) b, c {
    h b;
    h c;
    cx b, c;
    rz(a) c;
    cx b, c;
    h c;
    h b;
}

gate ryy(a) b, c {
    rx(pi/2) b;
    rx(pi/2) c;
    cx b, c;
    rz(a) c;
    cx b, c;
    rx(-pi/2) b;
    rx(-pi/2) c;
}

gate rzz(a) b, c {
    cx b, c;
    rz(a) c;
    cx b, c;
}

gate rccx a,b,c {
    u2(0,pi) c;
    u1(pi/4) c;
    cx b, c;
    u1(-pi/4) c;
    cx a, c;
    u1(pi/4) c;
    cx b, c;
    u1(-pi/4) c;
    u2(0,pi) c;
}