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
