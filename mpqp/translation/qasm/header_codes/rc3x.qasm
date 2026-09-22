gate rc3x a,b,c,d {
    u2(0,pi) d;
    u1(pi/4) d;
    cx c,d;
    u1(-pi/4) d;
    u2(0,pi) d;
    cx a,d;
    u1(pi/4) d;
    cx b,d;
    u1(-pi/4) d;
    cx a,d;
    u1(pi/4) d;
    cx b,d;
    u1(-pi/4) d;
    u2(0,pi) d;
    u1(pi/4) d;
    cx c,d;
    u1(-pi/4) d;
    u2(0,pi) d;
}
