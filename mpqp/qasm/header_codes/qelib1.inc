// file: qelib1.inc, OpenQASM2 standard header
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate u1(lambda) q { U(0,0,lambda) q; }
gate u0(gamma) q { U(0,0,0) q; }
gate cx c,t { CX c,t; }
gate id a { U(0,0,0) a; }
gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate p(lambda) q { U(0,0,lambda) q; }
gate x a { u3(pi,0,pi) a; }
gate y a { u3(pi,pi/2,pi/2) a; }
gate z a { u1(pi) a; }
gate h a { u2(0,pi) a; }
gate s a { u1(pi/2) a; }
gate sdg a { u1(-pi/2) a; }
gate t a { u1(pi/4) a; }
gate tdg a { u1(-pi/4) a; }
gate sx a { sdg a; h a; sdg a; }
gate sxdg a { s a; h a; s a; }
gate rx(theta) a { u3(theta,-pi/2,pi/2) a; }
gate ry(theta) a { u3(theta,0,0) a; }
gate rz(phi) a { u1(phi) a; }
gate swap a,b { cx a,b; cx b,a; cx a,b; }
gate cz a,b { h b; cx a,b; h b; }
gate cy a,b { sdg b; cx a,b; s b; }
gate ch a,b {
    h b; sdg b;
    cx a,b;
    h b; t b;
    cx a,b;
    t b; h b; s b; x b; s a;
}
gate ccx a,b,c
{
    h c;
    cx b,c; tdg c;
    cx a,c; t c;
    cx b,c; tdg c;
    cx a,c; t b; t c; h c;
    cx a,b; t a; tdg b;
    cx a,b;
}
gate crz(lambda) a,b
{
    u1(lambda/2) b;
    cx a,b;
    u1(-lambda/2) b;
    cx a,b;
}
gate cu1(lambda) a,b
{
    u1(lambda/2) a;
    cx a,b;
    u1(-lambda/2) b;
    cx a,b;
    u1(lambda/2) b;
}
gate cu3(theta,phi,lambda) c, t
{
    // implements controlled-U(theta,phi,lambda) with  target t and control c
    u1((lambda-phi)/2) t;
    cx c,t;
    u3(-theta/2,0,-(phi+lambda)/2) t;
    cx c,t;
    u3(theta/2,phi,0) t;
}
gate cswap a,b,c
{
    cx c,b;
    ccx a,b,c;
    cx c,b;
}
gate crx(lambda) a,b
{
    u1(pi/2) b;
    cx a,b;
    u3(-lambda/2,0,0) b;
    cx a,b;
    u3(lambda/2,-pi/2,0) b;
}
gate cry(lambda) a,b
{
    ry(lambda/2) b;
    cx a,b;
    ry(-lambda/2) b;
    cx a,b;
}
gate cp(lambda) a,b
{
  p(lambda/2) a;
  cx a,b;
  p(-lambda/2) b;
  cx a,b;
  p(lambda/2) b;
}
gate cu(theta,phi,lambda,gamma) c, t
{ p(gamma) c;
  p((lambda+phi)/2) c;
  p((lambda-phi)/2) t;
  cx c,t;
  u(-theta/2,0,-(phi+lambda)/2) t;
  cx c,t;
  u(theta/2,phi,0) t;
}