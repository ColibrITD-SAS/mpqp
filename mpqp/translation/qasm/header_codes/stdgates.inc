// file: stdgates.inc, OpenQASM3 standard header
gate p(lambda) a { ctrl @ gphase(lambda) a; }
gate x a { U(pi, 0, pi) a; gphase(-pi/2);}
gate y a { U(pi, pi/2, pi/2) a; gphase(-pi/2);}
gate z a { p(pi) a; }
gate h a { U(pi/2, 0, pi) a; gphase(-pi/4);}
gate s a { pow(1/2) @ z a; }
gate sdg a { inv @ pow(1/2) @ z a; }
gate t a { pow(1/2) @ s a; }
gate tdg a { inv @ pow(1/2) @ s a; }
gate sx a { pow(1/2) @ x a; }
gate rx(theta) a { U(theta, -pi/2, pi/2) a; gphase(-theta/2);}
gate ry(theta) a { U(theta, 0, 0) a; gphase(-theta/2);}
gate rz(lambda) a { gphase(-lambda/2); U(0, 0, lambda) a; }
gate cx a, b { ctrl @ x a, b; }
gate cy a, b { ctrl @ y a, b; }
gate cz a, b { ctrl @ z a, b; }
gate cp(lambda) a, b { ctrl @ p(lambda) a, b; }
gate crx(theta) a, b { ctrl @ rx(theta) a, b; }
gate cry(theta) a, b { ctrl @ ry(theta) a, b; }
gate crz(theta) a, b { ctrl @ rz(theta) a, b; }
gate ch a, b { ctrl @ h a, b; }
gate swap a, b { cx a, b; cx b, a; cx a, b; }
gate ccx a, b, c { ctrl @ ctrl @ x a, b, c; }
gate cswap a, b, c { ctrl @ swap a, b, c; }
gate cu(theta, phi, lambda, gamma) a, b { p(gamma-theta/2) a; ctrl @ U(theta, phi, lambda) a, b; }
gate CX a, b { ctrl @ U(pi, 0, pi) a, b; }
gate phase(lambda) q { U(0, 0, lambda) q; }
gate cphase(lambda) a, b { ctrl @ phase(lambda) a, b; }
gate id a { U(0, 0, 0) a; }
gate u1(lambda) q { U(0, 0, lambda) q; }
gate u2(phi, lambda) q { gphase(-(phi+lambda+pi/2)/2); U(pi/2, phi, lambda) q; }
gate u3(theta, phi, lambda) q { gphase(-(phi+lambda+theta)/2); U(theta, phi, lambda) q; }