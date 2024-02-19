// Custom include file for Braket SDK that doesn't support U native gate
gate ggphase(lambda) a { phaseshift(lambda) a; y a; phaseshift(lambda) a; y a; }
gate p(lambda) a { phaseshift(lambda) a; }
gate U(theta, phi, lambda) q { ggphase((phi+lambda)/2) q; rz(lambda) q; ry(theta) q; rz(phi) q;}
gate sdg a { si a; }
gate tdg a { ti a; }
gate sx a { v a; }
gate sxdg a { vi a; }
gate phase(lambda) q { phaseshift(lambda) q; }
gate cx a, b { cnot a, b; }
gate cp(lambda) a, b { ctrl @ p(lambda) a, b; }
gate crx(theta) a, b { ctrl @ rx(theta) a, b; }
gate cry(theta) a, b { ctrl @ ry(theta) a, b; }
gate crz(theta) a, b { ctrl @ rz(theta) a, b; }
gate ch a, b { ctrl @ h a, b; }
gate ccx a, b, c { ccnot a, b, c; }
gate cu(theta, phi, lambda, gamma) a, b { p(gamma-theta/2) a; ctrl @ U(theta, phi, lambda) a, b; }
gate CX a, b { cx a, b; }
gate cphase(lambda) a, b { cphaseshift a, b; }
gate id a { i a; }
gate u1(lambda) q { U(0, 0, lambda) q; }
gate u2(phi, lambda) q { ggphase(-(phi+lambda+pi/2)/2); U(pi/2, phi, lambda) q; }
gate u3(theta, phi, lambda) q { ggphase(-(phi+lambda+theta)/2); U(theta, phi, lambda) q; }