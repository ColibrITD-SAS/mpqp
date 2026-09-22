gate rxx(theta) a, b {
    u3(pi/2, theta, 0) a;
    h b;
    cx a,
    b;
    u1(-theta) b;
    cx a, b;
    h b;
    u2(-pi, pi-theta) a;
}
