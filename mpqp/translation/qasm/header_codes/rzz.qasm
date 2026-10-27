gate rzz(theta) a,b {
    cx a,b;
    u1(theta) b;
    cx a,b;
}
