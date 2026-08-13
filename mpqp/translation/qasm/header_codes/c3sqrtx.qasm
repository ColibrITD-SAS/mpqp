gate c3sqrtx a,b,c,d {
    h d; cu1(pi/8) a,d; h d;
    cx a,b;
    h d; cu1(-pi/8) b,d; h d;
    cx a,b;
    h d; cu1(pi/8) b,d; h d;
    cx b,c;
    h d; cu1(-pi/8) c,d; h d;
    cx a,c;
    h d; cu1(pi/8) c,d; h d;
    cx b,c;
    h d; cu1(-pi/8) c,d; h d;
    cx a,c;
    h d; cu1(pi/8) c,d; h d;
}
