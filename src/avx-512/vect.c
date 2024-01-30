#include "vect.h"


// r = a + b mod p (addition-based fast reduction)
void add_mod_384_8x1w(__m512i *r, const __m512i *a, const __m512i *b, 
                      const uint64_t *sp)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(sp[0]), p1 = VSET1(sp[1]), p2 = VSET1(sp[2]);
  const __m512i p3 = VSET1(sp[3]), p4 = VSET1(sp[4]), p5 = VSET1(sp[5]);
  const __m512i p6 = VSET1(sp[6]), p7 = VSET1(sp[7]), bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask;

  // r = a + b
  r0 = VADD(a0, b0); r1 = VADD(a1, b1); r2 = VADD(a2, b2); r3 = VADD(a3, b3);
  r4 = VADD(a4, b4); r5 = VADD(a5, b5); r6 = VADD(a6, b6); r7 = VADD(a7, b7);

  // r = r - p
  r0 = VSUB(r0, p0); r1 = VSUB(r1, p1); r2 = VSUB(r2, p2); r3 = VSUB(r3, p3); 
  r4 = VSUB(r4, p4); r5 = VSUB(r5, p5); r6 = VSUB(r6, p6); r7 = VSUB(r7, p7);

  // get sign mask 
  r1 = VADD(r1, VSRA(r0, BRADIX));
  r2 = VADD(r2, VSRA(r1, BRADIX));
  r3 = VADD(r3, VSRA(r2, BRADIX));
  r4 = VADD(r4, VSRA(r3, BRADIX));
  r5 = VADD(r5, VSRA(r4, BRADIX));
  r6 = VADD(r6, VSRA(r5, BRADIX));
  r7 = VADD(r7, VSRA(r6, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(r7, 63);
  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 
  r4 = VADD(r4, VAND(p4, smask)); r5 = VADD(r5, VAND(p5, smask)); 
  r6 = VADD(r6, VAND(p6, smask)); r7 = VADD(r7, VAND(p7, smask)); 

  // carry propagation 
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4, VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5, VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6, VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7, VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// r = a - b mod p (addition-based fast reduction)
void sub_mod_384_8x1w(__m512i *r, const __m512i *a, const __m512i *b, 
                      const uint64_t *sp)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(sp[0]), p1 = VSET1(sp[1]), p2    = VSET1(sp[2]);
  const __m512i p3 = VSET1(sp[3]), p4 = VSET1(sp[4]), p5    = VSET1(sp[5]);
  const __m512i p6 = VSET1(sp[6]), p7 = VSET1(sp[7]), bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask;

  // r = a - b
  r0 = VSUB(a0, b0); r1 = VSUB(a1, b1); r2 = VSUB(a2, b2); r3 = VSUB(a3, b3);
  r4 = VSUB(a4, b4); r5 = VSUB(a5, b5); r6 = VSUB(a6, b6); r7 = VSUB(a7, b7); 

  // get sign mask 
  r1 = VADD(r1, VSRA(r0, BRADIX));
  r2 = VADD(r2, VSRA(r1, BRADIX));
  r3 = VADD(r3, VSRA(r2, BRADIX));
  r4 = VADD(r4, VSRA(r3, BRADIX));
  r5 = VADD(r5, VSRA(r4, BRADIX));
  r6 = VADD(r6, VSRA(r5, BRADIX));
  r7 = VADD(r7, VSRA(r6, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(r7, 63);
  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 
  r4 = VADD(r4, VAND(p4, smask)); r5 = VADD(r5, VAND(p5, smask)); 
  r6 = VADD(r6, VAND(p6, smask)); r7 = VADD(r7, VAND(p7, smask)); 

  // carry propagation 
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4, VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5, VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6, VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7, VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// integer multiplication (Karatsuba)
void mul_384_8x1w_v1(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  __m512i ta0, ta1, ta2, ta3;
  __m512i tb0, tb1, tb2, tb3;
  __m512i z0  = VZERO, z1  = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4  = VZERO, z5  = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8  = VZERO, z9  = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i z12 = VZERO, z13 = VZERO, z14 = VZERO, z15 = VZERO;
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO, y15 = VZERO;
  __m512i m0  = VZERO, m1  = VZERO, m2  = VZERO, m3  = VZERO;
  __m512i m4  = VZERO, m5  = VZERO, m6  = VZERO, m7  = VZERO;

  // compute zL(z0-z7) by aL(a0-a3) * bL(b0-b4)

  z0 = VMACLO(z0, a0, b0);
  y0 = VMACHI(y0, a0, b0);
  y0 = VSHL(y0, BALIGN);

  z1 = VMACLO(y0, a0, b1); z1 = VMACLO(z1, a1, b0);
  y1 = VMACHI(y1, a0, b1); y1 = VMACHI(y1, a1, b0);
  y1 = VSHL(y1, BALIGN);

  z2 = VMACLO(y1, a0, b2); z2 = VMACLO(z2, a1, b1); z2 = VMACLO(z2, a2, b0);
  y2 = VMACHI(y2, a0, b2); y2 = VMACHI(y2, a1, b1); y2 = VMACHI(y2, a2, b0);
  y2 = VSHL(y2, BALIGN);

  z3 = VMACLO(y2, a0, b3); z3 = VMACLO(z3, a1, b2); z3 = VMACLO(z3, a2, b1); 
  z3 = VMACLO(z3, a3, b0);
  y3 = VMACHI(y3, a0, b3); y3 = VMACHI(y3, a1, b2); y3 = VMACHI(y3, a2, b1); 
  y3 = VMACHI(y3, a3, b0);
  y3 = VSHL(y3, BALIGN);

  z4 = VMACLO(y3, a1, b3); z4 = VMACLO(z4, a2, b2); z4 = VMACLO(z4, a3, b1);
  y4 = VMACHI(y4, a1, b3); y4 = VMACHI(y4, a2, b2); y4 = VMACHI(y4, a3, b1);
  y4 = VSHL(y4, BALIGN);

  z5 = VMACLO(y4, a2, b3); z5 = VMACLO(z5, a3, b2);
  y5 = VMACHI(y5, a2, b3); y5 = VMACHI(y5, a3, b2);
  y5 = VSHL(y5, BALIGN);

  z6 = VMACLO(y5, a3, b3);
  y6 = VMACHI(y6, a3, b3);
  y6 = VSHL(y6, BALIGN);

  z7 = y6;

  // compute zH(z8-z15) by aH(a4-a7) * bH(b4-b7)

  z8 = VMACLO(z8, a4, b4);
  y8 = VMACHI(y8, a4, b4);
  y8 = VSHL(y8, BALIGN);

  z9 = VMACLO(y8, a4, b5); z9 = VMACLO(z9, a5, b4);
  y9 = VMACHI(y9, a4, b5); y9 = VMACHI(y9, a5, b4);
  y9 = VSHL(y9, BALIGN);

  z10 = VMACLO(y9,  a4, b6); z10 = VMACLO(z10, a5, b5); 
  z10 = VMACLO(z10, a6, b4);
  y10 = VMACHI(y10, a4, b6); y10 = VMACHI(y10, a5, b5); 
  y10 = VMACHI(y10, a6, b4);
  y10 = VSHL(y10, BALIGN);

  z11 = VMACLO(y10, a4, b7); z11 = VMACLO(z11, a5, b6);
  z11 = VMACLO(z11, a6, b5); z11 = VMACLO(z11, a7, b4);
  y11 = VMACHI(y11, a4, b7); y11 = VMACHI(y11, a5, b6);
  y11 = VMACHI(y11, a6, b5); y11 = VMACHI(y11, a7, b4);
  y11 = VSHL(y11, BALIGN);

  z12 = VMACLO(y11, a5, b7); z12 = VMACLO(z12, a6, b6);
  z12 = VMACLO(z12, a7, b5);
  y12 = VMACHI(y12, a5, b7); y12 = VMACHI(y12, a6, b6);
  y12 = VMACHI(y12, a7, b5);
  y12 = VSHL(y12, BALIGN);

  z13 = VMACLO(y12, a6, b7); z13 = VMACLO(z13, a7, b6);
  y13 = VMACHI(y13, a6, b7); y13 = VMACHI(y13, a7, b6);
  y13 = VSHL(y13, BALIGN);

  z14 = VMACLO(y13, a7, b7);
  y14 = VMACHI(y14, a7, b7);
  y14 = VSHL(y14, BALIGN);

  z15 = y14;

  // ta(ta0-ta3) = aL(a0-a3) + aH(a4-a7)
  ta0 = VADD(a0, a4); ta1 = VADD(a1, a5);
  ta2 = VADD(a2, a6); ta3 = VADD(a3, a7);

  // tb(tb0-tb3) = bL(b0-b3) + bH(b4-b7)
  tb0 = VADD(b0, b4); tb1 = VADD(b1, b5); 
  tb2 = VADD(b2, b6); tb3 = VADD(b3, b7);

  // zM = ta * tb - zL - zH 
  
  y0 = y1 = y2 = y3 = y4 = y5 = y6 = y7 = VZERO;

  m0 = VMACLO(m0, ta0, tb0);
  y0 = VMACHI(y0, ta0, tb0);
  y0 = VSHL(y0, BALIGN);

  m1 = VMACLO(y0, ta0, tb1); m1 = VMACLO(m1, ta1, tb0);
  y1 = VMACHI(y1, ta0, tb1); y1 = VMACHI(y1, ta1, tb0);
  y1 = VSHL(y1, BALIGN);

  m2 = VMACLO(y1, ta0, tb2); m2 = VMACLO(m2, ta1, tb1); 
  m2 = VMACLO(m2, ta2, tb0);
  y2 = VMACHI(y2, ta0, tb2); y2 = VMACHI(y2, ta1, tb1); 
  y2 = VMACHI(y2, ta2, tb0);
  y2 = VSHL(y2, BALIGN);

  m3 = VMACLO(y2, ta0, tb3); m3 = VMACLO(m3, ta1, tb2); 
  m3 = VMACLO(m3, ta2, tb1); m3 = VMACLO(m3, ta3, tb0);
  y3 = VMACHI(y3, ta0, tb3); y3 = VMACHI(y3, ta1, tb2); 
  y3 = VMACHI(y3, ta2, tb1); y3 = VMACHI(y3, ta3, tb0);
  y3 = VSHL(y3, BALIGN);

  m4 = VMACLO(y3, ta1, tb3); m4 = VMACLO(m4, ta2, tb2); 
  m4 = VMACLO(m4, ta3, tb1);
  y4 = VMACHI(y4, ta1, tb3); y4 = VMACHI(y4, ta2, tb2); 
  y4 = VMACHI(y4, ta3, tb1);
  y4 = VSHL(y4, BALIGN);

  m5 = VMACLO(y4, ta2, tb3); m5 = VMACLO(m5, ta3, tb2);
  y5 = VMACHI(y5, ta2, tb3); y5 = VMACHI(y5, ta3, tb2);
  y5 = VSHL(y5, BALIGN);

  m6 = VMACLO(y5, ta3, tb3);
  y6 = VMACHI(y6, ta3, tb3);
  y6 = VSHL(y6, BALIGN);

  m7 = y6;

  m0 = VSUB(m0, VADD(z0, z8 )); m1 = VSUB(m1, VADD(z1, z9 ));
  m2 = VSUB(m2, VADD(z2, z10)); m3 = VSUB(m3, VADD(z3, z11));
  m4 = VSUB(m4, VADD(z4, z12)); m5 = VSUB(m5, VADD(z5, z13));
  m6 = VSUB(m6, VADD(z6, z14)); m7 = VSUB(m7, VADD(z7, z15));

  // z = z + zM
  z4  = VADD(z4 , m0); z5  = VADD(z5 , m1);
  z6  = VADD(z6 , m2); z7  = VADD(z7 , m3);
  z8  = VADD(z8 , m4); z9  = VADD(z9 , m5);
  z10 = VADD(z10, m6); z11 = VADD(z11, m7);

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
  r[12] = z12; r[13] = z13; r[14] = z14; r[15] = z15; 
}

// integer multiplication (product-scanning)
void mul_384_8x1w_v2(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  __m512i z0  = VZERO, z1  = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4  = VZERO, z5  = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8  = VZERO, z9  = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i z12 = VZERO, z13 = VZERO, z14 = VZERO, z15 = VZERO;
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO, y15 = VZERO;

  z0 = VMACLO(z0, a0, b0);
  y0 = VMACHI(y0, a0, b0);
  y0 = VSHL(y0, BALIGN);

  z1 = VMACLO(y0, a0, b1); z1 = VMACLO(z1, a1, b0);
  y1 = VMACHI(y1, a0, b1); y1 = VMACHI(y1, a1, b0);
  y1 = VSHL(y1, BALIGN);

  z2 = VMACLO(y1, a0, b2); z2 = VMACLO(z2, a1, b1); z2 = VMACLO(z2, a2, b0);
  y2 = VMACHI(y2, a0, b2); y2 = VMACHI(y2, a1, b1); y2 = VMACHI(y2, a2, b0);
  y2 = VSHL(y2, BALIGN);

  z3 = VMACLO(y2, a0, b3); z3 = VMACLO(z3, a1, b2); z3 = VMACLO(z3, a2, b1); 
  z3 = VMACLO(z3, a3, b0);
  y3 = VMACHI(y3, a0, b3); y3 = VMACHI(y3, a1, b2); y3 = VMACHI(y3, a2, b1); 
  y3 = VMACHI(y3, a3, b0);
  y3 = VSHL(y3, BALIGN);

  z4 = VMACLO(y3, a0, b4); z4 = VMACLO(z4, a1, b3); z4 = VMACLO(z4, a2, b2); 
  z4 = VMACLO(z4, a3, b1); z4 = VMACLO(z4, a4, b0);
  y4 = VMACHI(y4, a0, b4); y4 = VMACHI(y4, a1, b3); y4 = VMACHI(y4, a2, b2); 
  y4 = VMACHI(y4, a3, b1); y4 = VMACHI(y4, a4, b0);
  y4 = VSHL(y4, BALIGN);

  z5 = VMACLO(y4, a0, b5); z5 = VMACLO(z5, a1, b4); z5 = VMACLO(z5, a2, b3);
  z5 = VMACLO(z5, a3, b2); z5 = VMACLO(z5, a4, b1); z5 = VMACLO(z5, a5, b0);
  y5 = VMACHI(y5, a0, b5); y5 = VMACHI(y5, a1, b4); y5 = VMACHI(y5, a2, b3);
  y5 = VMACHI(y5, a3, b2); y5 = VMACHI(y5, a4, b1); y5 = VMACHI(y5, a5, b0);
  y5 = VSHL(y5, BALIGN);

  z6 = VMACLO(y5, a0, b6); z6 = VMACLO(z6, a1, b5); z6 = VMACLO(z6, a2, b4);
  z6 = VMACLO(z6, a3, b3); z6 = VMACLO(z6, a4, b2); z6 = VMACLO(z6, a5, b1);
  z6 = VMACLO(z6, a6, b0);
  y6 = VMACHI(y6, a0, b6); y6 = VMACHI(y6, a1, b5); y6 = VMACHI(y6, a2, b4);
  y6 = VMACHI(y6, a3, b3); y6 = VMACHI(y6, a4, b2); y6 = VMACHI(y6, a5, b1);
  y6 = VMACHI(y6, a6, b0);
  y6 = VSHL(y6, BALIGN);

  z7 = VMACLO(y6, a0, b7); z7 = VMACLO(z7, a1, b6); z7 = VMACLO(z7, a2, b5);
  z7 = VMACLO(z7, a3, b4); z7 = VMACLO(z7, a4, b3); z7 = VMACLO(z7, a5, b2);
  z7 = VMACLO(z7, a6, b1); z7 = VMACLO(z7, a7, b0);
  y7 = VMACHI(y7, a0, b7); y7 = VMACHI(y7, a1, b6); y7 = VMACHI(y7, a2, b5);
  y7 = VMACHI(y7, a3, b4); y7 = VMACHI(y7, a4, b3); y7 = VMACHI(y7, a5, b2);
  y7 = VMACHI(y7, a6, b1); y7 = VMACHI(y7, a7, b0);
  y7 = VSHL(y7, BALIGN);

  z8 = VMACLO(y7, a1, b7); z8 = VMACLO(z8, a2, b6); z8 = VMACLO(z8, a3, b5);
  z8 = VMACLO(z8, a4, b4); z8 = VMACLO(z8, a5, b3); z8 = VMACLO(z8, a6, b2);
  z8 = VMACLO(z8, a7, b1);
  y8 = VMACHI(y8, a1, b7); y8 = VMACHI(y8, a2, b6); y8 = VMACHI(y8, a3, b5);
  y8 = VMACHI(y8, a4, b4); y8 = VMACHI(y8, a5, b3); y8 = VMACHI(y8, a6, b2);
  y8 = VMACHI(y8, a7, b1);
  y8 = VSHL(y8, BALIGN);

  z9 = VMACLO(y8, a2, b7); z9 = VMACLO(z9, a3, b6); z9 = VMACLO(z9, a4, b5);
  z9 = VMACLO(z9, a5, b4); z9 = VMACLO(z9, a6, b3); z9 = VMACLO(z9, a7, b2);
  y9 = VMACHI(y9, a2, b7); y9 = VMACHI(y9, a3, b6); y9 = VMACHI(y9, a4, b5);
  y9 = VMACHI(y9, a5, b4); y9 = VMACHI(y9, a6, b3); y9 = VMACHI(y9, a7, b2);
  y9 = VSHL(y9, BALIGN);

  z10 = VMACLO(y9,  a3, b7); z10 = VMACLO(z10, a4, b6);
  z10 = VMACLO(z10, a5, b5); z10 = VMACLO(z10, a6, b4);
  z10 = VMACLO(z10, a7, b3);
  y10 = VMACHI(y10, a3, b7); y10 = VMACHI(y10, a4, b6);
  y10 = VMACHI(y10, a5, b5); y10 = VMACHI(y10, a6, b4);
  y10 = VMACHI(y10, a7, b3);
  y10 = VSHL(y10, BALIGN);

  z11 = VMACLO(y10, a4, b7); z11 = VMACLO(z11, a5, b6);
  z11 = VMACLO(z11, a6, b5); z11 = VMACLO(z11, a7, b4);
  y11 = VMACHI(y11, a4, b7); y11 = VMACHI(y11, a5, b6);
  y11 = VMACHI(y11, a6, b5); y11 = VMACHI(y11, a7, b4);
  y11 = VSHL(y11, BALIGN);

  z12 = VMACLO(y11, a5, b7); z12 = VMACLO(z12, a6, b6);
  z12 = VMACLO(z12, a7, b5);
  y12 = VMACHI(y12, a5, b7); y12 = VMACHI(y12, a6, b6);
  y12 = VMACHI(y12, a7, b5);
  y12 = VSHL(y12, BALIGN);

  z13 = VMACLO(y12, a6, b7); z13 = VMACLO(z13, a7, b6);
  y13 = VMACHI(y13, a6, b7); y13 = VMACHI(y13, a7, b6);
  y13 = VSHL(y13, BALIGN);

  z14 = VMACLO(y13, a7, b7);
  y14 = VMACHI(y14, a7, b7);
  y14 = VSHL(y14, BALIGN);

  z15 = y14;

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
  r[12] = z12; r[13] = z13; r[14] = z14; r[15] = z15; 
}

// integer squaring (product-scanning)
void sqr_384_8x1w(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i z0  = VZERO, z1  = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4  = VZERO, z5  = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8  = VZERO, z9  = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i z12 = VZERO, z13 = VZERO, z14 = VZERO, z15 = VZERO;
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO, y15 = VZERO;

  z0 = VMACLO(z0, a0, a0);
  y0 = VMACHI(y0, a0, a0);
  y0 = VSHL(y0, BALIGN);

  z1 = VMACLO(y0, a0, a1); z1 = VADD(z1, z1);
  y1 = VMACHI(y1, a0, a1); 
  y1 = VSHL(y1, BALIGN+1);

  z2 = VMACLO(y1, a0, a2); z2 = VADD(z2, z2);       z2 = VMACLO(z2, a1, a1); 
  y2 = VMACHI(y2, a0, a2); y2 = VADD(y2, y2);       y2 = VMACHI(y2, a1, a1);
  y2 = VSHL(y2, BALIGN);

  z3 = VMACLO(y2, a0, a3); z3 = VMACLO(z3, a1, a2); z3 = VADD(z3, z3);
  y3 = VMACHI(y3, a0, a3); y3 = VMACHI(y3, a1, a2); 
  y3 = VSHL(y3, BALIGN+1);

  z4 = VMACLO(y3, a0, a4); z4 = VMACLO(z4, a1, a3); z4 = VADD(z4, z4);
  z4 = VMACLO(z4, a2, a2); 
  y4 = VMACHI(y4, a0, a4); y4 = VMACHI(y4, a1, a3); y4 = VADD(y4, y4);
  y4 = VMACHI(y4, a2, a2); 
  y4 = VSHL(y4, BALIGN);

  z5 = VMACLO(y4, a0, a5); z5 = VMACLO(z5, a1, a4); z5 = VMACLO(z5, a2, a3);
  z5 = VADD(z5, z5);
  y5 = VMACHI(y5, a0, a5); y5 = VMACHI(y5, a1, a4); y5 = VMACHI(y5, a2, a3);
  y5 = VSHL(y5, BALIGN+1);

  z6 = VMACLO(y5, a0, a6); z6 = VMACLO(z6, a1, a5); z6 = VMACLO(z6, a2, a4);
  z6 = VADD(z6, z6);       z6 = VMACLO(z6, a3, a3); 
  y6 = VMACHI(y6, a0, a6); y6 = VMACHI(y6, a1, a5); y6 = VMACHI(y6, a2, a4);
  y6 = VADD(y6, y6);       y6 = VMACHI(y6, a3, a3); 
  y6 = VSHL(y6, BALIGN);

  z7 = VMACLO(y6, a0, a7); z7 = VMACLO(z7, a1, a6); z7 = VMACLO(z7, a2, a5);
  z7 = VMACLO(z7, a3, a4); z7 = VADD(z7, z7);
  y7 = VMACHI(y7, a0, a7); y7 = VMACHI(y7, a1, a6); y7 = VMACHI(y7, a2, a5);
  y7 = VMACHI(y7, a3, a4); 
  y7 = VSHL(y7, BALIGN+1);

  z8 = VMACLO(y7, a1, a7); z8 = VMACLO(z8, a2, a6); z8 = VMACLO(z8, a3, a5);
  z8 = VADD(z8, z8);       z8 = VMACLO(z8, a4, a4); 
  y8 = VMACHI(y8, a1, a7); y8 = VMACHI(y8, a2, a6); y8 = VMACHI(y8, a3, a5);
  y8 = VADD(y8, y8);       y8 = VMACHI(y8, a4, a4); 
  y8 = VSHL(y8, BALIGN);

  z9 = VMACLO(y8, a2, a7); z9 = VMACLO(z9, a3, a6); z9 = VMACLO(z9, a4, a5);
  z9 = VADD(z9, z9);
  y9 = VMACHI(y9, a2, a7); y9 = VMACHI(y9, a3, a6); y9 = VMACHI(y9, a4, a5);
  y9 = VSHL(y9, BALIGN+1);

  z10 = VMACLO(y9,  a3, a7); z10 = VMACLO(z10, a4, a6);
  z10 = VADD(z10, z10);      z10 = VMACLO(z10, a5, a5); 
  y10 = VMACHI(y10, a3, a7); y10 = VMACHI(y10, a4, a6);
  y10 = VADD(y10, y10);      y10 = VMACHI(y10, a5, a5); 
  y10 = VSHL(y10, BALIGN);

  z11 = VMACLO(y10, a4, a7); z11 = VMACLO(z11, a5, a6);
  z11 = VADD(z11, z11);
  y11 = VMACHI(y11, a4, a7); y11 = VMACHI(y11, a5, a6);
  y11 = VSHL(y11, BALIGN+1);

  z12 = VMACLO(y11, a5, a7); z12 = VADD(z12, z12);  
  z12 = VMACLO(z12, a6, a6);
  y12 = VMACHI(y12, a5, a7); y12 = VADD(y12, y12);
  y12 = VMACHI(y12, a6, a6);
  y12 = VSHL(y12, BALIGN);

  z13 = VMACLO(y12, a6, a7); z13 = VADD(z13, z13);
  y13 = VMACHI(y13, a6, a7);
  y13 = VSHL(y13, BALIGN+1);

  z14 = VMACLO(y13, a7, a7);
  y14 = VMACHI(y14, a7, a7);
  y14 = VSHL(y14, BALIGN);

  z15 = y14;

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
  r[12] = z12; r[13] = z13; r[14] = z14; r[15] = z15; 
}

// Montgomery reduction (including final reduction)
void redc_mont_384_8x1w(__m512i r, const __m512i *a, const uint64_t *sp, 
                        const uint64_t n0)
{
  __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO, y15 = VZERO;
  const __m512i p0 = VSET1(sp[0]), p1 = VSET1(sp[1]), p2 = VSET1(sp[2]);
  const __m512i p3 = VSET1(sp[3]), p4 = VSET1(sp[4]), p5 = VSET1(sp[5]);
  const __m512i p6 = VSET1(sp[6]), p7 = VSET1(sp[7]), bmask = VSET1(BMASK);
  const __m512i VN0 = VSET1(n0), zero = VZERO;
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask, u;

  u  = VMACLO(zero, a0, VN0);
  a0 = VMACLO(a0, u, p0); a1 = VMACLO(a1, u, p1); a2 = VMACLO(a2, u, p2);
  a3 = VMACLO(a3, u, p3); a4 = VMACLO(a4, u, p4); a5 = VMACLO(a5, u, p5);
  a6 = VMACLO(a6, u, p6); a7 = VMACLO(a7, u, p7); 
  y0 = VMACHI(y0, u, p0); y1 = VMACHI(y1, u, p1); y2 = VMACHI(y2, u, p2);
  y3 = VMACHI(y3, u, p3); y4 = VMACHI(y4, u, p4); y5 = VMACHI(y5, u, p5);
  y6 = VMACHI(y6, u, p6); y7 = VMACHI(y7, u, p7); 
  a1 = VADD(VADD(a1, VSRA(a0, BRADIX)), VSHL(y0, BALIGN));

  u  = VMACLO(zero, a1, VN0);
  a1 = VMACLO(a1, u, p0); a2 = VMACLO(a2, u, p1); a3 = VMACLO(a3, u, p2);
  a4 = VMACLO(a4, u, p3); a5 = VMACLO(a5, u, p4); a6 = VMACLO(a6, u, p5);
  a7 = VMACLO(a7, u, p6); a8 = VMACLO(a8, u, p7); 
  y1 = VMACHI(y1, u, p0); y2 = VMACHI(y2, u, p1); y3 = VMACHI(y3, u, p2);
  y4 = VMACHI(y4, u, p3); y5 = VMACHI(y5, u, p4); y6 = VMACHI(y6, u, p5);
  y7 = VMACHI(y7, u, p6); y8 = VMACHI(y8, u, p7); 
  a2 = VADD(VADD(a2, VSRA(a1, BRADIX)), VSHL(y1, BALIGN));

  u  = VMACLO(zero, a2, VN0);
  a2 = VMACLO(a2, u, p0); a3 = VMACLO(a3, u, p1); a4 = VMACLO(a4, u, p2);
  a5 = VMACLO(a5, u, p3); a6 = VMACLO(a6, u, p4); a7 = VMACLO(a7, u, p5);
  a8 = VMACLO(a8, u, p6); a9 = VMACLO(a9, u, p7); 
  y2 = VMACHI(y2, u, p0); y3 = VMACHI(y3, u, p1); y4 = VMACHI(y4, u, p2);
  y5 = VMACHI(y5, u, p3); y6 = VMACHI(y6, u, p4); y7 = VMACHI(y7, u, p5);
  y8 = VMACHI(y8, u, p6); y9 = VMACHI(y9, u, p7); 
  a3 = VADD(VADD(a3, VSRA(a2, BRADIX)), VSHL(y2, BALIGN));

  u  = VMACLO(zero, a3, VN0);
  a3 = VMACLO(a3, u, p0); a4  = VMACLO(a4,  u, p1); a5 = VMACLO(a5, u, p2);
  a6 = VMACLO(a6, u, p3); a7  = VMACLO(a7,  u, p4); a8 = VMACLO(a8, u, p5);
  a9 = VMACLO(a9, u, p6); a10 = VMACLO(a10, u, p7); 
  y3 = VMACHI(y3, u, p0); y4  = VMACHI(y4 , u, p1); y5 = VMACHI(y5, u, p2);
  y6 = VMACHI(y6, u, p3); y7  = VMACHI(y7 , u, p4); y8 = VMACHI(y8, u, p5);
  y9 = VMACHI(y9, u, p6); y10 = VMACHI(y10, u, p7); 
  a4 = VADD(VADD(a4, VSRA(a3, BRADIX)), VSHL(y3, BALIGN));

  u   = VMACLO(zero, a4, VN0);
  a4  = VMACLO(a4 , u, p0); a5  = VMACLO(a5 , u, p1); a6 = VMACLO(a6, u, p2);
  a7  = VMACLO(a7 , u, p3); a8  = VMACLO(a8 , u, p4); a9 = VMACLO(a9, u, p5);
  a10 = VMACLO(a10, u, p6); a11 = VMACLO(a11, u, p7); 
  y4  = VMACHI(y4 , u, p0); y5  = VMACHI(y5 , u, p1); y6 = VMACHI(y6, u, p2);
  y7  = VMACHI(y7 , u, p3); y8  = VMACHI(y8 , u, p4); y9 = VMACHI(y9, u, p5);
  y10 = VMACHI(y10, u, p6); y11 = VMACHI(y11, u, p7); 
  a5  = VADD(VADD(a5, VSRA(a4, BRADIX)), VSHL(y4, BALIGN));
}