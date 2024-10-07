#include "fp12_avx.h"


// ----------------------------------------------------------------------------
// Fp operations

void add_fp_8x1w(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask, t0;

  // r = a + b
  r0 = VADD(a0, b0); r1 = VADD(a1, b1); r2 = VADD(a2, b2); r3 = VADD(a3, b3);
  r4 = VADD(a4, b4); r5 = VADD(a5, b5); r6 = VADD(a6, b6); r7 = VADD(a7, b7);

  // r = r - p
  r0 = VSUB(r0, p0); r1 = VSUB(r1, p1); r2 = VSUB(r2, p2); r3 = VSUB(r3, p3);
  r4 = VSUB(r4, p4); r5 = VSUB(r5, p5); r6 = VSUB(r6, p6); r7 = VSUB(r7, p7);

  // get sign mask 
  t0 = VADD(r1, VSRA(r0, BRADIX));
  t0 = VADD(r2, VSRA(t0, BRADIX));
  t0 = VADD(r3, VSRA(t0, BRADIX));
  t0 = VADD(r4, VSRA(t0, BRADIX));
  t0 = VADD(r5, VSRA(t0, BRADIX));
  t0 = VADD(r6, VSRA(t0, BRADIX));
  t0 = VADD(r7, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
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

void add_fp_4x2w(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask = VSET1(BMASK); 
  __m512i r0, r1, r2, r3, smask, t0;

  // r = a + b
  r0 = VADD(a0, b0); r1 = VADD(a1, b1); r2 = VADD(a2, b2); r3 = VADD(a3, b3);

  // r = r - p
  r0 = VSUB(r0, p0); r1 = VSUB(r1, p1); r2 = VSUB(r2, p2); r3 = VSUB(r3, p3);

  // get sign mask
  t0 = VMADD(r1, 0x55, r1, VSRA(r0, BRADIX));
  t0 = VMADD(r2, 0x55, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0x55, r3, VSRA(t0, BRADIX));
  t0 = VMADD(r0, 0xAA, r0, VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0 = VMADD(r1, 0xAA, r1, VSRA(t0, BRADIX));
  t0 = VMADD(r2, 0xAA, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0xAA, r3, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 

  // carry propagation 
  // r0 is finally 49-bit not 48-bit
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r0 = VMADD(r0, 0xAA, r0, VSHUF(VSRA(r3, BRADIX), 0x4E)); r3 = VAND(r3, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
}

// ----------------------------------------------------------------------------
// Fp2 operations

static void add_fp2_4x2x1w(__m512i *r, const __m512i *a, const __m512i *b)
{
  add_fp_8x1w(r, a, b);
}

static void add_fp2_2x2x2w(__m512i *r, const __m512i *a, const __m512i *b)
{
  add_fp_4x2w(r, a, b);
}

// a = < D1 | D0 | C1 | C0 | B1 | B0 | A1 | A0 >  
// b = < H1 | H0 | G1 | G0 | F1 | F0 | E1 | E0 >
// r = < D1+H1 | D0+H0 | C1-G1 | C0-G0 | B1-F1 | B0-F0 | A1+E1 | A0+E0 >
void assa_fp2_4x2x1w(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask;
  __m512i t0, t1, t2, t3, t4, t5, t6, t7;

  // r = D1+H1 | D0+H0 | C1 | C0 | B1 | B0 | A1+E1 | A0+E0
  r0 = VMADD(a0, 0xC3, a0, b0); r1 = VMADD(a1, 0xC3, a1, b1);
  r2 = VMADD(a2, 0xC3, a2, b2); r3 = VMADD(a3, 0xC3, a3, b3);
  r4 = VMADD(a4, 0xC3, a4, b4); r5 = VMADD(a5, 0xC3, a5, b5);
  r6 = VMADD(a6, 0xC3, a6, b6); r7 = VMADD(a7, 0xC3, a7, b7);

  // t = p | p | G1 | G0 | F1 | F0 | p | p
  t0 = VMBLEND(0xC3, b0, p0); t1 = VMBLEND(0xC3, b1, p1);
  t2 = VMBLEND(0xC3, b2, p2); t3 = VMBLEND(0xC3, b3, p3);
  t4 = VMBLEND(0xC3, b4, p4); t5 = VMBLEND(0xC3, b5, p5);
  t6 = VMBLEND(0xC3, b6, p6); t7 = VMBLEND(0xC3, b7, p7); 

  // r = D1+H1-p | D0+H0-p | C1-G1 | C0-G0 | B1-F1 | B0-F0 | A1+E1-p | A0+E0-p
  r0 = VSUB(r0, t0); r1 = VSUB(r1, t1); r2 = VSUB(r2, t2); r3 = VSUB(r3, t3);
  r4 = VSUB(r4, t4); r5 = VSUB(r5, t5); r6 = VSUB(r6, t6); r7 = VSUB(r7, t7);

  // get sign mask 
  t0 = VADD(r1, VSRA(r0, BRADIX));
  t0 = VADD(r2, VSRA(t0, BRADIX));
  t0 = VADD(r3, VSRA(t0, BRADIX));
  t0 = VADD(r4, VSRA(t0, BRADIX));
  t0 = VADD(r5, VSRA(t0, BRADIX));
  t0 = VADD(r6, VSRA(t0, BRADIX));
  t0 = VADD(r7, VSRA(t0, BRADIX));

  // get sign mask 
  t0 = VADD(r1, VSRA(r0, BRADIX));
  t0 = VADD(r2, VSRA(t0, BRADIX));
  t0 = VADD(r3, VSRA(t0, BRADIX));
  t0 = VADD(r4, VSRA(t0, BRADIX));
  t0 = VADD(r5, VSRA(t0, BRADIX));
  t0 = VADD(r6, VSRA(t0, BRADIX));
  t0 = VADD(r7, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
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

// a = < B1' | B1 | B0' | B0 | A1' | A1 | A0' | A0 >
// b = < D1' | D1 | D0' | D0 | C1' | C1 | C0' | C0 >
// r = < B1'+D1' | B1+D1 | B0'+D0' | B0+D0 | A1'-C1' | A1-C1 | A0'-C0' | A0-C0 >
void as_fp2_2x2x2w(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask = VSET1(BMASK); 
  __m512i r0, r1, r2, r3, smask;
  __m512i t0, t1, t2, t3;

  // r =  B1'+D1' | B1+D1 | B0'+D0' | B0+D0 | A1' | A1 | A0' | A0 
  r0 = VMADD(a0, 0xF0, a0, b0); r1 = VMADD(a1, 0xF0, a1, b1);
  r2 = VMADD(a2, 0xF0, a2, b2); r3 = VMADD(a3, 0xF0, a3, b3);

  // t = p' | p | p' | p | C1' | C1 | C0' | C0
  t0 = VMBLEND(0xF0, b0, p0); t1 = VMBLEND(0xF0, b1, p1);
  t2 = VMBLEND(0xF0, b2, p2); t3 = VMBLEND(0xF0, b3, p3);

  // r = B1'+D1'-p' | B1+D1-p | B0'+D0'-p' | B0+D0-p | A1'-C1' | A1-C1 | A0'-C0' | A0-C0
  r0 = VSUB(r0, t0); r1 = VSUB(r1, t1); r2 = VSUB(r2, t2); r3 = VSUB(r3, t3);

  // get sign mask
  t0 = VMADD(r1, 0x55, r1, VSRA(r0, BRADIX));
  t0 = VMADD(r2, 0x55, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0x55, r3, VSRA(t0, BRADIX));
  t0 = VMADD(r0, 0xAA, r0, VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0 = VMADD(r1, 0xAA, r1, VSRA(t0, BRADIX));
  t0 = VMADD(r2, 0xAA, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0xAA, r3, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 

  // carry propagation 
  // r0 is finally 49-bit not 48-bit
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r0 = VMADD(r0, 0xAA, r0, VSHUF(VSRA(r3, BRADIX), 0x4E)); r3 = VAND(r3, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
}

// ----------------------------------------------------------------------------
// Fp4 operations

// r0 = a0^2 + (u+1)*a1^2
// r1 = 2*a0*a1
void sqr_fp4_2x2x2x1w(__m512i *r, __m512i *a)
{
  __m512i t0[NWORDS*2];

  // a = B1 | B0 | A1 | A0 at Fp layer
  sqr_fp2x2_4x2x1w(t0, a);              // B1^2 | B0^2 | A1^2 | A0^2
  mul_by_u_plus_1_fp2x2_TBA();          // (u+1)*B1^2, (u+1)*A1^2
  add_fp2x2_TBA();                      // ... | B0^2+(u+1)*B1^2 | ... | A0^2+(u+1)*(A1^2)
  mul_fp2x2_2x4x1w();                   //   B0*B1 | ... |   A0*A1 | ...
  add_fp2x2_TBA();                      // 2*B0*B1 | ... | 2*A0*A1 | ...  
  some_blend();                         // 2*B0*B1 | B0^2+(u+1)*B1^2 | 2*A0*A1 | A0^2+(u+1)*(A1^2)
  redc_fp2x2_2x2x2w();                  // 2*B0*B1 | B0^2+(u+1)*B1^2 | 2*A0*A1 | A0^2+(u+1)*(A1^2)
}


// ----------------------------------------------------------------------------
// Fp12 operations

// To understand the comments, see Listing 21 in "Guide to Pairing-Based Cryptography". 
void cyclotomic_sqr_fp12_vec(__m512i *ra, __m512i *rbc, const __m512i *a, const __m512i *bc)
{
  __m512i ta[VWORDS], tbc[NWORDS]; 

  // compute A in 1x2x2x2w 
  // a = z1 | z0 at Fp2 layer
  // sqr_fp4_1x2x2x2w(ta, a);              //        t1 |        t0      
  as_fp2_2x2x2w(ra, ta, a);             //     t1+z1 |     t0-z0     
  add_fp2_2x2x2w(ra, ra, ra);           // 2*(t1+z1) | 2*(t0-z0) 
  add_fp2_2x2x2w(ra, ra, ta);           // 3*t1+2*z1 | 3*t0-2*z0

  // compute B and C in 2x2x2x1w
  // bc = z5 | z4 | z3 | z2 at Fp2 layer
  // sqr_fp4_2x2x2x1w(tbc, bc);            //              t3 |        t2 |       t1  |              t0 
  // mul_by_u_plus_1_fp2_1w();             //        t3*(u+1)
  // some_permute(tbc, tbc);               //              t1 |        t0 |       t2  |        t3*(u+1)
  assa_fp2_4x2x1w(rbc, tbc, bc);        //           t1+z5 |     t0-z4 |    t2-z3  |     t3*(u+1)+z2
  add_fp2_4x2x1w(rbc, rbc, rbc);        //       2*(t1+z5) | 2*(t0-z4) | 2*(t2-z3) | 2*(t3*(u+1)+z2)
  add_fp2_4x2x1w(rbc, rbc, tbc);        //       3*t1+2*z5 | 3*t0-2*z4 | 3*t2-2*z3 | 3*t3*(u+1)+2*z2
}

// ----------------------------------------------------------------------------

