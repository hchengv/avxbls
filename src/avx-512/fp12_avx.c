#include "fp12_avx.h"


// ----------------------------------------------------------------------------
// modular operations

// modular addition
static void add_mod_384_8x1w(__m512i *r, const __m512i *a, const __m512i *b, const uint64_t *sp)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(sp[0]), p1 = VSET1(sp[1]), p2 = VSET1(sp[2]);
  const __m512i p3 = VSET1(sp[3]), p4 = VSET1(sp[4]), p5 = VSET1(sp[5]);
  const __m512i p6 = VSET1(sp[6]), p7 = VSET1(sp[7]), bmask = VSET1(BMASK);
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

static void add_mod_384_4x2w(__m512i *r, const __m512i *a, const __m512i *b, const uint64_t *sp)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i p0 = VSET(sp[0], sp[4], sp[0], sp[4], sp[0], sp[4], sp[0], sp[4]);
  const __m512i p1 = VSET(sp[1], sp[5], sp[1], sp[5], sp[1], sp[5], sp[1], sp[5]);
  const __m512i p2 = VSET(sp[2], sp[6], sp[2], sp[6], sp[2], sp[6], sp[2], sp[6]);
  const __m512i p3 = VSET(sp[3], sp[7], sp[3], sp[7], sp[3], sp[7], sp[3], sp[7]);
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

  // r = r + (2p & smask)
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
// Fp operations

void add_fp_8x1w(__m512i *r, const __m512i *a, const __m512i *b)
{
  add_mod_384_8x1w(r, a, b, BLS12_381_P_R48);
}

void add_fp_4x2w(__m512i *r, const __m512i *a, const __m512i *b)
{
  add_mod_384_4x2w(r, a, b, BLS12_381_P_R48);
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

// ----------------------------------------------------------------------------
// Fp12 operations

// To understand the comments, see Listing 21 in "Guide to Pairing-Based Cryptography". 
// void cyclotomic_sqr_fp12_vec(__m512i *ra, __m512i *rbc, const __m512i *a, const __m512i *bc)
// {
//   __m512i ta[VWORDS], tbc[NWORDS]; 

//   // compute A in 1x2x2x2w 
//   // a: z0 | z1 
//   sqr_fp4_1x2x2x2w(ta, a);              //        t0 |        t1
//   mix_sa_fp2_2x2x2w(ra, ta, a);         //     t0-z0 |     t1+z1
//   add_fp2_2x2x2w(ra, ra, ra);           // 2*(t0-z0) | 2*(t1+z1)
//   add_fp2_2x2x2w(ra, ra, ta);           // 3*t0-2*z0 | 3*t1+2*z1

//   // compute B and C in 2x2x2x1w
//   // bc: z2 | z3 | z4 | z5
//   sqr_fp4_2x2x2x1w(tbc, bc);            //              t0 |        t1 |        t2 |        t3 
//   mul_by_u_plus_1_fp2_1w();             //                                             t3*(u+1)
//   some_permute(tbc, tbc);               //     t3*(u+1)    |        t2 |        t0 |        t1
//   mix_assa_fp2_4x2x1w(rbc, tbc, bc);    //     t3*(u+1)+z2 |     t2-z3 |     t0-z4 |     t1+z5 
//   add_fp2_4x2x1w(rbc, rbc, rbc);        // 2*(t3*(u+1)+z2) | 2*(t2-z3) | 2*(t0-z4) | 2*(t1+z5)
//   add_fp2_4x2x1w(rbc, rbc, tbc);        // 3*t3*(u+1)+2*z2 | 3*t2-2*z3 | 3*t0-2*z4 | 3*t1+2*z5 
// }

// ----------------------------------------------------------------------------

