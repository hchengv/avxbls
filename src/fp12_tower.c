/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fields.h"
#include "fp12_avx.h"

/*
 * Fp2  = Fp[u]  / (u^2 + 1)
 * Fp6  = Fp2[v] / (v^3 - u - 1)
 * Fp12 = Fp6[w] / (w^2 - v)
 */

static inline void mul_by_u_plus_1_fp2(vec384x ret, const vec384x a)
{   mul_by_1_plus_i_mod_384x(ret, a, BLS12_381_P);   }

/*
 * Fp2x2 is a "widened" version of Fp2, which allows to consolidate
 * reductions from several multiplications. In other words instead of
 * "mul_redc-mul_redc-add" we get "mul-mul-add-redc," where latter
 * addition is double-width... To be more specific this gives ~7-10%
 * faster pairing depending on platform...
 */

static inline void add_fp2x2(vec768x ret, const vec768x a, const vec768x b)
{
    add_mod_384x384(ret[0], a[0], b[0], BLS12_381_P);
    add_mod_384x384(ret[1], a[1], b[1], BLS12_381_P);
}

static inline void sub_fp2x2(vec768x ret, const vec768x a, const vec768x b)
{
    sub_mod_384x384(ret[0], a[0], b[0], BLS12_381_P);
    sub_mod_384x384(ret[1], a[1], b[1], BLS12_381_P);
}

static inline void mul_by_u_plus_1_fp2x2(vec768x ret, const vec768x a)
{
    /* caveat lector! |ret| may not be same as |a| */
    sub_mod_384x384(ret[0], a[0], a[1], BLS12_381_P);
    add_mod_384x384(ret[1], a[0], a[1], BLS12_381_P);
}

static inline void redc_fp2x2(vec384x ret, const vec768x a)
{
    redc_mont_384(ret[0], a[0], BLS12_381_P, p0);
    redc_mont_384(ret[1], a[1], BLS12_381_P, p0);
}

#ifndef BENCHMARK
static 
#endif
void mul_fp2x2(vec768x ret, const vec384x a, const vec384x b)
{
#if 1
    mul_382x(ret, a, b, BLS12_381_P);   /* +~6% in Miller loop */
#else
    union { vec384 x[2]; vec768 x2; } t;

    add_mod_384(t.x[0], a[0], a[1], BLS12_381_P);
    add_mod_384(t.x[1], b[0], b[1], BLS12_381_P);
    mul_384(ret[1], t.x[0], t.x[1]);

    mul_384(ret[0], a[0], b[0]);
    mul_384(t.x2,   a[1], b[1]);

    sub_mod_384x384(ret[1], ret[1], ret[0], BLS12_381_P);
    sub_mod_384x384(ret[1], ret[1], t.x2, BLS12_381_P);

    sub_mod_384x384(ret[0], ret[0], t.x2, BLS12_381_P);
#endif
}

#ifndef BENCHMARK
static 
#endif
void sqr_fp2x2(vec768x ret, const vec384x a)
{
#if 1
    sqr_382x(ret, a, BLS12_381_P);      /* +~5% in final exponentiation */
#else
    vec384 t0, t1;

    add_mod_384(t0, a[0], a[1], BLS12_381_P);
    sub_mod_384(t1, a[0], a[1], BLS12_381_P);

    mul_384(ret[1], a[0], a[1]);
    add_mod_384x384(ret[1], ret[1], ret[1], BLS12_381_P);

    mul_384(ret[0], t0, t1);
#endif
}

/*
 * Fp6 extension
 */

static inline void sub_fp6x2(vec768fp6 ret, const vec768fp6 a,
                                            const vec768fp6 b)
{
    sub_fp2x2(ret[0], a[0], b[0]);
    sub_fp2x2(ret[1], a[1], b[1]);
    sub_fp2x2(ret[2], a[2], b[2]);
}

#ifndef BENCHMARK
static 
#endif
void mul_fp6x2(vec768fp6 ret, const vec384fp6 a, const vec384fp6 b)
{
    vec768x t0, t1, t2;
    vec384x aa, bb;

    mul_fp2x2(t0, a[0], b[0]);
    mul_fp2x2(t1, a[1], b[1]);
    mul_fp2x2(t2, a[2], b[2]);

    /* ret[0] = ((a1 + a2)*(b1 + b2) - a1*b1 - a2*b2)*(u+1) + a0*b0
              = (a1*b2 + a2*b1)*(u+1) + a0*b0 */
    add_fp2(aa, a[1], a[2]);
    add_fp2(bb, b[1], b[2]);
    mul_fp2x2(ret[0], aa, bb);
    sub_fp2x2(ret[0], ret[0], t1);
    sub_fp2x2(ret[0], ret[0], t2);
    mul_by_u_plus_1_fp2x2(ret[1], ret[0]);  /* borrow ret[1] for a moment */
    add_fp2x2(ret[0], ret[1], t0);

    /* ret[1] = (a0 + a1)*(b0 + b1) - a0*b0 - a1*b1 + a2*b2*(u+1)
              = a0*b1 + a1*b0 + a2*b2*(u+1) */
    add_fp2(aa, a[0], a[1]);
    add_fp2(bb, b[0], b[1]);
    mul_fp2x2(ret[1], aa, bb);
    sub_fp2x2(ret[1], ret[1], t0);
    sub_fp2x2(ret[1], ret[1], t1);
    mul_by_u_plus_1_fp2x2(ret[2], t2);      /* borrow ret[2] for a moment */
    add_fp2x2(ret[1], ret[1], ret[2]);

    /* ret[2] = (a0 + a2)*(b0 + b2) - a0*b0 - a2*b2 + a1*b1
              = a0*b2 + a2*b0 + a1*b1 */
    add_fp2(aa, a[0], a[2]);
    add_fp2(bb, b[0], b[2]);
    mul_fp2x2(ret[2], aa, bb);
    sub_fp2x2(ret[2], ret[2], t0);
    sub_fp2x2(ret[2], ret[2], t2);
    add_fp2x2(ret[2], ret[2], t1);
}

static inline void redc_fp6x2(vec384fp6 ret, const vec768fp6 a)
{
    redc_fp2x2(ret[0], a[0]);
    redc_fp2x2(ret[1], a[1]);
    redc_fp2x2(ret[2], a[2]);
}

static void mul_fp6(vec384fp6 ret, const vec384fp6 a, const vec384fp6 b)
{
    vec768fp6 r;

    mul_fp6x2(r, a, b);
    redc_fp6x2(ret, r); /* narrow to normal width */
}

static void sqr_fp6(vec384fp6 ret, const vec384fp6 a)
{
    vec768x s0, m01, m12, s2, rx;

    sqr_fp2x2(s0, a[0]);

    mul_fp2x2(m01, a[0], a[1]);
    add_fp2x2(m01, m01, m01);

    mul_fp2x2(m12, a[1], a[2]);
    add_fp2x2(m12, m12, m12);

    sqr_fp2x2(s2, a[2]);

    /* ret[2] = (a0 + a1 + a2)^2 - a0^2 - a2^2 - 2*(a0*a1) - 2*(a1*a2)
              = a1^2 + 2*(a0*a2) */
    add_fp2(ret[2], a[2], a[1]);
    add_fp2(ret[2], ret[2], a[0]);
    sqr_fp2x2(rx, ret[2]);
    sub_fp2x2(rx, rx, s0);
    sub_fp2x2(rx, rx, s2);
    sub_fp2x2(rx, rx, m01);
    sub_fp2x2(rx, rx, m12);
    redc_fp2x2(ret[2], rx);

    /* ret[0] = a0^2 + 2*(a1*a2)*(u+1) */
    mul_by_u_plus_1_fp2x2(rx, m12);
    add_fp2x2(rx, rx, s0);
    redc_fp2x2(ret[0], rx);

    /* ret[1] = a2^2*(u+1) + 2*(a0*a1) */
    mul_by_u_plus_1_fp2x2(rx, s2);
    add_fp2x2(rx, rx, m01);
    redc_fp2x2(ret[1], rx);
}

static void add_fp6(vec384fp6 ret, const vec384fp6 a, const vec384fp6 b)
{
    add_fp2(ret[0], a[0], b[0]);
    add_fp2(ret[1], a[1], b[1]);
    add_fp2(ret[2], a[2], b[2]);
}

static void sub_fp6(vec384fp6 ret, const vec384fp6 a, const vec384fp6 b)
{
    sub_fp2(ret[0], a[0], b[0]);
    sub_fp2(ret[1], a[1], b[1]);
    sub_fp2(ret[2], a[2], b[2]);
}

static void neg_fp6(vec384fp6 ret, const vec384fp6 a)
{
    neg_fp2(ret[0], a[0]);
    neg_fp2(ret[1], a[1]);
    neg_fp2(ret[2], a[2]);
}

/*
 * Fp12 extension
 */
void mul_fp12_scalar(vec384fp12 ret, const vec384fp12 a, const vec384fp12 b)
{
    vec768fp6 t0, t1, rx;
    vec384fp6 t2;

    mul_fp6x2(t0, a[0], b[0]);
    mul_fp6x2(t1, a[1], b[1]);

    /* ret[1] = (a0 + a1)*(b0 + b1) - a0*b0 - a1*b1
              = a0*b1 + a1*b0 */
    add_fp6(t2, a[0], a[1]);
    add_fp6(ret[1], b[0], b[1]);
    mul_fp6x2(rx, ret[1], t2);
    sub_fp6x2(rx, rx, t0);
    sub_fp6x2(rx, rx, t1);
    redc_fp6x2(ret[1], rx);

    /* ret[0] = a0*b0 + a1*b1*v */
    mul_by_u_plus_1_fp2x2(rx[0], t1[2]);
    add_fp2x2(rx[0], t0[0], rx[0]);
    add_fp2x2(rx[1], t0[1], t1[0]);
    add_fp2x2(rx[2], t0[2], t1[1]);
    redc_fp6x2(ret[0], rx);
}

void mul_fp12_vector(vec384fp12 ret, const vec384fp12 a, const vec384fp12 b)
{
  fp2_4x2x1w ab0, ab1, ab2;
  fp2_2x2x2w r001, r101, r2;
  __m512i t[3][SWORDS];
  int i;

  // form < b1 | a1 | b0 | a0 >
  for (i = 0; i < SWORDS; i++) {
    t[0][i] = VSET( b[1][0][1][i], b[1][0][0][i],
                    a[1][0][1][i], a[1][0][0][i],
                    b[0][0][1][i], b[0][0][0][i],
                    a[0][0][1][i], a[0][0][0][i]);
    t[1][i] = VSET( b[1][1][1][i], b[1][1][0][i],
                    a[1][1][1][i], a[1][1][0][i],
                    b[0][1][1][i], b[0][1][0][i],
                    a[0][1][1][i], a[0][1][0][i]);
    t[2][i] = VSET( b[1][2][1][i], b[1][2][0][i],
                    a[1][2][1][i], a[1][2][0][i],
                    b[0][2][1][i], b[0][2][0][i],
                    a[0][2][1][i], a[0][2][0][i]);
  }

  conv_64to48_fp_8x1w(ab0, t[0]);
  conv_64to48_fp_8x1w(ab1, t[1]);
  conv_64to48_fp_8x1w(ab2, t[2]);

  mul_fp12_vec_v3(r001, r101, r2, ab0, ab1, ab2);

  carryp_fp_4x2w(r001);
  carryp_fp_4x2w(r101);
  carryp_fp_4x2w(r2);

  conv_48to64_fp_4x2w(t[0], r001);
  conv_48to64_fp_4x2w(t[1], r101);
  conv_48to64_fp_4x2w(t[2], r2);

  for (i = 0; i < SWORDS/2; i++) {
    ret[0][0][0][i         ] = ((uint64_t *)&t[0][i])[0];
    ret[0][0][0][i+SWORDS/2] = ((uint64_t *)&t[0][i])[1];
    ret[0][0][1][i         ] = ((uint64_t *)&t[0][i])[2];
    ret[0][0][1][i+SWORDS/2] = ((uint64_t *)&t[0][i])[3];
    ret[0][1][0][i         ] = ((uint64_t *)&t[0][i])[4];
    ret[0][1][0][i+SWORDS/2] = ((uint64_t *)&t[0][i])[5];
    ret[0][1][1][i         ] = ((uint64_t *)&t[0][i])[6];
    ret[0][1][1][i+SWORDS/2] = ((uint64_t *)&t[0][i])[7];
    ret[1][0][0][i         ] = ((uint64_t *)&t[1][i])[0];
    ret[1][0][0][i+SWORDS/2] = ((uint64_t *)&t[1][i])[1];
    ret[1][0][1][i         ] = ((uint64_t *)&t[1][i])[2];
    ret[1][0][1][i+SWORDS/2] = ((uint64_t *)&t[1][i])[3];
    ret[1][1][0][i         ] = ((uint64_t *)&t[1][i])[4];
    ret[1][1][0][i+SWORDS/2] = ((uint64_t *)&t[1][i])[5];
    ret[1][1][1][i         ] = ((uint64_t *)&t[1][i])[6];
    ret[1][1][1][i+SWORDS/2] = ((uint64_t *)&t[1][i])[7];
    ret[0][2][0][i         ] = ((uint64_t *)&t[2][i])[0];
    ret[0][2][0][i+SWORDS/2] = ((uint64_t *)&t[2][i])[1];
    ret[0][2][1][i         ] = ((uint64_t *)&t[2][i])[2];
    ret[0][2][1][i+SWORDS/2] = ((uint64_t *)&t[2][i])[3];
    ret[1][2][0][i         ] = ((uint64_t *)&t[2][i])[4];
    ret[1][2][0][i+SWORDS/2] = ((uint64_t *)&t[2][i])[5];
    ret[1][2][1][i         ] = ((uint64_t *)&t[2][i])[6];
    ret[1][2][1][i+SWORDS/2] = ((uint64_t *)&t[2][i])[7];
  }
}

static inline void mul_by_0y0_fp6x2(vec768fp6 ret, const vec384fp6 a,
                                                   const vec384fp2 b)
{
    mul_fp2x2(ret[1], a[2], b);     /* borrow ret[1] for a moment */
    mul_by_u_plus_1_fp2x2(ret[0], ret[1]);
    mul_fp2x2(ret[1], a[0], b);
    mul_fp2x2(ret[2], a[1], b);
}

static void mul_by_xy0_fp6x2(vec768fp6 ret, const vec384fp6 a,
                                            const vec384fp6 b)
{
    vec768x t0, t1;
    vec384x aa, bb;

    mul_fp2x2(t0, a[0], b[0]);
    mul_fp2x2(t1, a[1], b[1]);

    /* ret[0] = ((a1 + a2)*(b1 + 0) - a1*b1 - a2*0)*(u+1) + a0*b0
              = (a1*0 + a2*b1)*(u+1) + a0*b0 */
    mul_fp2x2(ret[1], a[2], b[1]);  /* borrow ret[1] for a moment */
    mul_by_u_plus_1_fp2x2(ret[0], ret[1]);
    add_fp2x2(ret[0], ret[0], t0);

    /* ret[1] = (a0 + a1)*(b0 + b1) - a0*b0 - a1*b1 + a2*0*(u+1)
              = a0*b1 + a1*b0 + a2*0*(u+1) */
    add_fp2(aa, a[0], a[1]);
    add_fp2(bb, b[0], b[1]);
    mul_fp2x2(ret[1], aa, bb);
    sub_fp2x2(ret[1], ret[1], t0);
    sub_fp2x2(ret[1], ret[1], t1);

    /* ret[2] = (a0 + a2)*(b0 + 0) - a0*b0 - a2*0 + a1*b1
              = a0*0 + a2*b0 + a1*b1 */
    mul_fp2x2(ret[2], a[2], b[0]);
    add_fp2x2(ret[2], ret[2], t1);
}

void mul_by_xy00z0_fp12_scalar(vec384fp12 ret, const vec384fp12 a,
                                               const vec384fp6 xy00z0)
{
    vec768fp6 t0, t1, rr;
    vec384fp6 t2;

    mul_by_xy0_fp6x2(t0, a[0], xy00z0);
    mul_by_0y0_fp6x2(t1, a[1], xy00z0[2]);

    /* ret[1] = (a0 + a1)*(b0 + b1) - a0*b0 - a1*b1
              = a0*b1 + a1*b0 */
    vec_copy(t2[0], xy00z0[0], sizeof(t2[0]));
    add_fp2(t2[1], xy00z0[1], xy00z0[2]);
    add_fp6(ret[1], a[0], a[1]);
    mul_by_xy0_fp6x2(rr, ret[1], t2);
    sub_fp6x2(rr, rr, t0);
    sub_fp6x2(rr, rr, t1);
    redc_fp6x2(ret[1], rr);

    /* ret[0] = a0*b0 + a1*b1*v */
    mul_by_u_plus_1_fp2x2(rr[0], t1[2]);
    add_fp2x2(rr[0], t0[0], rr[0]);
    add_fp2x2(rr[1], t0[1], t1[0]);
    add_fp2x2(rr[2], t0[2], t1[1]);
    redc_fp6x2(ret[0], rr);
}

void mul_by_xy00z0_fp12_vector(vec384fp12 ret, const vec384fp12 a,
                                               const vec384fp6 xy00z0)
{
  fp2_4x2x1w r0, a01, a2, b01, b4;
  fp2_2x2x2w r1;
  __m512i t[4][SWORDS];
  int i;

  for (i = 0; i < SWORDS; i++) {
    t[0][i] = VSET(a[1][1][1][i], a[1][1][0][i],
                   a[1][0][1][i], a[1][0][0][i],
                   a[0][1][1][i], a[0][1][0][i],
                   a[0][0][1][i], a[0][0][0][i]);
    t[1][i] = VSET(a[1][2][1][i], a[1][2][0][i],
                   a[1][2][1][i], a[1][2][0][i],
                   a[0][2][1][i], a[0][2][0][i],
                   a[0][2][1][i], a[0][2][0][i]);
    t[2][i] = VSET(xy00z0[1][1][i], xy00z0[1][0][i],
                   xy00z0[0][1][i], xy00z0[0][0][i],
                   xy00z0[1][1][i], xy00z0[1][0][i],
                   xy00z0[0][1][i], xy00z0[0][0][i]);
    t[3][i] = VSET(xy00z0[2][1][i], xy00z0[2][0][i],
                   xy00z0[2][1][i], xy00z0[2][0][i],
                   xy00z0[2][1][i], xy00z0[2][0][i],
                   xy00z0[2][1][i], xy00z0[2][0][i]);
  }
  conv_64to48_fp_8x1w(a01, t[0]);
  conv_64to48_fp_8x1w(a2 , t[1]);
  conv_64to48_fp_8x1w(b01, t[2]);
  conv_64to48_fp_8x1w(b4 , t[3]);

  mul_by_xy00z0_fp12_vec_v1(r0, r1, a01, a2, b01, b4);

  carryp_fp_4x2w(r1);
  conv_48to64_fp_8x1w(t[0], r0);
  conv_48to64_fp_4x2w(t[1], r1);

  for (i = 0; i < SWORDS; i++) {
    ret[0][2][0][i] = ((uint64_t *)&t[0][i])[0];
    ret[0][2][1][i] = ((uint64_t *)&t[0][i])[1];
    ret[0][1][0][i] = ((uint64_t *)&t[0][i])[2];
    ret[0][1][1][i] = ((uint64_t *)&t[0][i])[3];
    ret[1][0][0][i] = ((uint64_t *)&t[0][i])[4];
    ret[1][0][1][i] = ((uint64_t *)&t[0][i])[5];
    ret[1][1][0][i] = ((uint64_t *)&t[0][i])[6];
    ret[1][1][1][i] = ((uint64_t *)&t[0][i])[7];
  }

  for (i = 0; i < SWORDS/2; i++) {
    ret[0][0][0][i         ] = ((uint64_t *)&t[1][i])[0];
    ret[0][0][0][i+SWORDS/2] = ((uint64_t *)&t[1][i])[1];
    ret[0][0][1][i         ] = ((uint64_t *)&t[1][i])[2];
    ret[0][0][1][i+SWORDS/2] = ((uint64_t *)&t[1][i])[3];
    ret[1][2][0][i         ] = ((uint64_t *)&t[1][i])[4];
    ret[1][2][0][i+SWORDS/2] = ((uint64_t *)&t[1][i])[5];
    ret[1][2][1][i         ] = ((uint64_t *)&t[1][i])[6];
    ret[1][2][1][i+SWORDS/2] = ((uint64_t *)&t[1][i])[7];
  }
}

void sqr_fp12_scalar(vec384fp12 ret, const vec384fp12 a)
{
    vec384fp6 t0, t1;

    add_fp6(t0, a[0], a[1]);
    mul_by_u_plus_1_fp2(t1[2], a[1][2]);
    add_fp2(t1[0], a[0][0], t1[2]);
    add_fp2(t1[1], a[0][1], a[1][0]);
    add_fp2(t1[2], a[0][2], a[1][1]);
    mul_fp6(t0, t0, t1);
    mul_fp6(t1, a[0], a[1]);

    /* ret[1] = 2*(a0*a1) */
    add_fp6(ret[1], t1, t1);

    /* ret[0] = (a0 + a1)*(a0 + a1*v) - a0*a1 - a0*a1*v
              = a0^2 + a1^2*v */
    sub_fp6(ret[0], t0, t1);
    mul_by_u_plus_1_fp2(t1[2], t1[2]);
    sub_fp2(ret[0][0], ret[0][0], t1[2]);
    sub_fp2(ret[0][1], ret[0][1], t1[0]);
    sub_fp2(ret[0][2], ret[0][2], t1[1]);
}

void sqr_fp12_vector(vec384fp12 ret, const vec384fp12 a)
{
  fp2_4x2x1w r0, r1, a0, a1;
  __m512i t[2][SWORDS];
  int i;

  for (i = 0; i < SWORDS; i++) {
    t[0][i] = VSET(a[1][2][0][i], a[1][2][0][i],
                   a[0][2][1][i], a[0][2][0][i],
                   a[0][1][1][i], a[0][1][0][i],
                   a[0][0][1][i], a[0][0][0][i]);
    t[1][i] = VSET(a[1][2][1][i], a[1][2][1][i],
                   a[1][2][1][i], a[1][2][0][i],
                   a[1][1][1][i], a[1][1][0][i],
                   a[1][0][1][i], a[1][0][0][i]);
  }
  conv_64to48_fp_8x1w(a0, t[0]);
  conv_64to48_fp_8x1w(a1, t[1]);

  sqr_fp12_vec_v1(r0, r1, a0, a1);

  conv_48to64_fp_8x1w(t[0], r0);
  conv_48to64_fp_8x1w(t[1], r1);

  for (i = 0; i < SWORDS; i++) {
    ret[0][0][0][i] = ((uint64_t *)&t[0][i])[0];
    ret[0][0][1][i] = ((uint64_t *)&t[0][i])[1];
    ret[0][1][0][i] = ((uint64_t *)&t[0][i])[2];
    ret[0][1][1][i] = ((uint64_t *)&t[0][i])[3];
    ret[0][2][0][i] = ((uint64_t *)&t[0][i])[4];
    ret[0][2][1][i] = ((uint64_t *)&t[0][i])[5];
    ret[1][0][0][i] = ((uint64_t *)&t[1][i])[0];
    ret[1][0][1][i] = ((uint64_t *)&t[1][i])[1];
    ret[1][1][0][i] = ((uint64_t *)&t[1][i])[2];
    ret[1][1][1][i] = ((uint64_t *)&t[1][i])[3];
    ret[1][2][0][i] = ((uint64_t *)&t[1][i])[4];
    ret[1][2][1][i] = ((uint64_t *)&t[1][i])[5];
  }
}

void conjugate_fp12(vec384fp12 a)
{   neg_fp6(a[1], a[1]);   }

static void inverse_fp6(vec384fp6 ret, const vec384fp6 a)
{
    vec384x c0, c1, c2, t0, t1;

    /* c0 = a0^2 - (a1*a2)*(u+1) */
    sqr_fp2(c0, a[0]);
    mul_fp2(t0, a[1], a[2]);
    mul_by_u_plus_1_fp2(t0, t0);
    sub_fp2(c0, c0, t0);

    /* c1 = a2^2*(u+1) - (a0*a1) */
    sqr_fp2(c1, a[2]);
    mul_by_u_plus_1_fp2(c1, c1);
    mul_fp2(t0, a[0], a[1]);
    sub_fp2(c1, c1, t0);

    /* c2 = a1^2 - a0*a2 */
    sqr_fp2(c2, a[1]);
    mul_fp2(t0, a[0], a[2]);
    sub_fp2(c2, c2, t0);

    /* (a2*c1 + a1*c2)*(u+1) + a0*c0 */
    mul_fp2(t0, c1, a[2]);
    mul_fp2(t1, c2, a[1]);
    add_fp2(t0, t0, t1);
    mul_by_u_plus_1_fp2(t0, t0);
    mul_fp2(t1, c0, a[0]);
    add_fp2(t0, t0, t1);

    reciprocal_fp2(t1, t0);

    mul_fp2(ret[0], c0, t1);
    mul_fp2(ret[1], c1, t1);
    mul_fp2(ret[2], c2, t1);
}

void inverse_fp12(vec384fp12 ret, const vec384fp12 a)
{
    vec384fp6 t0, t1;

    sqr_fp6(t0, a[0]);
    sqr_fp6(t1, a[1]);
    mul_by_u_plus_1_fp2(t1[2], t1[2]);
    sub_fp2(t0[0], t0[0], t1[2]);
    sub_fp2(t0[1], t0[1], t1[0]);
    sub_fp2(t0[2], t0[2], t1[1]);

    inverse_fp6(t1, t0);

    mul_fp6(ret[0], a[0], t1);
    mul_fp6(ret[1], a[1], t1);
    neg_fp6(ret[1], ret[1]);
}

#ifndef BENCHMARK
static 
#endif
void sqr_fp4(vec384fp4 ret, const vec384x a0, const vec384x a1)
{
    vec768x t0, t1, t2;

    sqr_fp2x2(t0, a0);
    sqr_fp2x2(t1, a1);
    add_fp2(ret[1], a0, a1);

    mul_by_u_plus_1_fp2x2(t2, t1);
    add_fp2x2(t2, t2, t0);
    redc_fp2x2(ret[0], t2);

    sqr_fp2x2(t2, ret[1]);
    sub_fp2x2(t2, t2, t0);
    sub_fp2x2(t2, t2, t1);
    redc_fp2x2(ret[1], t2);
}

void cyclotomic_sqr_fp12_scalar(vec384fp12 ret, const vec384fp12 a)
{
    vec384fp4 t0, t1, t2;

    sqr_fp4(t0, a[0][0], a[1][1]);
    sqr_fp4(t1, a[1][0], a[0][2]);
    sqr_fp4(t2, a[0][1], a[1][2]);

    sub_fp2(ret[0][0], t0[0],     a[0][0]);
    add_fp2(ret[0][0], ret[0][0], ret[0][0]);
    add_fp2(ret[0][0], ret[0][0], t0[0]);

    sub_fp2(ret[0][1], t1[0],     a[0][1]);
    add_fp2(ret[0][1], ret[0][1], ret[0][1]);
    add_fp2(ret[0][1], ret[0][1], t1[0]);

    sub_fp2(ret[0][2], t2[0],     a[0][2]);
    add_fp2(ret[0][2], ret[0][2], ret[0][2]);
    add_fp2(ret[0][2], ret[0][2], t2[0]);

    mul_by_u_plus_1_fp2(t2[1], t2[1]);
    add_fp2(ret[1][0], t2[1],     a[1][0]);
    add_fp2(ret[1][0], ret[1][0], ret[1][0]);
    add_fp2(ret[1][0], ret[1][0], t2[1]);

    add_fp2(ret[1][1], t0[1],     a[1][1]);
    add_fp2(ret[1][1], ret[1][1], ret[1][1]);
    add_fp2(ret[1][1], ret[1][1], t0[1]);

    add_fp2(ret[1][2], t1[1],     a[1][2]);
    add_fp2(ret[1][2], ret[1][2], ret[1][2]);
    add_fp2(ret[1][2], ret[1][2], t1[1]);
}

void cyclotomic_sqr_fp12_vector(vec384fp12 ret, const vec384fp12 a)
{
  fp4_1x2x2x2w _ra, _a;
  fp4_2x2x2x1w _rbc, _bc;
  __m512i t[SWORDS], s[SWORDS/2];
  uint64_t r48[NWORDS];
  int i;

  // form < a11 | a00 >
  for (i = 0; i < SWORDS/2; i++) {
    s[i] = VSET(a[1][1][1][i+SWORDS/2], a[1][1][1][i],
                a[1][1][0][i+SWORDS/2], a[1][1][0][i],
                a[0][0][1][i+SWORDS/2], a[0][0][1][i],
                a[0][0][0][i+SWORDS/2], a[0][0][0][i] );
  }
  conv_64to48_fp_4x2w(_a, s);

  // form < a12 | a01 | a02 | a10 >
  for (i = 0; i < SWORDS; i++) {
    t[i] = VSET(a[1][2][1][i], a[1][2][0][i], 
                a[0][1][1][i], a[0][1][0][i], 
                a[0][2][1][i], a[0][2][0][i], 
                a[1][0][1][i], a[1][0][0][i] );
  }
  conv_64to48_fp_8x1w(_bc, t);

  cyclotomic_sqr_fp12_vec_v1(_ra, _rbc, _a,  _bc);

  carryp_fp_4x2w(_ra);
  conv_48to64_fp_4x2w(s, _ra);
  for(i = 0; i < SWORDS/2; i++) {
    ret[0][0][0][i         ] = ((uint64_t *)&s[i])[0];
    ret[0][0][0][i+SWORDS/2] = ((uint64_t *)&s[i])[1];
    ret[0][0][1][i         ] = ((uint64_t *)&s[i])[2];
    ret[0][0][1][i+SWORDS/2] = ((uint64_t *)&s[i])[3];
    ret[1][1][0][i         ] = ((uint64_t *)&s[i])[4];
    ret[1][1][0][i+SWORDS/2] = ((uint64_t *)&s[i])[5];
    ret[1][1][1][i         ] = ((uint64_t *)&s[i])[6];
    ret[1][1][1][i+SWORDS/2] = ((uint64_t *)&s[i])[7];
  }

  conv_48to64_fp_8x1w(t, _rbc);
  for (i = 0; i < SWORDS; i++) {
    ret[1][0][0][i] = ((uint64_t *)&t[i])[0];
    ret[1][0][1][i] = ((uint64_t *)&t[i])[1];
    ret[0][2][0][i] = ((uint64_t *)&t[i])[2];
    ret[0][2][1][i] = ((uint64_t *)&t[i])[3];
    ret[0][1][0][i] = ((uint64_t *)&t[i])[4];
    ret[0][1][1][i] = ((uint64_t *)&t[i])[5];
    ret[1][2][0][i] = ((uint64_t *)&t[i])[6];
    ret[1][2][1][i] = ((uint64_t *)&t[i])[7];
  }
}

/*
 * caveat lector! |n| has to be non-zero and not more than 3!
 */
static inline void frobenius_map_fp2(vec384x ret, const vec384x a, size_t n)
{
    vec_copy(ret[0], a[0], sizeof(ret[0]));
    cneg_fp(ret[1], a[1], n & 1);
}

static void frobenius_map_fp6(vec384fp6 ret, const vec384fp6 a, size_t n)
{
    static const vec384x coeffs1[] = {  /* (u + 1)^((P^n - 1) / 3) */
      { { 0 },
        { TO_LIMB_T(0xcd03c9e48671f071), TO_LIMB_T(0x5dab22461fcda5d2),
          TO_LIMB_T(0x587042afd3851b95), TO_LIMB_T(0x8eb60ebe01bacb9e),
          TO_LIMB_T(0x03f97d6e83d050d2), TO_LIMB_T(0x18f0206554638741) } },
      { { TO_LIMB_T(0x30f1361b798a64e8), TO_LIMB_T(0xf3b8ddab7ece5a2a),
          TO_LIMB_T(0x16a8ca3ac61577f7), TO_LIMB_T(0xc26a2ff874fd029b),
          TO_LIMB_T(0x3636b76660701c6e), TO_LIMB_T(0x051ba4ab241b6160) } },
      { { 0 }, { ONE_MONT_P } }
    };
    static const vec384 coeffs2[] = {  /* (u + 1)^((2P^n - 2) / 3) */
      {   TO_LIMB_T(0x890dc9e4867545c3), TO_LIMB_T(0x2af322533285a5d5),
          TO_LIMB_T(0x50880866309b7e2c), TO_LIMB_T(0xa20d1b8c7e881024),
          TO_LIMB_T(0x14e4f04fe2db9068), TO_LIMB_T(0x14e56d3f1564853a)   },
      {   TO_LIMB_T(0xcd03c9e48671f071), TO_LIMB_T(0x5dab22461fcda5d2),
          TO_LIMB_T(0x587042afd3851b95), TO_LIMB_T(0x8eb60ebe01bacb9e),
          TO_LIMB_T(0x03f97d6e83d050d2), TO_LIMB_T(0x18f0206554638741)   },
      {   TO_LIMB_T(0x43f5fffffffcaaae), TO_LIMB_T(0x32b7fff2ed47fffd),
          TO_LIMB_T(0x07e83a49a2e99d69), TO_LIMB_T(0xeca8f3318332bb7a),
          TO_LIMB_T(0xef148d1ea0f4c069), TO_LIMB_T(0x040ab3263eff0206)   }
    };

    frobenius_map_fp2(ret[0], a[0], n);
    frobenius_map_fp2(ret[1], a[1], n);
    frobenius_map_fp2(ret[2], a[2], n);
    --n;    /* implied ONE_MONT_P at index 0 */
    mul_fp2(ret[1], ret[1], coeffs1[n]);
    mul_fp(ret[2][0], ret[2][0], coeffs2[n]);
    mul_fp(ret[2][1], ret[2][1], coeffs2[n]);
}

void frobenius_map_fp12(vec384fp12 ret, const vec384fp12 a, size_t n)
{
    static const vec384x coeffs[] = {  /* (u + 1)^((P^n - 1) / 6) */
      { { TO_LIMB_T(0x07089552b319d465), TO_LIMB_T(0xc6695f92b50a8313),
          TO_LIMB_T(0x97e83cccd117228f), TO_LIMB_T(0xa35baecab2dc29ee),
          TO_LIMB_T(0x1ce393ea5daace4d), TO_LIMB_T(0x08f2220fb0fb66eb) },
	{ TO_LIMB_T(0xb2f66aad4ce5d646), TO_LIMB_T(0x5842a06bfc497cec),
          TO_LIMB_T(0xcf4895d42599d394), TO_LIMB_T(0xc11b9cba40a8e8d0),
          TO_LIMB_T(0x2e3813cbe5a0de89), TO_LIMB_T(0x110eefda88847faf) } },
      { { TO_LIMB_T(0xecfb361b798dba3a), TO_LIMB_T(0xc100ddb891865a2c),
          TO_LIMB_T(0x0ec08ff1232bda8e), TO_LIMB_T(0xd5c13cc6f1ca4721),
          TO_LIMB_T(0x47222a47bf7b5c04), TO_LIMB_T(0x0110f184e51c5f59) } },
      { { TO_LIMB_T(0x3e2f585da55c9ad1), TO_LIMB_T(0x4294213d86c18183),
          TO_LIMB_T(0x382844c88b623732), TO_LIMB_T(0x92ad2afd19103e18),
          TO_LIMB_T(0x1d794e4fac7cf0b9), TO_LIMB_T(0x0bd592fc7d825ec8) },
	{ TO_LIMB_T(0x7bcfa7a25aa30fda), TO_LIMB_T(0xdc17dec12a927e7c),
          TO_LIMB_T(0x2f088dd86b4ebef1), TO_LIMB_T(0xd1ca2087da74d4a7),
          TO_LIMB_T(0x2da2596696cebc1d), TO_LIMB_T(0x0e2b7eedbbfd87d2) } },
    };

    frobenius_map_fp6(ret[0], a[0], n);
    frobenius_map_fp6(ret[1], a[1], n);
    --n;    /* implied ONE_MONT_P at index 0 */
    mul_fp2(ret[1][0], ret[1][0], coeffs[n]);
    mul_fp2(ret[1][1], ret[1][1], coeffs[n]);
    mul_fp2(ret[1][2], ret[1][2], coeffs[n]);
}

// ----------------------------------------------------------

void compressed_cyclotomic_sqr_fp12_scalar(vec384fp12 ret, const vec384fp12 a)
{
  vec384fp2 t0, t1, t2, t3, t4, t5, t6;

  sqr_fp2(t0, a[0][1]);
  sqr_fp2(t1, a[1][2]);
  add_fp2(t5, a[0][1], a[1][2]);
  sqr_fp2(t2, t5);

  add_fp2(t3, t0, t1);
  sub_fp2(t5, t2, t3);

  add_fp2(t6, a[1][0], a[0][2]);
  sqr_fp2(t3, t6);
  sqr_fp2(t2, a[1][0]);

  mul_by_u_plus_1_fp2(t6, t5);
  add_fp2(t5, t6, a[1][0]);
  add_fp2(t5, t5, t5);
  add_fp2(ret[1][0], t5, t6);

  mul_by_u_plus_1_fp2(t4, t1);
  add_fp2(t5, t0, t4);
  sub_fp2(t6, t5, a[0][2]);

  sqr_fp2(t1, a[0][2]);

  add_fp2(t6, t6, t6);
  add_fp2(ret[0][2], t6, t5);

  mul_by_u_plus_1_fp2(t4, t1);
  add_fp2(t5, t2, t4);
  sub_fp2(t6, t5, a[0][1]);
  add_fp2(t6, t6, t6);
  add_fp2(ret[0][1], t6, t5);

  add_fp2(t0, t2, t1);
  sub_fp2(t5, t3, t0);
  add_fp2(t6, t5, a[1][2]);
  add_fp2(t6, t6, t6);
  add_fp2(ret[1][2], t5, t6);
}

static void reciprocal_sim_fp2(vec384fp2 ret[], const vec384fp2 a[], const int n)
{

  vec384fp2 t[n], u;
  int i;

  vec_copy(ret[0], a[0], sizeof(vec384fp2));
  vec_copy(t[0], a[0], sizeof(vec384fp2));

  for (i = 1; i < n; i++) {
    vec_copy(t[i], a[i], sizeof(vec384fp2));
    mul_fp2(ret[i], ret[i - 1], t[i]);
  }

  reciprocal_fp2(u, ret[n - 1]);

  for (i = n - 1; i > 0; i--) {
    mul_fp2(ret[i], ret[i - 1], u);
    mul_fp2(u, u, t[i]);
  }
  vec_copy(ret[0], u, sizeof(vec384fp2));
}

void back_cyclotomic_sim_fp12(vec384fp12 ret[], const vec384fp12 a[], const int n)
{
  vec384fp2 t0[n], t1[n], t2[n];
  vec384 t3 = { ONE_MONT_P };
  int i;

  for (i = 0; i < n; i++) {
    /* t0 = g4^2. */
    sqr_fp2(t0[i], a[i][0][1]);

    /* t1 = 3 * g4^2 - 2 * g3. */
    sub_fp2(t1[i], t0[i], a[i][0][2]);
    add_fp2(t1[i], t1[i], t1[i]);
    add_fp2(t1[i], t1[i], t0[i]);

    /* t0 = E * g5^2 + t1. */
    sqr_fp2(t2[i], a[i][1][2]);
    mul_by_u_plus_1_fp2(t0[i], t2[i]);
    add_fp2(t0[i], t0[i], t1[i]);

    /* t1 = (4 * g2). */
    add_fp2(t1[i], a[i][1][0], a[i][1][0]);
    add_fp2(t1[i], t1[i], t1[i]);
  }

  /* t1 = 1 / t1. */
  reciprocal_sim_fp2(t1, t1, n);

  for (i = 0; i < n; i++) {
    /* t0 = g1. */
    mul_fp2(ret[i][1][1], t0[i], t1[i]);

    /* t1 = g3 * g4. */
    mul_fp2(t1[i], a[i][0][2], a[i][0][1]);
    
    /* t2 = 2 * g1^2 - 3 * g3 * g4. */
    sqr_fp2(t2[i], ret[i][1][1]);
    sub_fp2(t2[i], t2[i], t1[i]);
    add_fp2(t2[i], t2[i], t2[i]);
    sub_fp2(t2[i], t2[i], t1[i]);

    /* t1 = g2 * g5. */
    mul_fp2(t1[i], a[i][1][0], a[i][1][2]);

    /* c_0 = E * (2 * g1^2 + g2 * g5 - 3 * g3 * g4) + 1. */
    add_fp2(t2[i], t2[i], t1[i]);
    mul_by_u_plus_1_fp2(ret[i][0][0], t2[i]);
    add_fp(ret[i][0][0][0], ret[i][0][0][0], t3);
  }
}
