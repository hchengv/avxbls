/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pairing.h"

/*
 * Line evaluations from  https://eprint.iacr.org/2010/354.pdf
 * with a twist moving common expression to line_by_Px2.
 */
void line_add_scalar(vec384fp6 line, POINTonE2 *T, const POINTonE2 *R,
                                                   const POINTonE2_affine *Q)
{
    vec384x Z1Z1, U2, S2, H, HH, I, J, V;
#if 1
# define r line[1]
#else
    vec384x r;
#endif

    /*
     * https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
     * with XYZ3 being |T|, XYZ1 - |R|, XY2 - |Q|, i.e. Q is affine
     */
    sqr_fp2(Z1Z1, R->Z);                /* Z1Z1 = Z1^2 */
    mul_fp2(U2, Q->X, Z1Z1);            /* U2 = X2*Z1Z1 */

    mul_fp2(S2, Q->Y, R->Z);
    mul_fp2(S2, S2, Z1Z1);              /* S2 = Y2*Z1*Z1Z1 */

    sub_fp2(H, U2, R->X);               /* H = U2-X1 */

    sqr_fp2(HH, H);                     /* HH = H^2 */
    add_fp2(I, HH, HH);
    add_fp2(I, I, I);                   /* I = 4*HH */

    mul_fp2(J, H, I);                   /* J = H*I */

    sub_fp2(r, S2, R->Y);
    add_fp2(r, r, r);                   /* r = 2*(S2-Y1) */

    mul_fp2(V, R->X, I);                /* V = X1*I */

    sqr_fp2(T->X, r);
    sub_fp2(T->X, T->X, J);
    sub_fp2(T->X, T->X, V);
    sub_fp2(T->X, T->X, V);             /* X3 = r^2-J-2*V */

    mul_fp2(J, J, R->Y);
    sub_fp2(T->Y, V, T->X);
    mul_fp2(T->Y, T->Y, r);
    sub_fp2(T->Y, T->Y, J);
    sub_fp2(T->Y, T->Y, J);             /* Y3 = r*(V-X3)-2*Y1*J */

    add_fp2(T->Z, R->Z, H);
    sqr_fp2(T->Z, T->Z);
    sub_fp2(T->Z, T->Z, Z1Z1);
    sub_fp2(T->Z, T->Z, HH);            /* Z3 = (Z1+H)^2-Z1Z1-HH */

    /*
     * line evaluation
     */
    mul_fp2(I, r, Q->X);
    mul_fp2(J, Q->Y, T->Z);
    sub_fp2(I, I, J);
    add_fp2(line[0], I, I);          /* 2*(r*X2 - Y2*Z3) */
#ifdef r
# undef r
#else
    vec_copy(line[1], r, sizeof(r));
#endif
    vec_copy(line[2], T->Z, sizeof(T->Z));
}

void line_add_vector(vec384fp6 line, POINTonE2 *T, const POINTonE2 *R,
                                                   const POINTonE2_affine *Q)
{
  fp2_2x2x2w X1Y1, Z1Y2, X2, l0Y3, l1, X3, Z3;
  __m512i t[4][SWORDS/2];
  int i;

  for (i = 0; i < SWORDS/2; i++) {
    t[0][i] = VSET(R->X[1][i+SWORDS/2], R->X[1][i],
                   R->X[0][i+SWORDS/2], R->X[0][i],
                   R->Y[1][i+SWORDS/2], R->Y[1][i],
                   R->Y[0][i+SWORDS/2], R->Y[0][i]);
    t[1][i] = VSET(R->Z[1][i+SWORDS/2], R->Z[1][i],
                   R->Z[0][i+SWORDS/2], R->Z[0][i],
                   Q->Y[1][i+SWORDS/2], Q->Y[1][i],
                   Q->Y[0][i+SWORDS/2], Q->Y[0][i]);
    t[2][i] = VSET(Q->X[1][i+SWORDS/2], Q->X[1][i],
                   Q->X[0][i+SWORDS/2], Q->X[0][i],
                   Q->X[1][i+SWORDS/2], Q->X[1][i],
                   Q->X[0][i+SWORDS/2], Q->X[0][i]);
  }

  conv_64to48_fp_4x2w(X1Y1, t[0]);
  conv_64to48_fp_4x2w(Z1Y2, t[1]);
  conv_64to48_fp_4x2w(X2  , t[2]);

  line_add_vec_v1(l0Y3, l1, X3, Z3, X1Y1, Z1Y2, X2);

  carryp_fp_4x2w(l0Y3);
  carryp_fp_4x2w(l1);
  carryp_fp_4x2w(X3);
  carryp_fp_4x2w(Z3);

  conv_48to64_fp_4x2w(t[0], l0Y3);
  conv_48to64_fp_4x2w(t[1], l1);
  conv_48to64_fp_4x2w(t[2], X3);
  conv_48to64_fp_4x2w(t[3], Z3);

  for (i = 0; i < SWORDS/2; i++) {
    line   [0][0][i         ] = ((uint64_t *)&t[0][i])[0];
    line   [0][0][i+SWORDS/2] = ((uint64_t *)&t[0][i])[1];
    line   [0][1][i         ] = ((uint64_t *)&t[0][i])[2];
    line   [0][1][i+SWORDS/2] = ((uint64_t *)&t[0][i])[3];
    line   [1][0][i         ] = ((uint64_t *)&t[1][i])[0];
    line   [1][0][i+SWORDS/2] = ((uint64_t *)&t[1][i])[1];
    line   [1][1][i         ] = ((uint64_t *)&t[1][i])[2];
    line   [1][1][i+SWORDS/2] = ((uint64_t *)&t[1][i])[3];
    T   ->X[0][   i         ] = ((uint64_t *)&t[2][i])[4];
    T   ->X[0][   i+SWORDS/2] = ((uint64_t *)&t[2][i])[5];
    T   ->X[1][   i         ] = ((uint64_t *)&t[2][i])[6];
    T   ->X[1][   i+SWORDS/2] = ((uint64_t *)&t[2][i])[7];
    T   ->Y[0][   i         ] = ((uint64_t *)&t[0][i])[4];
    T   ->Y[0][   i+SWORDS/2] = ((uint64_t *)&t[0][i])[5];
    T   ->Y[1][   i         ] = ((uint64_t *)&t[0][i])[6];
    T   ->Y[1][   i+SWORDS/2] = ((uint64_t *)&t[0][i])[7];
    T   ->Z[0][   i         ] = ((uint64_t *)&t[3][i])[4];
    T   ->Z[0][   i+SWORDS/2] = ((uint64_t *)&t[3][i])[5];
    T   ->Z[1][   i         ] = ((uint64_t *)&t[3][i])[6];
    T   ->Z[1][   i+SWORDS/2] = ((uint64_t *)&t[3][i])[7];
    line   [2][0][i         ] = T   ->Z[0][   i         ];
    line   [2][0][i+SWORDS/2] = T   ->Z[0][   i+SWORDS/2];
    line   [2][1][i         ] = T   ->Z[1][   i         ];
    line   [2][1][i+SWORDS/2] = T   ->Z[1][   i+SWORDS/2];
  } 
}

void line_dbl_scalar(vec384fp6 line, POINTonE2 *T, const POINTonE2 *Q)
{
    vec384x ZZ, A, B, C, D, E, F;

    /*
     * https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-alnr
     */
    sqr_fp2(A, Q->X);                   /* A = X1^2 */
    sqr_fp2(B, Q->Y);                   /* B = Y1^2 */
    sqr_fp2(ZZ, Q->Z);                  /* ZZ = Z1^2 */
    sqr_fp2(C, B);                      /* C = B^2 */

    add_fp2(D, Q->X, B);                /* X1+B */
    sqr_fp2(D, D);                      /* (X1+B)^2 */
    sub_fp2(D, D, A);                   /* (X1+B)^2-A */
    sub_fp2(D, D, C);                   /* (X1+B)^2-A-C */
    add_fp2(D, D, D);                   /* D = 2*((X1+B)^2-A-C) */

    mul_by_3_fp2(E, A);                 /* E = 3*A */
    sqr_fp2(F, E);                      /* F = E^2 */

    add_fp2(line[0], E, Q->X);          /* 3*A+X1 for line evaluation */

    sub_fp2(T->X, F, D);
    sub_fp2(T->X, T->X, D);             /* X3 = F-2*D */

    add_fp2(T->Z, Q->Y, Q->Z);
    sqr_fp2(T->Z, T->Z);
    sub_fp2(T->Z, T->Z, B);
    sub_fp2(T->Z, T->Z, ZZ);            /* Z3 = (Y1+Z1)^2-B-ZZ */

    mul_by_8_fp2(C, C);                 /* 8*C */
    sub_fp2(T->Y, D, T->X);             /* D-X3 */
    mul_fp2(T->Y, T->Y, E);             /* E*(D-X3) */
    sub_fp2(T->Y, T->Y, C);             /* Y3 = E*(D-X3)-8*C */

    /*
     * line evaluation
     */
    sqr_fp2(line[0], line[0]);
    sub_fp2(line[0], line[0], A);
    sub_fp2(line[0], line[0], F);       /* (3*A+X1)^2 - X1^2 - 9*A^2 */
    lshift_fp2(B, B, 2);
    sub_fp2(line[0], line[0], B);       /* 6*X1^3 - 4*Y1^2 */

    mul_fp2(line[1], E, ZZ);            /* 3*X1^2 * Z1^2 */

    mul_fp2(line[2], T->Z, ZZ);         /* Z3 * Z1^2 */
}

void line_dbl_vector(vec384fp6 line, POINTonE2 *T, const POINTonE2 *Q)
{
  fp2_4x2x1w X1Y1Z1, l0, l12, X3, Y3, Z3;
  __m512i t[5][SWORDS];
  int i;

  for (i = 0; i < SWORDS; i++) {
    t[0][i] = VSET(Q->Y[1][i], Q->Y[0][i], Q->X[1][i], Q->X[0][i],
                   Q->Z[1][i], Q->Z[0][i], Q->Y[1][i], Q->Y[0][i]);
  }
  conv_64to48_fp_8x1w(X1Y1Z1, t[0]);

  line_dbl_vec_v2(l0, l12, X3, Y3, Z3, X1Y1Z1);

  conv_48to64_fp_8x1w(t[0], l0);
  conv_48to64_fp_8x1w(t[1], l12);
  conv_48to64_fp_8x1w(t[2], X3);
  conv_48to64_fp_8x1w(t[3], Y3);
  conv_48to64_fp_8x1w(t[4], Z3);

  for (i = 0; i < SWORDS; i++) {
    line   [0][0][i] = ((uint64_t *)&t[0][i])[2];
    line   [0][1][i] = ((uint64_t *)&t[0][i])[3];
    line   [1][0][i] = ((uint64_t *)&t[1][i])[4];
    line   [1][1][i] = ((uint64_t *)&t[1][i])[5];
    line   [2][0][i] = ((uint64_t *)&t[1][i])[2];
    line   [2][1][i] = ((uint64_t *)&t[1][i])[3];
    T   ->X[0][   i] = ((uint64_t *)&t[2][i])[6];
    T   ->X[1][   i] = ((uint64_t *)&t[2][i])[7];
    T   ->Y[0][   i] = ((uint64_t *)&t[3][i])[6];
    T   ->Y[1][   i] = ((uint64_t *)&t[3][i])[7];
    T   ->Z[0][   i] = ((uint64_t *)&t[4][i])[0];
    T   ->Z[1][   i] = ((uint64_t *)&t[4][i])[1];
  }
}

static void line_by_Px2(vec384fp6 line, const POINTonE1_affine *Px2)
{
    mul_fp(line[1][0], line[1][0], Px2->X);   /* "b01" *= -2*P->X */
    mul_fp(line[1][1], line[1][1], Px2->X);

    mul_fp(line[2][0], line[2][0], Px2->Y);   /* "b11" *= 2*P->Y */
    mul_fp(line[2][1], line[2][1], Px2->Y);
}

static void start_dbl_n(vec384fp12 ret, POINTonE2 T[],
                                        const POINTonE1_affine Px2[], size_t n)
{
    size_t i;
    vec384fp6 line; /* it's not actual fp6, but 3 packed fp2, "xy00z0"  */

    /* first step is ret = 1^2*line, which is replaced with ret = line  */
    line_dbl(line, T+0, T+0);           line_by_Px2(line, Px2+0);
    vec_zero(ret, sizeof(vec384fp12));
    vec_copy(ret[0][0], line[0], 2*sizeof(vec384fp2));
    vec_copy(ret[1][1], line[2], sizeof(vec384fp2));

    for (i = 1; i < n; i++) {
        line_dbl(line, T+i, T+i);       line_by_Px2(line, Px2+i);
        mul_by_xy00z0_fp12(ret, ret, line);
    }
}

static void add_n_dbl_n_scalar(vec384fp12 ret, POINTonE2 T[],
                                               const POINTonE2_affine Q[],
                                               const POINTonE1_affine Px2[],
                                               size_t n, size_t k)
{
    size_t i;
    vec384fp6 line; /* it's not actual fp6, but 3 packed fp2, "xy00z0"  */

    for (i = 0; i < n; i++) {
        line_add(line, T+i, T+i, Q+i);  line_by_Px2(line, Px2+i);
        mul_by_xy00z0_fp12(ret, ret, line);
    }
    while (k--) {
        sqr_fp12(ret, ret);
        for (i = 0; i < n; i++) {
            line_dbl(line, T+i, T+i);   line_by_Px2(line, Px2+i);
            mul_by_xy00z0_fp12(ret, ret, line);
        }
    }
}

static void add_n_dbl_n_vector(vec384fp12 ret, POINTonE2 T[],
                                               const POINTonE2_affine Q[],
                                               const POINTonE1_affine Px2[],
                                               size_t n, size_t k)
{
  __m512i t[3][SWORDS], s[4][SWORDS/2];
  int i;

  // Step 1: line_add
  fp2_2x2x2w X1Y1, Z1Y2, X2, l0Y3, l1, X3, Z3;
  const __m512i hh = VSET(7, 6, 5, 4, 7, 6, 5, 4);

  // X1Y1 = X1 | Y1 at Fp2 layer
  // Z1Y2 = Z1 | Y2 at Fp2 layer
  //   X2 = X2 | X2 at Fp2 layer
  for (i = 0; i < SWORDS/2; i++) {
    s[0][i] = VSET(T->X[1][i+SWORDS/2], T->X[1][i],
                   T->X[0][i+SWORDS/2], T->X[0][i],
                   T->Y[1][i+SWORDS/2], T->Y[1][i],
                   T->Y[0][i+SWORDS/2], T->Y[0][i]);
    s[1][i] = VSET(T->Z[1][i+SWORDS/2], T->Z[1][i],
                   T->Z[0][i+SWORDS/2], T->Z[0][i],
                   Q->Y[1][i+SWORDS/2], Q->Y[1][i],
                   Q->Y[0][i+SWORDS/2], Q->Y[0][i]);
    s[2][i] = VSET(Q->X[1][i+SWORDS/2], Q->X[1][i],
                   Q->X[0][i+SWORDS/2], Q->X[0][i],
                   Q->X[1][i+SWORDS/2], Q->X[1][i],
                   Q->X[0][i+SWORDS/2], Q->X[0][i]);
  }
  
  conv_64to48_fp_4x2w(X1Y1, s[0]);
  conv_64to48_fp_4x2w(Z1Y2, s[1]);
  conv_64to48_fp_4x2w(X2  , s[2]);

  line_add_vec_v1(l0Y3, l1, X3, Z3, X1Y1, Z1Y2, X2);

  perm_var_hl(X3, X3, hh);
  perm_var_hl(Z3, Z3, hh);

  // l0Y3 = Y3 | l0 at Fp2 layer
  // l1   = .. | l1 at Fp2 layer
  // l2   = Z3 | Z3 at Fp2 layer
  // Z3   = Z3 | Z3 at Fp2 layer
  // X3   = X3 | X3 at Fp2 layer

  // Step 2: line_by_Px2
  fp2_2x2x2w l12, _Px2;

  // l12  = l2 | l1 at Fp2 layer 
  // _Px2 = Px2->Y | Px2->X at Fp2 layer
  blend_0x0F_hl(l12, Z3, l1);
  for (i = 0; i < SWORDS/2; i++) {
      s[0][i] = VSET(Px2->Y[i+SWORDS/2], Px2->Y[i],
                     Px2->Y[i+SWORDS/2], Px2->Y[i],
                     Px2->X[i+SWORDS/2], Px2->X[i],
                     Px2->X[i+SWORDS/2], Px2->X[i]);
  }
  conv_64to48_fp_4x2w(_Px2, s[0]);

  line_by_Px2_2x2x2w(l12, l12, _Px2);

  // l12  = l2 | l1 at Fp2 layer 

  // Step 3:  mul_by_xy00z0_fp12
  fp2_4x2x1w a01, a2, b01, b4, r0;
  fp2_2x2x2w r1;
  const __m512i m0 = VSET(6, 4, 6, 4, 6, 4, 6, 4);
  const __m512i m1 = VSET(7, 5, 7, 5, 7, 5, 7, 5);
  const __m512i m2 = VSET(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i m3 = VSET(6, 4, 2, 0, 6, 4, 2, 0);
  const __m512i m4 = VSET(7, 5, 3, 1, 7, 5, 3, 1);

  // a01 = ret[1][1] | ret[1][0] | ret[0][1] | ret[0][0] at Fp2 layer
  // a2  = ret[1][2] | ret[1][2] | ret[0][2] | ret[0][2] at Fp2 layer
  // b01 =   line[1] |   line[0] |   line[1] |   line[0] at Fp2 layer
  // b4  =   line[2] |   line[2] |   line[2] |   line[2] at Fp2 layer
  for (i = 0; i < SWORDS; i++) {
    t[0][i] = VSET(ret[1][1][1][i], ret[1][1][0][i],
                   ret[1][0][1][i], ret[1][0][0][i],
                   ret[0][1][1][i], ret[0][1][0][i],
                   ret[0][0][1][i], ret[0][0][0][i]);
    t[1][i] = VSET(ret[1][2][1][i], ret[1][2][0][i],
                   ret[1][2][1][i], ret[1][2][0][i],
                   ret[0][2][1][i], ret[0][2][0][i],
                   ret[0][2][1][i], ret[0][2][0][i]);
  }
  conv_64to48_fp_8x1w(a01, t[0]);
  conv_64to48_fp_8x1w(a2 , t[1]);

  carryp_fp_4x2w(l12);
  carryp_fp_4x2w(l0Y3);

  for (i = 0; i < VWORDS; i++) {
    b4[i]        = VPERMV(m0, l12[i]);
    b4[i+VWORDS] = VPERMV(m1, l12[i]);
  }
  perm_var_hl(l12, l12, m2);
  blend_0x0F_hl(l12, l12, l0Y3);
  for (i = 0; i < VWORDS; i++) {
    b01[i]        = VPERMV(m3, l12[i]);
    b01[i+VWORDS] = VPERMV(m4, l12[i]);
  }

  mul_by_xy00z0_fp12_vec_v1(r0, r1, a01, a2, b01, b4);

  // r0 = ret[1][1] | ret[1][0] | ret[0][1] | ret[0][2] at Fp2 layer
  // r1 =             ret[1][2] |             ret[0][0] at Fp2 layer

  // while-loop
  fp2_4x2x1w a0, a1, _r0, _r1;
  // fp2_2x2x2w l2, _Z3;
  fp2_4x2x1w X1Y1Z1, _l0, _l12, _X3, _Y3, _Z3, __Px2;
  fp2_2x2x2w t0;
  const __m512i m5 = VSET(0, 0, 1, 0, 3, 2, 0, 0); 
  const __m512i m6 = VSET(4, 4, 0, 0, 0, 0, 2, 0);
  const __m512i m7 = VSET(5, 5, 0, 0, 0, 0, 3, 1);
  const __m512i m8 = VSET(6, 6, 6, 4, 0, 0, 0, 0);
  const __m512i m9 = VSET(7, 7, 7, 5, 0, 0, 0, 0);
  const __m512i m10 = VSET(5, 4, 5, 4, 5, 4, 5, 4);
  const __m512i m11 = VSET(6, 4, 6, 4, 2, 0, 2, 0);
  const __m512i m12 = VSET(7, 5, 7, 5, 3, 1, 3, 1);
  const __m512i m13 = VSET(3, 2, 3, 2, 3, 2, 3, 2);
  const __m512i m14 = VSET(5, 4, 5, 4, 5, 4, 5, 4);
  const __m512i m15 = VSET(0, 0, 7, 6, 1, 0, 0, 0);
  const __m512i m16 = VSET(7, 6, 7, 6, 7, 6, 7, 6);
  int first_iteration = 1;

  for (i = 0; i < SWORDS; i++) {
      t[0][i] = VSET(Px2->X[i], Px2->X[i], Px2->X[i], Px2->X[i],
                     Px2->Y[i], Px2->Y[i], Px2->Y[i], Px2->Y[i]);
  }

  conv_64to48_fp_8x1w(__Px2, t[0]);

  while (k--) {
    // Step 4: sqr_fp12 

    //  a0 = ret[1][2]00 | ret[0][2] | ret[0][1] | ret[0][0] at Fp2 layer
    //  a1 = ret[1][2]11 | ret[1][2] | ret[1][1] | ret[1][0] at Fp2 layer

    //  a0 =         ... | ret[0][2] | ret[0][1] |       ... at Fp2 layer
    perm_var(a0, r0, m5);
    //  a1 =         ... |       ... | ret[1][1] | ret[1][0] at Fp2 layer
    perm_var(a1, r0, m2);
    // _r0 = ret[1][2]00 |       ... |       ... | ret[0][0] at Fp2 layer
    // _r1 = ret[1][2]11 | ret[1][2] |       ... |       ... at Fp2 layer
    for (i = 0; i < VWORDS; i++) {
      _r0[i       ] = VPERMV(m6, r1[i]);
      _r0[i+VWORDS] = VPERMV(m7, r1[i]);
      _r1[i       ] = VPERMV(m8, r1[i]);
      _r1[i+VWORDS] = VPERMV(m9, r1[i]);
    }
    //  a0 =         ... | ret[0][2] | ret[0][1] |       ... at Fp2 layer
    blend_0xC3(a0, a0, _r0); 
    //  a1 = ret[1][2]11 | ret[1][2] | ret[1][1] | ret[1][0] at Fp2 layer
    blend_0x0F(a1, _r1, a1);

    sqr_fp12_vec_v1(_r0, _r1, a0, a1);

    // _r0 =  ... | ret[0][2] | ret[0][1] | ret[0][0] at Fp2 layer 
    // _r1 =  ... | ret[1][2] | ret[1][1] | ret[1][0] at Fp2 layer

    // Step 5: line_dbl
    // X1Y1Z1 = Y1 | X1 | Z1 | Y1 at Fp2 layer

    if (first_iteration) {
      // t0 = X1 | Z1 at Fp2 layer
      blend_0x0F_hl(t0, X3, Z3);
      // X1Y1Z1 = X1 | X1 | Z1 | Z1 at Fp2 layer 
      for (i = 0; i < VWORDS; i++) {
        X1Y1Z1[i]        = VPERMV(m11, t0[i]);
        X1Y1Z1[i+VWORDS] = VPERMV(m12, t0[i]);
      }
      // _Y3 =  Y1 | Y1 | Y1 | Y1 at Fp2 layer
      for (i = 0; i < VWORDS; i++) {
        _Y3[i]        = VPERMV(m0, l0Y3[i]);
        _Y3[i+VWORDS] = VPERMV(m1, l0Y3[i]);
      }
      // X1Y1Z1 = Y1 | X1 | Z1 | Y1 at Fp2 layer
      blend_0xC3(X1Y1Z1, X1Y1Z1, _Y3);

      first_iteration = 0;
    }
    else {
      // X1Y1Z1 = X1 | .. | .. | Z1 at Fp2 layer
      blend_0x0F(X1Y1Z1, _X3, _Z3);
      // X1Y1Z1 = .. | X1 | Z1 | .. at Fp2 layer
      perm_var(X1Y1Z1, X1Y1Z1, m15);
      //    _Y3 = Y1 | Y1 | Y1 | Y1 at Fp2 layer 
      perm_var(_Y3, _Y3, m16);
      // X1Y1Z1 = Y1 | X1 | Z1 | Y1 at Fp2 layer
      blend_0xC3(X1Y1Z1, X1Y1Z1, _Y3);
    }

    line_dbl_vec_v2(_l0, _l12, _X3, _Y3, _Z3, X1Y1Z1);

    // _l0  = .. | .. | l0 | .. at Fp2 layer
    // _l12 = .. | l1 | l2 | .. at Fp2 layer
    // _X3  = X3 | .. | .. | .. at Fp2 layer
    // _Y3  = Y3 | .. | .. | .. at Fp2 layer
    // _Z3  = .. | .. | .. | Z3 at Fp2 layer 

    // Step 6: line_by_Px2

    line_by_Px2_4x2x1w(_l12, _l12, __Px2);

    // _l12  = .. | l1 | l2 | .. at Fp2 layer 

    // Step 7: mul_by_xy00z0_fp12_vec_v1

    // b01 =   line[1] |   line[0] |   line[1] |   line[0] at Fp2 layer
    // b4  =   line[2] |   line[2] |   line[2] |   line[2] at Fp2 layer

    // b01 =   line[1] |   line[1] |   line[1] |   line[1] at Fp2 layer
    perm_var(b01, _l12, m14);
    // b4  =   line[0] |   line[0] |   line[0] |   line[0] at Fp2 layer
    perm_var(b4, _l0, m13);
    // b01 =   line[1] |   line[0] |   line[1] |   line[0] at Fp2 layer
    blend_0x33(b01, b01, b4);
    // b4  =   line[2] |   line[2] |   line[2] |   line[2] at Fp2 layer
    perm_var(b4, _l12, m13);

    // a01 = ret[1][1] | ret[1][0] | ret[0][1] | ret[0][0] at Fp2 layer
    // a2  = ret[1][2] | ret[1][2] | ret[0][2] | ret[0][2] at Fp2 layer

    // a01 = ret[1][1] | ret[1][0] |       ... |       ... at Fp2 layer
    perm_var(a01, _r1, m2);
    // a01 = ret[1][1] | ret[1][0] | ret[0][1] | ret[0][0] at Fp2 layer
    blend_0x0F(a01, a01, _r0);
    // a2  = ret[1][2] | ret[1][2] | ret[0][2] | ret[0][2] at Fp2 layer
    perm_var(_r0, _r0, m10);
    perm_var(_r1, _r1, m10);
    blend_0x0F(a2, _r1, _r0);

    mul_by_xy00z0_fp12_vec_v1(r0, r1, a01, a2, b01, b4);

    // r0 = ret[1][1] | ret[1][0] | ret[0][1] | ret[0][2] at Fp2 layer
    // r1 =             ret[1][2] |             ret[0][0] at Fp2 layer
  }

  carryp_fp_4x2w(r1);
  conv_48to64_fp_8x1w(t[0], r0);
  conv_48to64_fp_4x2w(s[0], r1);

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
    ret[0][0][0][i         ] = ((uint64_t *)&s[0][i])[0];
    ret[0][0][0][i+SWORDS/2] = ((uint64_t *)&s[0][i])[1];
    ret[0][0][1][i         ] = ((uint64_t *)&s[0][i])[2];
    ret[0][0][1][i+SWORDS/2] = ((uint64_t *)&s[0][i])[3];
    ret[1][2][0][i         ] = ((uint64_t *)&s[0][i])[4];
    ret[1][2][0][i+SWORDS/2] = ((uint64_t *)&s[0][i])[5];
    ret[1][2][1][i         ] = ((uint64_t *)&s[0][i])[6];
    ret[1][2][1][i+SWORDS/2] = ((uint64_t *)&s[0][i])[7];
  }

    conv_48to64_fp_8x1w(t[0], _X3);
    conv_48to64_fp_8x1w(t[1], _Y3);
    conv_48to64_fp_8x1w(t[2], _Z3);

  for (i = 0; i < SWORDS; i++) {
    T   ->X[0][i] = ((uint64_t *)&t[0][i])[6];
    T   ->X[1][i] = ((uint64_t *)&t[0][i])[7];
    T   ->Y[0][i] = ((uint64_t *)&t[1][i])[6];
    T   ->Y[1][i] = ((uint64_t *)&t[1][i])[7];
    T   ->Z[0][i] = ((uint64_t *)&t[2][i])[0];
    T   ->Z[1][i] = ((uint64_t *)&t[2][i])[1];
  }
}

void miller_loop_n(vec384fp12 ret, const POINTonE2_affine Q[],
                                   const POINTonE1_affine P[], size_t n)
{
#if !defined(__STDC_VERSION__) || __STDC_VERSION__<199901 \
                               || defined(__STDC_NO_VLA__)
    POINTonE2 *T = alloca(n*sizeof(POINTonE2));
    POINTonE1_affine *Px2 = alloca(n*sizeof(POINTonE1_affine));
#else
    POINTonE2 T[n];
    POINTonE1_affine Px2[n];
#endif
    size_t i;

    if ((n == 1) && (vec_is_zero(&Q[0], sizeof(Q[0])) |
                     vec_is_zero(&P[0], sizeof(P[0]))) ) {
        /*
         * Special case of infinite aggregated signature, pair the additive
         * group's identity with the multiplicative group's identity.
         */
        vec_copy(ret, BLS12_381_Rx.p12, sizeof(vec384fp12));
        return;
    }

    for (i = 0; i < n; i++) {
        /* Move common expression from line evaluation to line_by_Px2.  */
        add_fp(Px2[i].X, P[i].X, P[i].X);
        neg_fp(Px2[i].X, Px2[i].X);
        add_fp(Px2[i].Y, P[i].Y, P[i].Y);

        vec_copy(T[i].X, Q[i].X, 2*sizeof(T[i].X));
        vec_copy(T[i].Z, BLS12_381_Rx.p2, sizeof(T[i].Z));
    }

    /* first step is ret = 1^2*line, which is replaced with ret = line  */
    start_dbl_n(ret, T, Px2, n);                /* 0x2                  */
    add_n_dbl_n(ret, T, Q, Px2, n, 2);          /* ..0xc                */
    add_n_dbl_n(ret, T, Q, Px2, n, 3);          /* ..0x68               */
    add_n_dbl_n(ret, T, Q, Px2, n, 9);          /* ..0xd200             */
    add_n_dbl_n(ret, T, Q, Px2, n, 32);         /* ..0xd20100000000     */
    add_n_dbl_n(ret, T, Q, Px2, n, 16);         /* ..0xd201000000010000 */
    conjugate_fp12(ret);                /* account for z being negative */
}

static void mul_n_sqr_scalar(vec384fp12 ret, const vec384fp12 a, size_t n)
{
    mul_fp12(ret, ret, a);
    while (n--)
        cyclotomic_sqr_fp12(ret, ret);
}

static void mul_n_sqr_vector(vec384fp12 ret, const vec384fp12 a, size_t n)
{
  fp2_4x2x1w ab0, ab1, ab2;
  fp2_2x2x2w r001, r101, r2;
  fp4_1x2x2x2w  _a;
  fp4_2x2x2x1w _bc;
  __m512i t[3][SWORDS], s[SWORDS/2];
  uint64_t r48[NWORDS];
  int i;

  // form < b1 | a1 | b0 | a0 >
  for (i = 0; i < SWORDS; i++) {
    t[0][i] = VSET( ret[1][0][1][i], ret[1][0][0][i],
                      a[1][0][1][i],   a[1][0][0][i],
                    ret[0][0][1][i], ret[0][0][0][i],
                      a[0][0][1][i],   a[0][0][0][i]);
    t[1][i] = VSET( ret[1][1][1][i], ret[1][1][0][i],
                      a[1][1][1][i],   a[1][1][0][i],
                    ret[0][1][1][i], ret[0][1][0][i],
                      a[0][1][1][i],   a[0][1][0][i]);
    t[2][i] = VSET( ret[1][2][1][i], ret[1][2][0][i],
                      a[1][2][1][i],   a[1][2][0][i],
                    ret[0][2][1][i], ret[0][2][0][i],
                      a[0][2][1][i],   a[0][2][0][i]);
  }

  conv_64to48_fp_8x1w(ab0, t[0]);
  conv_64to48_fp_8x1w(ab1, t[1]);
  conv_64to48_fp_8x1w(ab2, t[2]);

  mul_fp12_vec_v3(r001, r101, r2, ab0, ab1, ab2);

  carryp_fp_4x2w(r001);
  carryp_fp_4x2w(r101);
  carryp_fp_4x2w(r2);

  // form < a11 | a00 >
  blend_0x0F_hl(_a, r101, r001);

  // form < a12 | a01 | a02 | a10 >
  for (i = 0; i < VWORDS; i++) {
    _bc[i]        = VSET(((uint64_t *)&r2  [i])[6],
                         ((uint64_t *)&r2  [i])[4],
                         ((uint64_t *)&r001[i])[6],
                         ((uint64_t *)&r001[i])[4],
                         ((uint64_t *)&r2  [i])[2],
                         ((uint64_t *)&r2  [i])[0],
                         ((uint64_t *)&r101[i])[2],
                         ((uint64_t *)&r101[i])[0] );
    _bc[i+VWORDS] = VSET(((uint64_t *)&r2  [i])[7],
                         ((uint64_t *)&r2  [i])[5],
                         ((uint64_t *)&r001[i])[7],
                         ((uint64_t *)&r001[i])[5],
                         ((uint64_t *)&r2  [i])[3],
                         ((uint64_t *)&r2  [i])[1],
                         ((uint64_t *)&r101[i])[3],
                         ((uint64_t *)&r101[i])[1] );
  }

  while (n--)
    cyclotomic_sqr_fp12_vec_v1(_a, _bc, _a, _bc);

  carryp_fp_4x2w(_a);
  conv_48to64_fp_4x2w(s, _a);
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

  conv_48to64_fp_8x1w(t[0], _bc);
  for (i = 0; i < SWORDS; i++) {
    ret[1][0][0][i] = ((uint64_t *)&t[0][i])[0]; 
    ret[1][0][1][i] = ((uint64_t *)&t[0][i])[1];
    ret[0][2][0][i] = ((uint64_t *)&t[0][i])[2];
    ret[0][2][1][i] = ((uint64_t *)&t[0][i])[3];
    ret[0][1][0][i] = ((uint64_t *)&t[0][i])[4];
    ret[0][1][1][i] = ((uint64_t *)&t[0][i])[5];
    ret[1][2][0][i] = ((uint64_t *)&t[0][i])[6];
    ret[1][2][1][i] = ((uint64_t *)&t[0][i])[7];
  } 
}

#if COMPRESSED_CYCLOTOMIC_SQR
static void raise_to_z_div_by_2(vec384fp12 ret, const vec384fp12 a)
{
  fp4_2x2x2x1w _bc;
  __m512i t[SWORDS];
  vec384fp12 s[6];
  int i;

  for (i = 0; i < SWORDS; i++) {
    t[i] = VSET(a[1][2][1][i], a[1][2][0][i], 
                a[0][1][1][i], a[0][1][0][i], 
                a[0][2][1][i], a[0][2][0][i], 
                a[1][0][1][i], a[1][0][0][i] );
  }

  conv_64to48_fp_8x1w(_bc, t);

  for (i = 0; i < 15; i++) compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[0][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[0][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[0][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[0][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[0][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[0][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[0][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[0][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }

  for (i = 0; i < 32; i++) compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[1][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[1][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[1][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[1][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[1][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[1][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[1][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[1][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }

  for (i = 0; i < 9; i++) compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[2][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[2][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[2][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[2][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[2][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[2][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[2][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[2][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }

  for (i = 0; i < 3; i++) compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[3][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[3][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[3][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[3][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[3][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[3][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[3][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[3][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }

  for (i = 0; i < 2; i++) compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[4][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[4][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[4][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[4][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[4][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[4][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[4][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[4][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }

  compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[5][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[5][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[5][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[5][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[5][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[5][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[5][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[5][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }


  back_cyclotomic_sim_fp12(s, s, 6);


  fp2_4x2x1w ab0[4], ab1[4], ab2[4], _ab3;
  fp2_2x2x2w r001[4], r101[4], r2[4];
  __m512i _t[3][3][SWORDS];

  for (i = 0; i < SWORDS; i++) {
    t[0][0][i] = VSET( s[1][1][0][1][i], s[1][1][0][0][i],
                       s[0][1][0][1][i], s[0][1][0][0][i],
                       s[1][0][0][1][i], s[1][0][0][0][i],
                       s[0][0][0][1][i], s[0][0][0][0][i]);
    t[0][1][i] = VSET( s[1][1][1][1][i], s[1][1][1][0][i],
                       s[0][1][1][1][i], s[0][1][1][0][i],
                       s[1][0][1][1][i], s[1][0][1][0][i],
                       s[0][0][1][1][i], s[0][0][1][0][i]);
    t[0][2][i] = VSET( s[1][1][2][1][i], s[1][1][2][0][i],
                       s[0][1][2][1][i], s[0][1][2][0][i],
                       s[1][0][2][1][i], s[1][0][2][0][i],
                       s[0][0][2][1][i], s[0][0][2][0][i]);
  }

  conv_64to48_fp_8x1w(ab0[0], t[0][0]);
  conv_64to48_fp_8x1w(ab1[0], t[0][1]);
  conv_64to48_fp_8x1w(ab2[0], t[0][2]);

  for (i = 0; i < SWORDS; i++) {
    t[1][0][i] = VSET( s[3][1][0][1][i], s[3][1][0][0][i],
                       s[2][1][0][1][i], s[2][1][0][0][i],
                       s[3][0][0][1][i], s[3][0][0][0][i],
                       s[2][0][0][1][i], s[2][0][0][0][i]);
    t[1][1][i] = VSET( s[3][1][1][1][i], s[3][1][1][0][i],
                       s[2][1][1][1][i], s[2][1][1][0][i],
                       s[3][0][1][1][i], s[3][0][1][0][i],
                       s[2][0][1][1][i], s[2][0][1][0][i]);
    t[1][2][i] = VSET( s[3][1][2][1][i], s[3][1][2][0][i],
                       s[2][1][2][1][i], s[2][1][2][0][i],
                       s[3][0][2][1][i], s[3][0][2][0][i],
                       s[2][0][2][1][i], s[2][0][2][0][i]);
  }

  conv_64to48_fp_8x1w(ab0[1], t[1][0]);
  conv_64to48_fp_8x1w(ab1[1], t[1][1]);
  conv_64to48_fp_8x1w(ab2[1], t[1][2]);

  for (i = 0; i < SWORDS; i++) {
    t[2][0][i] = VSET( s[5][1][0][1][i], s[5][1][0][0][i],
                       s[4][1][0][1][i], s[4][1][0][0][i],
                       s[5][0][0][1][i], s[5][0][0][0][i],
                       s[4][0][0][1][i], s[4][0][0][0][i]);
    t[2][1][i] = VSET( s[5][1][1][1][i], s[5][1][1][0][i],
                       s[4][1][1][1][i], s[4][1][1][0][i],
                       s[5][0][1][1][i], s[5][0][1][0][i],
                       s[4][0][1][1][i], s[4][0][1][0][i]);
    t[2][2][i] = VSET( s[5][1][2][1][i], s[5][1][2][0][i],
                       s[4][1][2][1][i], s[4][1][2][0][i],
                       s[5][0][2][1][i], s[5][0][2][0][i],
                       s[4][0][2][1][i], s[4][0][2][0][i]);
  }

  conv_64to48_fp_8x1w(ab0[2], t[2][0]);
  conv_64to48_fp_8x1w(ab1[2], t[2][1]);
  conv_64to48_fp_8x1w(ab2[2], t[2][2]);

  mul_fp12_vec_v3(r001[0], r101[0], r2[0], ab0[0], ab1[0], ab2[0]);
  mul_fp12_vec_v3(r001[1], r101[1], r2[1], ab0[1], ab1[1], ab2[1]);
  mul_fp12_vec_v3(r001[2], r101[2], r2[2], ab0[2], ab1[2], ab2[2]);

  fp2_2x2x2w _t0, _t1;
  const __m512i m0 = VSET(3, 2, 1, 0, 7, 6, 5, 4); 
  const __m512i m1 = VSET(6, 4, 6, 4, 2, 0, 2, 0);
  const __m512i m2 = VSET(7, 5, 7, 5, 3, 1, 3, 1);
  const __m512i m3 = VSET(2, 0, 2, 0, 6, 4, 6, 4);
  const __m512i m4 = VSET(3, 1, 3, 1, 7, 5, 7, 5);

  // r101[0] =  r1[0] |  r1[1] 
  perm_var_hl(r101[0], r101[0], m0);
  // r101[1] = r1'[0] | r1'[1]
  perm_var_hl(r101[1], r101[1], m0);
  //    _t0 =   r1[0] |  r0[0]
  blend_0x0F_hl(_t0, r101[0], r001[0]);
  //    _t1 =  r1'[0] | r0'[0]
  blend_0x0F_hl(_t1, r101[1], r001[1]);
  //   ab0[3] =   r1[0] |  r1[0] |  r0[0] |  r0[0]
  //   ab1[3] =  r1'[0] | r1'[0] | r0'[0] | r0'[0] 
  for (i = 0; i < VWORDS; i++) {
    ab0[3][i]        = VPERMV(m1, _t0[i]);
    ab0[3][i+VWORDS] = VPERMV(m2, _t0[i]);
    ab1[3][i]        = VPERMV(m1, _t1[i]);
    ab1[3][i+VWORDS] = VPERMV(m2, _t1[i]);
  }
  // ab0[3] =  r1'[0] | r1[0] | r0'[0] | r0'[0] 
  blend_0x33(ab0[3], ab1[3], ab0[3]);

  //  _t0 =   r0[1] |  r1[1]
  blend_0x0F_hl(_t0, r001[0], r101[0]);
  //  _t1 =  r0'[1] | r1'[1]
  blend_0x0F_hl(_t1, r001[1], r101[1]);
  // ab1[3] =   r1[1] |  r1[1] |  r0[1] |  r0[1]
  // ab2[3] =  r1'[1] |  r1'[1]| r0'[1] | r0'[1] 
  for (i = 0; i < VWORDS; i++) {
    ab1[3][i]        = VPERMV(m3, _t0[i]);
    ab1[3][i+VWORDS] = VPERMV(m4, _t0[i]);
    ab2[3][i]        = VPERMV(m3, _t1[i]);
    ab2[3][i+VWORDS] = VPERMV(m4, _t1[i]);
  }
  // ab1[3] =  r1'[1] |  r1[1] | r0'[1] | r0'[1] 
  blend_0x33(ab1[3], ab2[3], ab1[3]);

  // ab2[3] =   r1[2] |  r1[2] |  r0[2] |  r0[2]
  // _ab3 =  r1'[2] | r1'[2] | r0'[2] | r0'[2] 
  for (i = 0; i < VWORDS; i++) {
    ab2[3][i]        = VPERMV(m1, r2[0][i]);
    ab2[3][i+VWORDS] = VPERMV(m2, r2[0][i]);
    _ab3[i]          = VPERMV(m1, r2[1][i]);
    _ab3[i+VWORDS]   = VPERMV(m2, r2[1][i]);
  }
  // ab2[3] =  r1'[2] |  r1[2] | r0'[2] | r0'[2] 
  blend_0x33(ab2[3], _ab3, ab2[3]);

  mul_fp12_vec_v3(r001[3], r101[3], r2[3], ab0[3], ab1[3], ab2[3]);

  perm_var_hl(r101[2], r101[2], m0);
  perm_var_hl(r101[3], r101[3], m0);
  blend_0x0F_hl(_t0, r101[2], r001[2]);
  blend_0x0F_hl(_t1, r101[3], r001[3]);
  for (i = 0; i < VWORDS; i++) {
    ab0[3][i]        = VPERMV(m1, _t0[i]);
    ab0[3][i+VWORDS] = VPERMV(m2, _t0[i]);
    ab1[3][i]        = VPERMV(m1, _t1[i]);
    ab1[3][i+VWORDS] = VPERMV(m2, _t1[i]);
  }
  blend_0x33(ab0[3], ab1[3], ab0[3]);

  blend_0x0F_hl(_t0, r001[2], r101[2]);
  blend_0x0F_hl(_t1, r001[3], r101[3]);
  for (i = 0; i < VWORDS; i++) {
    ab1[3][i]        = VPERMV(m3, _t0[i]);
    ab1[3][i+VWORDS] = VPERMV(m4, _t0[i]);
    ab2[3][i]        = VPERMV(m3, _t1[i]);
    ab2[3][i+VWORDS] = VPERMV(m4, _t1[i]);
  }
  blend_0x33(ab1[3], ab2[3], ab1[3]);

  for (i = 0; i < VWORDS; i++) {
    ab2[3][i]        = VPERMV(m1, r2[2][i]);
    ab2[3][i+VWORDS] = VPERMV(m2, r2[2][i]);
    _ab3[i]          = VPERMV(m1, r2[3][i]);
    _ab3[i+VWORDS]   = VPERMV(m2, r2[3][i]);
  }
  blend_0x33(ab2[3], _ab3, ab2[3]);

  mul_fp12_vec_v3(r001[3], r101[3], r2[3], ab0[3], ab1[3], ab2[3]);

  carryp_fp_4x2w(r001[3]);
  carryp_fp_4x2w(r101[3]);
  carryp_fp_4x2w(r2[3]);

  conv_48to64_fp_4x2w(t[0][0], r001[3]);
  conv_48to64_fp_4x2w(t[0][1], r101[3]);
  conv_48to64_fp_4x2w(t[0][2], r2[3]);

  for (i = 0; i < SWORDS/2; i++) {
    ret[0][0][0][i         ] = ((uint64_t *)&t[0][0][i])[0];
    ret[0][0][0][i+SWORDS/2] = ((uint64_t *)&t[0][0][i])[1];
    ret[0][0][1][i         ] = ((uint64_t *)&t[0][0][i])[2];
    ret[0][0][1][i+SWORDS/2] = ((uint64_t *)&t[0][0][i])[3];
    ret[0][1][0][i         ] = ((uint64_t *)&t[0][0][i])[4];
    ret[0][1][0][i+SWORDS/2] = ((uint64_t *)&t[0][0][i])[5];
    ret[0][1][1][i         ] = ((uint64_t *)&t[0][0][i])[6];
    ret[0][1][1][i+SWORDS/2] = ((uint64_t *)&t[0][0][i])[7];
    ret[1][0][0][i         ] = ((uint64_t *)&t[0][1][i])[0];
    ret[1][0][0][i+SWORDS/2] = ((uint64_t *)&t[0][1][i])[1];
    ret[1][0][1][i         ] = ((uint64_t *)&t[0][1][i])[2];
    ret[1][0][1][i+SWORDS/2] = ((uint64_t *)&t[0][1][i])[3];
    ret[1][1][0][i         ] = ((uint64_t *)&t[0][1][i])[4];
    ret[1][1][0][i+SWORDS/2] = ((uint64_t *)&t[0][1][i])[5];
    ret[1][1][1][i         ] = ((uint64_t *)&t[0][1][i])[6];
    ret[1][1][1][i+SWORDS/2] = ((uint64_t *)&t[0][1][i])[7];
    ret[0][2][0][i         ] = ((uint64_t *)&t[0][2][i])[0];
    ret[0][2][0][i+SWORDS/2] = ((uint64_t *)&t[0][2][i])[1];
    ret[0][2][1][i         ] = ((uint64_t *)&t[0][2][i])[2];
    ret[0][2][1][i+SWORDS/2] = ((uint64_t *)&t[0][2][i])[3];
    ret[1][2][0][i         ] = ((uint64_t *)&t[0][2][i])[4];
    ret[1][2][0][i+SWORDS/2] = ((uint64_t *)&t[0][2][i])[5];
    ret[1][2][1][i         ] = ((uint64_t *)&t[0][2][i])[6];
    ret[1][2][1][i+SWORDS/2] = ((uint64_t *)&t[0][2][i])[7];
  }


  // mul_fp12_vector(s[0], s[0], s[1]);
  // mul_fp12_vector(s[1], s[2], s[3]);
  // mul_fp12_vector(s[2], s[4], s[5]);
  // mul_fp12_vector(ret,  s[0], s[1]);
  // mul_fp12_vector(ret,  ret,  s[2]);

  conjugate_fp12(ret);                /* account for z being negative */
}
#else 
static void raise_to_z_div_by_2(vec384fp12 ret, const vec384fp12 a)
{
  cyclotomic_sqr_fp12(ret, a);                /* 0x2                  */
  mul_n_sqr(ret, a, 2);                       /* ..0xc                */
  mul_n_sqr(ret, a, 3);                       /* ..0x68               */
  mul_n_sqr(ret, a, 9);                       /* ..0xd200             */
  mul_n_sqr(ret, a, 32);                      /* ..0xd20100000000     */
  mul_n_sqr(ret, a, 16-1);                    /* ..0x6900800000008000 */
  conjugate_fp12(ret);                /* account for z being negative */
}
#endif

#if COMPRESSED_CYCLOTOMIC_SQR
static void raise_to_z(vec384fp12 ret, const vec384fp12 a)
{
  fp4_2x2x2x1w _bc;
  __m512i t[SWORDS];
  vec384fp12 s[6];
  int i;

  for (i = 0; i < SWORDS; i++) {
    t[i] = VSET(a[1][2][1][i], a[1][2][0][i], 
                a[0][1][1][i], a[0][1][0][i], 
                a[0][2][1][i], a[0][2][0][i], 
                a[1][0][1][i], a[1][0][0][i] );
  }

  conv_64to48_fp_8x1w(_bc, t);

  for (i = 0; i < 16; i++) 
    compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[0][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[0][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[0][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[0][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[0][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[0][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[0][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[0][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }
  for (i = 0; i < 32; i++) 
    compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[1][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[1][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[1][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[1][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[1][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[1][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[1][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[1][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }
  for (i = 0; i < 9; i++) 
    compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[2][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[2][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[2][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[2][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[2][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[2][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[2][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[2][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }
  for (i = 0; i < 3; i++) 
    compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[3][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[3][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[3][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[3][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[3][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[3][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[3][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[3][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }
  for (i = 0; i < 2; i++) 
    compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[4][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[4][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[4][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[4][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[4][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[4][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[4][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[4][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }
    compressed_cyclotomic_sqr_fp12_vec_v1(_bc, _bc);
  conv_48to64_fp_8x1w(t, _bc);
  for (i = 0; i < SWORDS; i++) {
    s[5][1][0][0][i] = ((uint64_t *)&t[i])[0];
    s[5][1][0][1][i] = ((uint64_t *)&t[i])[1];
    s[5][0][2][0][i] = ((uint64_t *)&t[i])[2];
    s[5][0][2][1][i] = ((uint64_t *)&t[i])[3];
    s[5][0][1][0][i] = ((uint64_t *)&t[i])[4];
    s[5][0][1][1][i] = ((uint64_t *)&t[i])[5];
    s[5][1][2][0][i] = ((uint64_t *)&t[i])[6];
    s[5][1][2][1][i] = ((uint64_t *)&t[i])[7];
  }

  back_cyclotomic_sim_fp12(s, s, 6);

  mul_fp12_vector(ret, s[0], s[1]);
  mul_fp12_vector(ret, ret,  s[2]);
  mul_fp12_vector(ret, ret,  s[3]);
  mul_fp12_vector(ret, ret,  s[4]);
  mul_fp12_vector(ret, ret,  s[5]);

  conjugate_fp12(ret);                /* account for z being negative */
}
#else
#define raise_to_z(a, b) (raise_to_z_div_by_2(a, b), cyclotomic_sqr_fp12(a, a)) 
#endif 

/*
 * Adaptation from <zkcrypto>/pairing/src/bls12_381/mod.rs
 */
void final_exp(vec384fp12 ret, const vec384fp12 f)
{
    vec384fp12 y0, y1, y2, y3;

    vec_copy(y1, f, sizeof(y1));
    conjugate_fp12(y1);
    inverse_fp12(y2, f);
    mul_fp12(ret, y1, y2);
    frobenius_map_fp12(y2, ret, 2);
    mul_fp12(ret, ret, y2);

    cyclotomic_sqr_fp12(y0, ret);
    raise_to_z(y1, y0);
    raise_to_z_div_by_2(y2, y1);
    vec_copy(y3, ret, sizeof(y3));
    conjugate_fp12(y3);
    mul_fp12(y1, y1, y3);
    conjugate_fp12(y1);
    mul_fp12(y1, y1, y2);
    raise_to_z(y2, y1);
    raise_to_z(y3, y2);
    conjugate_fp12(y1);
    mul_fp12(y3, y3, y1);
    conjugate_fp12(y1);
    frobenius_map_fp12(y1, y1, 3);
    frobenius_map_fp12(y2, y2, 2);
    mul_fp12(y1, y1, y2);
    raise_to_z(y2, y3);
    mul_fp12(y2, y2, y0);
    mul_fp12(y2, y2, ret);
    mul_fp12(y1, y1, y2);
    frobenius_map_fp12(y2, y3, 1);
    mul_fp12(ret, y1, y2);
}

// optimal ate pairing 

void optimal_ate_pairing(vec384fp12 ret, POINTonE2_affine Q[], 
                                         POINTonE1_affine P[], size_t n)
{
  vec384fp12 f;

  miller_loop_n(f, Q, P, n);
  final_exp(ret, f);
}
