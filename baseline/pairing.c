/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pairing.h"

#ifdef PROFILING
  extern uint64_t read_tsc();
  extern uint64_t line_add_cycles;
  extern uint64_t line_dbl_cycles;
  extern uint64_t line_by_Px2_cycles;
#endif 

/*
 * Line evaluations from  https://eprint.iacr.org/2010/354.pdf
 * with a twist moving common expression to line_by_Px2.
 */
static void line_add(vec384fp6 line, POINTonE2 *T, const POINTonE2 *R,
                                                   const POINTonE2_affine *Q)
{

#ifdef PROFILING
  uint64_t start_cycles = read_tsc();
#endif

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

#ifdef PROFILING
  uint64_t end_cycles = read_tsc();
  line_add_cycles += end_cycles - start_cycles;
#endif  
}

static void line_dbl(vec384fp6 line, POINTonE2 *T, const POINTonE2 *Q)
{
#ifdef PROFILING
  uint64_t start_cycles = read_tsc();
#endif

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

#ifdef PROFILING
  uint64_t end_cycles = read_tsc();
  line_dbl_cycles += end_cycles - start_cycles;
#endif  
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

static void add_n_dbl_n(vec384fp12 ret, POINTonE2 T[],
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

static void mul_n_sqr(vec384fp12 ret, const vec384fp12 a, size_t n)
{
    mul_fp12(ret, ret, a);
    while (n--)
        cyclotomic_sqr_fp12(ret, ret);
}

static void raise_to_z_div_by_2(vec384fp12 ret, const vec384fp12 a)
{
#if COMPRESSED_CYCLOTOMIC_SQR
    vec384fp12 s[6];
    int i;

    vec_copy(ret, a, sizeof(vec384fp12));
    for (i = 0; i < 15; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[0], ret, sizeof(vec384fp12));
    for (i = 0; i < 32; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[1], ret, sizeof(vec384fp12));
    for (i = 0; i < 9; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[2], ret, sizeof(vec384fp12));
    for (i = 0; i < 3; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[3], ret, sizeof(vec384fp12));
    for (i = 0; i < 2; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[4], ret, sizeof(vec384fp12));
    compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[5], ret, sizeof(vec384fp12));

    back_cyclotomic_sim_fp12(s, s, 6);

    mul_fp12(ret, s[0], s[1]);
    mul_fp12(ret, ret,  s[2]);
    mul_fp12(ret, ret,  s[3]);
    mul_fp12(ret, ret,  s[4]);
    mul_fp12(ret, ret,  s[5]);

    conjugate_fp12(ret);                /* account for z being negative */
#else 
    cyclotomic_sqr_fp12(ret, a);                /* 0x2                  */
    mul_n_sqr(ret, a, 2);                       /* ..0xc                */
    mul_n_sqr(ret, a, 3);                       /* ..0x68               */
    mul_n_sqr(ret, a, 9);                       /* ..0xd200             */
    mul_n_sqr(ret, a, 32);                      /* ..0xd20100000000     */
    mul_n_sqr(ret, a, 16-1);                    /* ..0x6900800000008000 */
    conjugate_fp12(ret);                /* account for z being negative */
#endif 
}


#if COMPRESSED_CYCLOTOMIC_SQR
static void raise_to_z(vec384fp12 ret, const vec384fp12 a)
{
    vec384fp12 s[6];
    int i;

    vec_copy(ret, a, sizeof(vec384fp12));
    for (i = 0; i < 16; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[0], ret, sizeof(vec384fp12));
    for (i = 0; i < 32; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[1], ret, sizeof(vec384fp12));
    for (i = 0; i < 9; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[2], ret, sizeof(vec384fp12));
    for (i = 0; i < 3; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[3], ret, sizeof(vec384fp12));
    for (i = 0; i < 2; i++) compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[4], ret, sizeof(vec384fp12));
    compressed_cyclotomic_sqr_fp12(ret, ret);
    vec_copy(s[5], ret, sizeof(vec384fp12));

    back_cyclotomic_sim_fp12(s, s, 6);

    mul_fp12(ret, s[0], s[1]);
    mul_fp12(ret, ret,  s[2]);
    mul_fp12(ret, ret,  s[3]);
    mul_fp12(ret, ret,  s[4]);
    mul_fp12(ret, ret,  s[5]);

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
