/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __BLS12_381_ASM_FIELDS_H__
#define __BLS12_381_ASM_FIELDS_H__

#include "vect.h"
#include "consts.h"

/*
 * BLS12-381-specific Fp shortcuts to assembly.
 */
static inline void add_fp(vec384 ret, const vec384 a, const vec384 b)
{   add_mod_384(ret, a, b, BLS12_381_P);   }

static inline void sub_fp(vec384 ret, const vec384 a, const vec384 b)
{   sub_mod_384(ret, a, b, BLS12_381_P);   }

static inline void mul_by_3_fp(vec384 ret, const vec384 a)
{   mul_by_3_mod_384(ret, a, BLS12_381_P);   }

static inline void mul_by_8_fp(vec384 ret, const vec384 a)
{   mul_by_8_mod_384(ret, a, BLS12_381_P);   }

static inline void mul_fp(vec384 ret, const vec384 a, const vec384 b)
{   mul_mont_384(ret, a, b, BLS12_381_P, p0);   }

static inline void sqr_fp(vec384 ret, const vec384 a)
{   sqr_mont_384(ret, a, BLS12_381_P, p0);   }

static inline void cneg_fp(vec384 ret, const vec384 a, bool_t flag)
{   cneg_mod_384(ret, a, flag, BLS12_381_P);   }

static inline void redc_fp(vec384 ret, const vec768 a)
{   redc_mont_384(ret, a, BLS12_381_P, p0);   }

/*
 * BLS12-381-specific Fp2 shortcuts to assembly.
 */
static inline void add_fp2(vec384x ret, const vec384x a, const vec384x b)
{   add_mod_384x(ret, a, b, BLS12_381_P);   }

static inline void sub_fp2(vec384x ret, const vec384x a, const vec384x b)
{   sub_mod_384x(ret, a, b, BLS12_381_P);   }

static inline void mul_by_3_fp2(vec384x ret, const vec384x a)
{   mul_by_3_mod_384x(ret, a, BLS12_381_P);   }

static inline void mul_by_8_fp2(vec384x ret, const vec384x a)
{   mul_by_8_mod_384x(ret, a, BLS12_381_P);   }

static inline void lshift_fp2(vec384x ret, const vec384x a, size_t count)
{
    lshift_mod_384(ret[0], a[0], count, BLS12_381_P);
    lshift_mod_384(ret[1], a[1], count, BLS12_381_P);
}

static inline void mul_fp2(vec384x ret, const vec384x a, const vec384x b)
{   mul_mont_384x(ret, a, b, BLS12_381_P, p0);   }

static inline void sqr_fp2(vec384x ret, const vec384x a)
{   sqr_mont_384x(ret, a, BLS12_381_P, p0);   }

static inline void cneg_fp2(vec384x ret, const vec384x a, bool_t flag)
{
    cneg_mod_384(ret[0], a[0], flag, BLS12_381_P);
    cneg_mod_384(ret[1], a[1], flag, BLS12_381_P);
}

void reciprocal_fp(vec384 out, const vec384 inp);
void reciprocal_fp2(vec384x out, const vec384x inp);

typedef vec384x   vec384fp2;
typedef vec384fp2 vec384fp6[3];
typedef vec384fp6 vec384fp12[2];

#define sqr_fp12 sqr_fp12_vector
void sqr_fp12_scalar(vec384fp12 ret, const vec384fp12 a);
void sqr_fp12_vector(vec384fp12 ret, const vec384fp12 a);

#define cyclotomic_sqr_fp12 cyclotomic_sqr_fp12_vector
void cyclotomic_sqr_fp12_scalar(vec384fp12 ret, const vec384fp12 a);
void cyclotomic_sqr_fp12_vector(vec384fp12 ret, const vec384fp12 a);

#define mul_fp12 mul_fp12_vector
void mul_fp12_scalar(vec384fp12 ret, const vec384fp12 a, const vec384fp12 b);
void mul_fp12_vector(vec384fp12 ret, const vec384fp12 a, const vec384fp12 b);

#define mul_by_xy00z0_fp12 mul_by_xy00z0_fp12_vector
void mul_by_xy00z0_fp12_scalar(vec384fp12 ret, const vec384fp12 a,
                                               const vec384fp6 xy00z0);
void mul_by_xy00z0_fp12_vector(vec384fp12 ret, const vec384fp12 a,
                                               const vec384fp6 xy00z0);


void conjugate_fp12(vec384fp12 a);
void inverse_fp12(vec384fp12 ret, const vec384fp12 a);
/* caveat lector! |n| has to be non-zero and not more than 3! */
void frobenius_map_fp12(vec384fp12 ret, const vec384fp12 a, size_t n);

#define neg_fp(r,a) cneg_fp((r),(a),1)
#define neg_fp2(r,a) cneg_fp2((r),(a),1)

typedef vec768 vec768x[2];
typedef vec768x vec768fp6[3];
typedef vec384x vec384fp4[2];

// for timing measurement
#ifdef BENCHMARK
void mul_fp2x2(vec768x ret, const vec384x a, const vec384x b);
void sqr_fp2x2(vec768x ret, const vec384x a);
void sqr_fp4(vec384fp4 ret, const vec384x a0, const vec384x a1);
void mul_fp6x2(vec768fp6 ret, const vec384fp6 a, const vec384fp6 b);
#endif

void compressed_cyclotomic_sqr_fp12_scalar(vec384fp12 ret, const vec384fp12 a);
void back_cyclotomic_sim_fp12(vec384fp12 ret[], const vec384fp12 a[], const int n);

#endif /* __BLS12_381_ASM_FIELDS_H__ */
