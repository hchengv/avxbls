/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __BLS12_381_ASM_POINT_H__
#define __BLS12_381_ASM_POINT_H__

#include "vect.h"
#include "fields.h"
#include "ec_ops.h"

#define DECLARE_POINT(ptype, bits) \
typedef struct { vec##bits X,Y,Z; } ptype; \
typedef struct { vec##bits X,Y; } ptype##_affine; \
\
static void ptype##_dadd(ptype *out, const ptype *p1, const ptype *p2,	\
                         const vec##bits a4);				\
static void ptype##_dadd_affine(ptype *out, const ptype *p1,		\
                                            const ptype##_affine *p2);	\
static void ptype##_add(ptype *out, const ptype *p1, const ptype *p2);	\
static void ptype##_add_affine(ptype *out, const ptype *p1,		\
                                           const ptype##_affine *p2);	\
static void ptype##_double(ptype *out, const ptype *p1);		\
static void ptype##_mult_w5(ptype *out, const ptype *point,		\
                            const byte *scalar, size_t nbits);		\
static void ptype##_cneg(ptype *p, limb_t cbit);			\
void ptype##_to_affine(ptype##_affine *out, const ptype *in);	\
static void ptype##_from_Jacobian(ptype *out, const ptype *in);		\
\
static inline void ptype##_cswap(ptype *restrict a,			\
                                 ptype *restrict b, bool_t cbit) {	\
    vec_cswap(a, b, sizeof(ptype), cbit);				\
} \
static inline void ptype##_ccopy(ptype *restrict a,			\
                                 const ptype *restrict b, bool_t cbit) {\
    vec_select(a, b, a, sizeof(ptype), cbit);				\
}

DECLARE_POINT(POINTonE1, 384)
DECLARE_POINT(POINTonE2, 384x)

extern const POINTonE1 BLS12_381_G1;
extern const POINTonE2 BLS12_381_G2;

POINT_DOUBLE_IMPL_A0(POINTonE1, 384, fp)
POINT_DOUBLE_IMPL_A0(POINTonE2, 384x, fp2)
POINT_ADD_IMPL(POINTonE1, 384, fp)
POINT_ADD_IMPL(POINTonE2, 384x, fp2)

#ifdef __GNUC__
# pragma GCC diagnostic ignored "-Wunused-function"
#endif

#endif
