/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __BLS12_381_ASM_VECT_H__
#define __BLS12_381_ASM_VECT_H__

#include <stddef.h>
#include <stdint.h>

typedef uint64_t limb_t;

#define LIMB_T_BITS   64

#define TO_LIMB_T(limb64)     limb64

#define NLIMBS(bits)   (bits/LIMB_T_BITS)

typedef limb_t vec384[NLIMBS(384)];
typedef limb_t vec768[NLIMBS(768)];
typedef vec384 vec384x[2];      /* 0 is "real" part, 1 is "imaginary" */

typedef unsigned char byte;

/*
 * Internal Boolean type, Boolean by value, hence safe to cast to or
 * reinterpret as 'bool'.
 */
typedef limb_t bool_t;

/*
 * Assembly subroutines...
 */
# define sqr_n_mul_mont_383 sqrx_n_mul_mont_383
# define mul_mont_384 mulx_mont_384
# define sqr_mont_384 sqrx_mont_384
# define mul_384 mulx_384
# define redc_mont_384 redcx_mont_384
# define ct_inverse_mod_383 ctx_inverse_mod_383

void sqr_n_mul_mont_383(vec384 ret, const vec384 a, size_t count,
                        const vec384 p, limb_t n0, const vec384 b);

void mul_mont_384(vec384 ret, const vec384 a, const vec384 b,
                  const vec384 p, limb_t n0);
void sqr_mont_384(vec384 ret, const vec384 a, const vec384 p, limb_t n0);

void mul_384(vec768 ret, const vec384 a, const vec384 b);
void redc_mont_384(vec384 ret, const vec768 a, const vec384 p, limb_t n0);

void add_mod_384(vec384 ret, const vec384 a, const vec384 b, const vec384 p);
void sub_mod_384(vec384 ret, const vec384 a, const vec384 b, const vec384 p);
void mul_by_8_mod_384(vec384 ret, const vec384 a, const vec384 p);
void mul_by_3_mod_384(vec384 ret, const vec384 a, const vec384 p);
void cneg_mod_384(vec384 ret, const vec384 a, bool_t flag, const vec384 p);
void lshift_mod_384(vec384 ret, const vec384 a, size_t count, const vec384 p);
void ct_inverse_mod_383(vec768 ret, const vec384 inp, const vec384 mod,
                                                      const vec384 modx);

# define mul_mont_384x mulx_mont_384x
# define sqr_mont_384x sqrx_mont_384x
# define mul_382x mulx_382x
# define sqr_382x sqrx_382x

void mul_mont_384x(vec384x ret, const vec384x a, const vec384x b,
                   const vec384 p, limb_t n0);
void sqr_mont_384x(vec384x ret, const vec384x a, const vec384 p, limb_t n0);
void mul_382x(vec768 ret[2], const vec384x a, const vec384x b, const vec384 p);
void sqr_382x(vec768 ret[2], const vec384x a, const vec384 p);

void add_mod_384x(vec384x ret, const vec384x a, const vec384x b,
                  const vec384 p);
void sub_mod_384x(vec384x ret, const vec384x a, const vec384x b,
                  const vec384 p);
void mul_by_8_mod_384x(vec384x ret, const vec384x a, const vec384 p);
void mul_by_3_mod_384x(vec384x ret, const vec384x a, const vec384 p);
void mul_by_1_plus_i_mod_384x(vec384x ret, const vec384x a, const vec384 p);
void add_mod_384x384(vec768 ret, const vec768 a, const vec768 b,
                     const vec384 p);
void sub_mod_384x384(vec768 ret, const vec768 a, const vec768 b,
                     const vec384 p);

/*
 * C subroutines
 */
#if !defined(restrict)
# if !defined(__STDC_VERSION__) || __STDC_VERSION__<199901
#  if defined(__GNUC__) && __GNUC__>=2
#   define restrict __restrict__
#  elif defined(_MSC_VER)
#   define restrict __restrict
#  else
#   define restrict
#  endif
# endif
#endif

#if !defined(inline) && !defined(__cplusplus)
# if !defined(__STDC_VERSION__) || __STDC_VERSION__<199901
#  if defined(__GNUC__) && __GNUC__>=2
#   define inline __inline__
#  elif defined(_MSC_VER)
#   define inline __inline
#  else
#   define inline
#  endif
# endif
#endif

#if defined(__GNUC__) || defined(__clang__)
# define launder(var) __asm__ __volatile__("" : "+r"(var))
#else
# define launder(var)
#endif

static inline void vec_cswap(void *restrict a, void *restrict b, size_t num,
                             bool_t cbit)
{
    limb_t ai, *ap = (limb_t *)a;
    limb_t bi, *bp = (limb_t *)b;
    limb_t xorm, mask;
    size_t i;

    launder(cbit);
    mask = (limb_t)0 - cbit;

    num /= sizeof(limb_t);

    for (i = 0; i < num; i++) {
        xorm = ((ai = ap[i]) ^ (bi = bp[i])) & mask;
        ap[i] = ai ^ xorm;
        bp[i] = bi ^ xorm;
    }
}

/* ret = bit ? a : b */
void vec_select_32(void *ret, const void *a, const void *b, bool_t sel_a);
void vec_select_48(void *ret, const void *a, const void *b, bool_t sel_a);
void vec_select_96(void *ret, const void *a, const void *b, bool_t sel_a);
void vec_select_144(void *ret, const void *a, const void *b, bool_t sel_a);
void vec_select_192(void *ret, const void *a, const void *b, bool_t sel_a);
void vec_select_288(void *ret, const void *a, const void *b, bool_t sel_a);
static inline void vec_select(void *ret, const void *a, const void *b,
                              size_t num, bool_t sel_a)
{
    launder(sel_a);
#ifndef __BLST_NO_ASM__
    if (num == 32)          vec_select_32(ret, a, b, sel_a);
    else if (num == 48)     vec_select_48(ret, a, b, sel_a);
    else if (num == 96)     vec_select_96(ret, a, b, sel_a);
    else if (num == 144)    vec_select_144(ret, a, b, sel_a);
    else if (num == 192)    vec_select_192(ret, a, b, sel_a);
    else if (num == 288)    vec_select_288(ret, a, b, sel_a);
#else
    if (0) ;
#endif
    else {
        limb_t bi;
        volatile limb_t *rp = (limb_t *)ret;
        const limb_t *ap = (const limb_t *)a;
        const limb_t *bp = (const limb_t *)b;
        limb_t xorm, mask = (limb_t)0 - sel_a;
        size_t i;

        num /= sizeof(limb_t);

        for (i = 0; i < num; i++) {
            xorm = (ap[i] ^ (bi = bp[i])) & mask;
            rp[i] = bi ^ xorm;
        }
    }
}

static inline bool_t is_zero(limb_t l)
{
    limb_t ret = (~l & (l - 1)) >> (LIMB_T_BITS - 1);
    launder(ret);
    return ret;
}

static inline bool_t vec_is_zero(const void *a, size_t num)
{
    const limb_t *ap = (const limb_t *)a;
    limb_t acc;
    size_t i;

#ifndef __BLST_NO_ASM__
    bool_t vec_is_zero_16x(const void *a, size_t num);
    if ((num & 15) == 0)
        return vec_is_zero_16x(a, num);
#endif

    num /= sizeof(limb_t);

    for (acc = 0, i = 0; i < num; i++)
        acc |= ap[i];

    return is_zero(acc);
}

static inline bool_t vec_is_equal(const void *a, const void *b, size_t num)
{
    const limb_t *ap = (const limb_t *)a;
    const limb_t *bp = (const limb_t *)b;
    limb_t acc;
    size_t i;

#ifndef __BLST_NO_ASM__
    bool_t vec_is_equal_16x(const void *a, const void *b, size_t num);
    if ((num & 15) == 0)
        return vec_is_equal_16x(a, b, num);
#endif

    num /= sizeof(limb_t);

    for (acc = 0, i = 0; i < num; i++)
        acc |= ap[i] ^ bp[i];

    return is_zero(acc);
}

static inline void vec_copy(void *restrict ret, const void *a, size_t num)
{
    limb_t *rp = (limb_t *)ret;
    const limb_t *ap = (const limb_t *)a;
    size_t i;

    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = ap[i];
}

static inline void vec_zero(void *ret, size_t num)
{
    volatile limb_t *rp = (volatile limb_t *)ret;
    size_t i;

    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = 0;

#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : : "r"(ret) : "memory");
#endif
}

/*
 * Some compilers get arguably overzealous(*) when passing pointer to
 * multi-dimensional array [such as vec384x] as 'const' argument.
 * General direction seems to be to legitimize such constification,
 * so it's argued that suppressing the warning is appropriate.
 *
 * (*)  http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1923.htm
 */
#if defined(__INTEL_COMPILER)
# pragma warning(disable:167)
# pragma warning(disable:556)
#elif defined(__GNUC__) && !defined(__clang__)
# pragma GCC diagnostic ignored "-Wpedantic"
#elif defined(_MSC_VER)
# pragma warning(disable: 4127 4189)
#endif

#if !defined(__wasm__) && __STDC_HOSTED__-0 != 0
# include <stdlib.h>
#endif

#if defined(__GNUC__)
# ifndef alloca
#  define alloca(s) __builtin_alloca(s)
# endif
#elif defined(__sun)
# include <alloca.h>
#elif defined(_WIN32)
# include <malloc.h>
# ifndef alloca
#  define alloca(s) _alloca(s)
# endif
#endif

#endif /* __BLS12_381_ASM_VECT_H__ */
