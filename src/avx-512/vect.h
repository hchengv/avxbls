#ifndef _VECT_H
#define _VECT_H


#include <stdint.h>
#include "intrin.h"


// 48-bit-per-limb representation: 8 * 48-bit = 384-bit
#define BRADIX  48
#define NWORDS  8
#define BMASK   0xFFFFFFFFFFFFULL
#define BALIGN  4 // 52 (AVX-512IFMA) - 48 (BRADIX) = 4 


// Fp arithmetic prototypes 
void add_mod_384_8x1w(__m512i *r, const __m512i *a, const __m512i *b, 
                      const uint64_t *sp);
void sub_mod_384_8x1w(__m512i *r, const __m512i *a, const __m512i *b, 
                      const uint64_t *sp);

void mul_384_8x1w_v1(__m512i *r, const __m512i *a, const __m512i *b);
void mul_384_8x1w_v2(__m512i *r, const __m512i *a, const __m512i *b);
void sqr_384_8x1w(__m512i *r, const __m512i *a);

#endif