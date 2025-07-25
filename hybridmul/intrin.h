#ifndef _INTRIN_H
#define _INTRIN_H

// AVX-512 header file
#include <immintrin.h>

// integer arithmetic (AVX-512IFMA)
#define VADD(X, Y)            _mm512_add_epi64(X, Y)
#define VMADD(W, X, Y, Z)     _mm512_mask_add_epi64(W, X, Y, Z)
#define VSUB(X, Y)            _mm512_sub_epi64(X, Y)
#define VMSUB(W, X, Y, Z)     _mm512_mask_sub_epi64(W, X, Y, Z)
#define VMACLO(X, Y, Z)       _mm512_madd52lo_epu64(X, Y, Z)
#define VMACHI(X, Y, Z)       _mm512_madd52hi_epu64(X, Y, Z)

// bitwise logical 
#define VAND(X, Y)            _mm512_and_si512 (X, Y)
#define VOR(X, Y)             _mm512_or_si512(X, Y)
#define VXOR(X, Y)            _mm512_xor_si512 (X, Y)
#define VSHR(X, Y)            _mm512_srli_epi64(X, Y)
#define VSHL(X, Y)            _mm512_slli_epi64(X, Y)
#define VSLLV(X, Y)           _mm512_sllv_epi64(X, Y)
#define VSRA(X, Y)            _mm512_srai_epi64(X, Y)

// broadcast
#define VZERO                 _mm512_setzero_si512()
#define VSET1(X)              _mm512_set1_epi64(X)
#define VSET(X7, X6, X5, X4, X3, X2, X1, X0) \
                              _mm512_set_epi64(X7, X6, X5, X4, X3, X2, X1, X0)
#define VEXTR256(X, Y)        _mm512_extracti64x4_epi64(X, Y)
#define VEXTR64(X, Y)         _mm256_extract_epi64(X, Y) 

// permutation (256/512-bit, intra) & shuffle (128-bit) & blend
#define VSHUF(X, Y)           _mm512_shuffle_epi32(X, Y)            // 128-bit 
#define VZSHUF(X, Y, Z)       _mm512_maskz_shuffle_epi32(X, Y, Z)   // 128-bit
#define VPERM(X, Y)           _mm512_permutex_epi64(X, Y)           // 256-bit
#define VZPERM(X, Y, Z)       _mm512_maskz_permutex_epi64(X, Y, Z)  // 256-bit
#define VPERMV(X, Y)          _mm512_permutexvar_epi64(X, Y)        // 512-bit
#define VMBLEND(X, Y, Z)      _mm512_mask_blend_epi64(X, Y, Z)
#endif 
