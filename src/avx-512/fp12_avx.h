#ifndef _FP12_AVX_H
#define _FP12_AVX_H

#include <stdint.h>
#include "intrin.h"

// ----------------------------------------------------------------------------
// 48-bit-per-limb representation: 8 * 48-bit = 384-bit

#define BRADIX  48
#define NWORDS  8
#define VWORDS  4 // NWORDS / 2
#define BMASK   0xFFFFFFFFFFFFULL
#define BALIGN  4 // 52 (AVX-512IFMA) - 48 (BRADIX) = 4 

// ----------------------------------------------------------------------------
// p := 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

// prime p of the base field
const uint64_t BLS12_381_P_R48[NWORDS] = {
  0xFFFFFFFFAAABULL, 0xB153FFFFB9FEULL, 0xF6241EABFFFEULL, 0x6730D2A0F6B0ULL,
  0x4B84F38512BFULL, 0x434BACD76477ULL, 0xE69A4B1BA7B6ULL, 0x1A0111EA397FULL, };

// R = 2^384 mod p
#define ONE_MONT_P_R48 \
  0x00000002FFFDULL, 0xC40C00027609ULL, 0x58BAEBF4000BULL, 0x5F48985753C7ULL, \
  0x585370525745ULL, 0xA256EC6D77CEULL, 0xE4935C071A97ULL, 0x15F65EC3FA80ULL

// w = -1/p: constant in Montgomery reduction 
#define MONT_W_R48 0xFFFCFFFCFFFDULL

// ----------------------------------------------------------------------------
// modular operations

void add_fp_8x1w(__m512i *r, const __m512i *a, const __m512i *b);

#endif

