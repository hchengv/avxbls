#ifndef _FP12_AVX_H
#define _FP12_AVX_H

#include <stdint.h>
#include <stdio.h>
#include "intrin.h"
#include "fields.h"

// ----------------------------------------------------------------------------
// 48-bit-per-limb representation: 8 * 48-bit = 384-bit

#define BRADIX  48
#define NWORDS  8
#define VWORDS  4 // NWORDS / 2
#define SWORDS  6
#define BMASK   0xFFFFFFFFFFFFULL
#define BALIGN  4 // 52 (AVX-512IFMA) - 48 (BRADIX) = 4 

// Fp element
typedef __m512i   fp_8x1w[NWORDS];
typedef __m512i   fpx2_8x1w[2*NWORDS];
typedef __m512i   fp_4x2w[VWORDS];
typedef __m512i   fpx2_4x2w[3*VWORDS];
// Fp2 element
typedef fp_8x1w   fp2_8x1x1w[2];
typedef fpx2_8x1w fp2x2_8x1x1w[2];
typedef __m512i   fp2_4x2x1w[NWORDS];
typedef __m512i   fp2x2_4x2x1w[2*NWORDS];
typedef __m512i   fp2_2x4x1w[NWORDS];
typedef __m512i   fp2x2_2x4x1w[2*NWORDS];
typedef __m512i   fp2_2x2x2w[VWORDS];
typedef __m512i   fp2_1x4x2w[VWORDS];
// Fp4 element
typedef __m512i   fp4_2x2x2x1w[NWORDS];
typedef __m512i   fp4_1x2x2x2w[VWORDS];

// ----------------------------------------------------------------------------
// p := 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab;

// prime p of the base field
static uint64_t P48[NWORDS] = {
  0xFFFFFFFFAAABULL, 0xB153FFFFB9FEULL, 0xF6241EABFFFEULL, 0x6730D2A0F6B0ULL,
  0x4B84F38512BFULL, 0x434BACD76477ULL, 0xE69A4B1BA7B6ULL, 0x1A0111EA397FULL, };

// R = 2^384 mod p
#define ONE_MONT_P_R48 \
  0x00000002FFFDULL, 0xC40C00027609ULL, 0x58BAEBF4000BULL, 0x5F48985753C7ULL, \
  0x585370525745ULL, 0xA256EC6D77CEULL, 0xE4935C071A97ULL, 0x15F65EC3FA80ULL

// w = -1/p: constant in Montgomery reduction 
#define MONT_W_R48 0xFFFCFFFCFFFDULL

// ----------------------------------------------------------------------------
// prototypes: mp operations

#define mul_mp_8x1w mul_mp_8x1w_v2

void mul_mp_8x1w_v1(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b);
void mul_mp_8x1w_v2(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b);
void mul_mp_8x1w_v3(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b);
void mul_mp_8x1w_v4(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b);

#define mul_mp_4x2w mul_mp_4x2w_v1

void mul_mp_4x2w_v1(fpx2_4x2w r, const fpx2_4x2w a, const fpx2_4x2w b);

// ----------------------------------------------------------------------------
// prototypes: Fp single-length operations

void add_fp_8x1w(fp_8x1w r, const fp_8x1w a, const fp_8x1w b);
void sub_fp_8x1w(fp_8x1w r, const fp_8x1w a, const fp_8x1w b);

void add_fp_4x2w(fp_4x2w r, const fp_4x2w a, const fp_4x2w b);
void asx2_fp_4x2w(fp_4x2w r, const fp_4x2w a, const fp_4x2w b);

// ----------------------------------------------------------------------------
// prototypes: Fp double-length operations

void redc_fpx2_8x1w(fp_8x1w r, const fpx2_8x1w a);

void redc_fpx2_4x2w(fp_4x2w r, const fpx2_4x2w a);

// ----------------------------------------------------------------------------
// prototypes: Fp2 single-length operations

void assa_fp2_4x2x1w(fp2_4x2x1w r, const fp2_4x2x1w a, const fp2_4x2x1w b);

void as_fp2_2x2x2w(fp2_2x2x2w r, const fp2_2x2x2w a, const fp2_2x2x2w b);
void sqr_fp2_2x2x2w(fp2_2x2x2w r, const fp2_2x2x2w a);
void mul_by_u_plus_1_fp2_2x2x2w(fp2_2x2x2w r, const fp2_2x2x2w a);

// ----------------------------------------------------------------------------
// prototypes: Fp2 double-length operations

void sqr_fp2x2_4x2x1w(fp2x2_4x2x1w r, const fp2_4x2x1w a);
void mul_by_u_plus_1_fp2x2_4x2x1w(fp2x2_4x2x1w r, const fp2x2_4x2x1w a);

void mul_fp2x2_2x4x1w(fp2x2_2x4x1w r, const fp2_2x4x1w a, const fp2_2x4x1w b);

// ----------------------------------------------------------------------------
// prototypes: Fp4 operations

void sqr_fp4_2x2x2x1w_v1(fp4_2x2x2x1w r, const fp4_2x2x2x1w a);
void sqr_fp4_2x2x2x1w_v2(fp4_2x2x2x1w r, const fp4_2x2x2x1w a);

// ----------------------------------------------------------------------------
// prototypes: Fp12 operations

void cyclotomic_sqr_fp12_vec_v1(fp4_1x2x2x2w ra, fp4_2x2x2x1w rbc, 
                                const fp4_1x2x2x2w a, const fp4_2x2x2x1w bc);
void cyclotomic_sqr_fp12_vec_v2(fp4_1x2x2x2w ra, fp4_2x2x2x1w rbc, 
                                const fp4_1x2x2x2w a, const fp4_2x2x2x1w bc);

// ----------------------------------------------------------------------------
// utils 

static void mpi_print(const char *c, const uint64_t *a, int len)
{
  int i;

  printf("%s", c);
  for (i = len-1; i > 0; i--) printf("%016lX", a[i]);
  printf("%016lX\n", a[0]);
}

static void conv_64to48_fp(uint64_t *r, const uint64_t *a)
{
  r[0] =                  a[0]         & BMASK;
  r[1] = ((a[0] >> 48) | (a[1] << 16)) & BMASK;
  r[2] = ((a[1] >> 32) | (a[2] << 32)) & BMASK;
  r[3] =  (a[2] >> 16)                 & BMASK;
  r[4] =                  a[3]         & BMASK;
  r[5] = ((a[3] >> 48) | (a[4] << 16)) & BMASK;
  r[6] = ((a[4] >> 32) | (a[5] << 32)) & BMASK;
  r[7] =  (a[5] >> 16)                 & BMASK;
}

static void conv_64to48_fp_8x1w(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2];
  const __m512i a3 = a[3], a4 = a[4], a5 = a[5];
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VAND(                       a0,       bmask);
  r1 = VAND(VOR(VSHR(a0, 48), VSHL(a1, 16)), bmask);
  r2 = VAND(VOR(VSHR(a1, 32), VSHL(a2, 32)), bmask);
  r3 = VAND(    VSHR(a2, 16),                bmask);
  r4 = VAND(                       a3,       bmask);
  r5 = VAND(VOR(VSHR(a3, 48), VSHL(a4, 16)), bmask);
  r6 = VAND(VOR(VSHR(a4, 32), VSHL(a5, 32)), bmask);
  r7 = VAND(    VSHR(a5, 16),                bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

static void conv_64to48_fp_4x2w(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2];
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3;

  r0 = VAND(                       a0,       bmask);
  r1 = VAND(VOR(VSHR(a0, 48), VSHL(a1, 16)), bmask);
  r2 = VAND(VOR(VSHR(a1, 32), VSHL(a2, 32)), bmask);
  r3 = VAND(    VSHR(a2, 16),                bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
}

static void conv_64to48_mpi(uint64_t *r, const uint64_t *a, int rlen, int alen)
{
  int i, j, shr_pos, shl_pos;
  uint64_t word, temp;

  i = j = 0;
  shr_pos = 64; shl_pos = 0;
  temp = 0;
  while ((i < rlen) && (j < alen)) {
    word = ((temp >> shr_pos) | (a[j] << shl_pos));
    r[i] = (word & BMASK);
    shr_pos -= 16, shl_pos += 16;
    if ((shr_pos > 0) && (shl_pos < 64)) temp = a[j++];
    if (shr_pos <= 0) shr_pos += 64;
    if (shl_pos >= 64) shl_pos -= 64;
    // Any shift past 63 is undefined!
    if (shr_pos == 64) temp = 0;
    i++;
  }
  if (i < rlen) r[i++] = ((temp >> shr_pos) & BMASK);
  for (; i < rlen; i++) r[i] = 0;
}

static void conv_48to64_fp(uint64_t *r, const uint64_t *a)
{
  r[0] = (a[1] << 48) | a[0];
  r[1] = (a[2] << 32) | a[1] >> 16;
  r[2] = (a[3] << 16) | a[2] >> 32;
  r[3] = (a[5] << 48) | a[4];
  r[4] = (a[6] << 32) | a[5] >> 16;
  r[5] = (a[7] << 16) | a[6] >> 32;  
}

static void conv_48to64_fp_8x1w(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5;

  r0 = VOR(VSHL(a1, 48),      a0     );
  r1 = VOR(VSHL(a2, 32), VSHR(a1, 16));
  r2 = VOR(VSHL(a3, 16), VSHR(a2, 32));
  r3 = VOR(VSHL(a5, 48),      a4     );
  r4 = VOR(VSHL(a6, 32), VSHR(a5, 16));
  r5 = VOR(VSHL(a7, 16), VSHR(a6, 32));

  r[0] = r0; r[1] = r1; r[2] = r2; 
  r[3] = r3; r[4] = r4; r[5] = r5; 
}

static void conv_48to64_fp_4x2w(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  __m512i r0, r1, r2;

  r0 = VOR(VSHL(a1, 48),      a0     );
  r1 = VOR(VSHL(a2, 32), VSHR(a1, 16));
  r2 = VOR(VSHL(a3, 16), VSHR(a2, 32));

  r[0] = r0; r[1] = r1; r[2] = r2; 
}

static void conv_48to64_mpi(uint64_t *r, const uint64_t *a, int rlen, int alen)
{
  int i, j, bits_in_word, bits_to_shift;
  uint64_t word;

  i = j = 0;
  bits_in_word = bits_to_shift = 0;
  word = 0;
  while ((i < rlen) && (j < alen)) {
    word |= (a[j] << bits_in_word);
    bits_to_shift = (64 - bits_in_word);
    bits_in_word += 48;
    if (bits_in_word >= 64) {
      r[i++] = word;
      word = ((bits_to_shift > 0) ? (a[j] >> bits_to_shift) : 0);
      bits_in_word = ((bits_to_shift > 0) ? (48 - bits_to_shift) : 0);
    }
    j++;
  }
  if (i < rlen) r[i++] = word;
  for (; i < rlen; i++) r[i] = 0;
}

static void carryp_mpi48(uint64_t *a)
{
  int i;

  for (i = 0; i < NWORDS-1; i++) {
    a[i+1] += a[i]>>BRADIX;
    a[i] &= BMASK;
  }
}

static void carryp_fp_4x2w(__m512i *a)
{
  __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i bmask = VSET1(BMASK);

  a1 = VADD(a1, VSRA(a0, BRADIX)); a0 = VAND(a0, bmask);
  a2 = VADD(a2, VSRA(a1, BRADIX)); a1 = VAND(a1, bmask);
  a3 = VADD(a3, VSRA(a2, BRADIX)); a2 = VAND(a2, bmask);

  a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
}

static void get_channel_8x1w(uint64_t *r, const __m512i *a, const int ch) 
{
  int i;

  for(i = 0; i < NWORDS; i++) {
    r[i] = ((uint64_t *)&a[i])[ch];
  }
}

static void get_channel_4x2w(uint64_t *r, const __m512i *a, const int ch) 
{
  int i;

  for(i = 0; i < VWORDS; i++) {
    r[i] = ((uint64_t *)&a[i])[ch];
    r[i+VWORDS] = ((uint64_t *)&a[i])[ch+1];
  }
}

// ----------------------------------------------------------------------------

#endif

