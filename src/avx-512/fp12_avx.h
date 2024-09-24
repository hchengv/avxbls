#ifndef _FP12_AVX_H
#define _FP12_AVX_H

#include <stdint.h>
#include <stdio.h>
#include "intrin.h"

// ----------------------------------------------------------------------------
// 48-bit-per-limb representation: 8 * 48-bit = 384-bit

#define BRADIX  48
#define NWORDS  8
#define VWORDS  4 // NWORDS / 2
#define BMASK   0xFFFFFFFFFFFFULL
#define BALIGN  4 // 52 (AVX-512IFMA) - 48 (BRADIX) = 4 

// ----------------------------------------------------------------------------
// p := 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab;

// prime p of the base field
static uint64_t BLS12_381_P_R48[NWORDS] = {
  0xFFFFFFFFAAABULL, 0xB153FFFFB9FEULL, 0xF6241EABFFFEULL, 0x6730D2A0F6B0ULL,
  0x4B84F38512BFULL, 0x434BACD76477ULL, 0xE69A4B1BA7B6ULL, 0x1A0111EA397FULL, };

// R = 2^384 mod p
#define ONE_MONT_P_R48 \
  0x00000002FFFDULL, 0xC40C00027609ULL, 0x58BAEBF4000BULL, 0x5F48985753C7ULL, \
  0x585370525745ULL, 0xA256EC6D77CEULL, 0xE4935C071A97ULL, 0x15F65EC3FA80ULL

// w = -1/p: constant in Montgomery reduction 
#define MONT_W_R48 0xFFFCFFFCFFFDULL

// ----------------------------------------------------------------------------
// prototypes: modular operations

void add_fp_8x1w(__m512i *r, const __m512i *a, const __m512i *b);


// ----------------------------------------------------------------------------
// utils 

static void mpi_print(const char *c, const uint64_t *a, int len)
{
  int i;

  printf("%s", c);
  for (i = len-1; i > 0; i--) printf("%016lX", a[i]);
  printf("%016lX\n", a[0]);
}


static void mpi_conv_64to48(uint64_t *r, const uint64_t *a, int rlen, int alen)
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

static void mpi_conv_48to64(uint64_t *r, const uint64_t *a, int rlen, int alen)
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

static void get_channel_8x1w(uint64_t *r, const __m512i *a, const int ch) 
{
  int i;

  for(i = 0; i < NWORDS; i++) {
    r[i] = ((uint64_t *)&a[i])[ch];
  }
}



#endif

