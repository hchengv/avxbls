#include "fp12_avx.h"


// ----------------------------------------------------------------------------
// vector single-length transform 

// a = < H | G | F | E | D | C | B | A >
// r = < G | G | E | E | C | C | A | A >
static void shuf_00(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VSHUF(a0, 0x44); r1 = VSHUF(a1, 0x44);
  r2 = VSHUF(a2, 0x44); r3 = VSHUF(a3, 0x44);
  r4 = VSHUF(a4, 0x44); r5 = VSHUF(a5, 0x44);
  r6 = VSHUF(a6, 0x44); r7 = VSHUF(a7, 0x44);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// r = < G | H | E | F | C | D | A | B >
static void shuf_01(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VSHUF(a0, 0x4E); r1 = VSHUF(a1, 0x4E);
  r2 = VSHUF(a2, 0x4E); r3 = VSHUF(a3, 0x4E);
  r4 = VSHUF(a4, 0x4E); r5 = VSHUF(a5, 0x4E);
  r6 = VSHUF(a6, 0x4E); r7 = VSHUF(a7, 0x4E);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

static void shuf_01_fp2_8x1x1w(fp2_8x1x1w r, const fp2_8x1x1w a)
{
  shuf_01(r[0], a[0]);
  shuf_01(r[1], a[1]);
}

// a = < H | G | F | E | D | C | B | A >
// r = < H | H | F | F | D | D | B | B >
static void shuf_11(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VSHUF(a0, 0xEE); r1 = VSHUF(a1, 0xEE);
  r2 = VSHUF(a2, 0xEE); r3 = VSHUF(a3, 0xEE);
  r4 = VSHUF(a4, 0xEE); r5 = VSHUF(a5, 0xEE);
  r6 = VSHUF(a6, 0xEE); r7 = VSHUF(a7, 0xEE);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// r = < 0 | H | 0 | F | 0 | D | 0 | B >
static void shuf_z1(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VZSHUF(0x3333, a0, 0xEE); r1 = VZSHUF(0x3333, a1, 0xEE);
  r2 = VZSHUF(0x3333, a2, 0xEE); r3 = VZSHUF(0x3333, a3, 0xEE);
  r4 = VZSHUF(0x3333, a4, 0xEE); r5 = VZSHUF(0x3333, a5, 0xEE);
  r6 = VZSHUF(0x3333, a6, 0xEE); r7 = VZSHUF(0x3333, a7, 0xEE);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// r = < F | E | 0 | 0 | B | A | 0 | 0 >
static void perm_10zz(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VZPERM(0xCC, a0, 0x44); r1 = VZPERM(0xCC, a1, 0x44);
  r2 = VZPERM(0xCC, a2, 0x44); r3 = VZPERM(0xCC, a3, 0x44);
  r4 = VZPERM(0xCC, a4, 0x44); r5 = VZPERM(0xCC, a5, 0x44);
  r6 = VZPERM(0xCC, a6, 0x44); r7 = VZPERM(0xCC, a7, 0x44);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// r = < F | E | H | G | B | A | D | C >
static void perm_1032(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0  = VPERM(a0 , 0x4E); r1  = VPERM(a1 , 0x4E);
  r2  = VPERM(a2 , 0x4E); r3  = VPERM(a3 , 0x4E);
  r4  = VPERM(a4 , 0x4E); r5  = VPERM(a5 , 0x4E);
  r6  = VPERM(a6 , 0x4E); r7  = VPERM(a7 , 0x4E);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// r = < H | H | G | G | D | D | C | C >
static void perm_3322(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VPERM(a0, 0xFA); r1 = VPERM(a1, 0xFA);
  r2 = VPERM(a2, 0xFA); r3 = VPERM(a3, 0xFA);
  r4 = VPERM(a4, 0xFA); r5 = VPERM(a5, 0xFA);
  r6 = VPERM(a6, 0xFA); r7 = VPERM(a7, 0xFA);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// r = < G | H | H | G | C | D | D | C >
static void perm_2332(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VPERM(a0, 0xBE); r1 = VPERM(a1, 0xBE);
  r2 = VPERM(a2, 0xBE); r3 = VPERM(a3, 0xBE);
  r4 = VPERM(a4, 0xBE); r5 = VPERM(a5, 0xBE);
  r6 = VPERM(a6, 0xBE); r7 = VPERM(a7, 0xBE);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// r = < 0 | 0 | H | G | 0 | 0 | D | C >
static void perm_zz32(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VZPERM(0x33, a0, 0xEE); r1 = VZPERM(0x33, a1, 0xEE);
  r2 = VZPERM(0x33, a2, 0xEE); r3 = VZPERM(0x33, a3, 0xEE);
  r4 = VZPERM(0x33, a4, 0xEE); r5 = VZPERM(0x33, a5, 0xEE);
  r6 = VZPERM(0x33, a6, 0xEE); r7 = VZPERM(0x33, a7, 0xEE);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H | G | N | M | D | C | J | I >
static void blend_0x33(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VMBLEND(0x33, a0, b0); r1 = VMBLEND(0x33, a1, b1);
  r2 = VMBLEND(0x33, a2, b2); r3 = VMBLEND(0x33, a3, b3);
  r4 = VMBLEND(0x33, a4, b4); r5 = VMBLEND(0x33, a5, b5);
  r6 = VMBLEND(0x33, a6, b6); r7 = VMBLEND(0x33, a7, b7);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H | O | F | M | D | K | B | I >
static void blend_0x55(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VMBLEND(0x55, a0, b0); r1 = VMBLEND(0x55, a1, b1);
  r2 = VMBLEND(0x55, a2, b2); r3 = VMBLEND(0x55, a3, b3);
  r4 = VMBLEND(0x55, a4, b4); r5 = VMBLEND(0x55, a5, b5);
  r6 = VMBLEND(0x55, a6, b6); r7 = VMBLEND(0x55, a7, b7);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

static void blend_0x55_fp2_8x1x1w(fp2_8x1x1w r, const fp2_8x1x1w a, const fp2_8x1x1w b)
{
  blend_0x55(r[0], a[0], b[0]);
  blend_0x55(r[1], a[1], b[1]);
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < P | O | N | M | D | C | B | A >
static void blend_0xF0(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VMBLEND(0xF0, a0, b0); r1 = VMBLEND(0xF0, a1, b1);
  r2 = VMBLEND(0xF0, a2, b2); r3 = VMBLEND(0xF0, a3, b3);
  r4 = VMBLEND(0xF0, a4, b4); r5 = VMBLEND(0xF0, a5, b5);
  r6 = VMBLEND(0xF0, a6, b6); r7 = VMBLEND(0xF0, a7, b7);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}


// ----------------------------------------------------------------------------
// vector double-length transform 

// a = < H | G | F | E | D | C | B | A >
// r = < G | G | E | E | C | C | A | A >
static void shuf_00_dl(__m512i *r, const __m512i *a)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VSHUF(a0 , 0x44); r1  = VSHUF(a1 , 0x44);
  r2  = VSHUF(a2 , 0x44); r3  = VSHUF(a3 , 0x44);
  r4  = VSHUF(a4 , 0x44); r5  = VSHUF(a5 , 0x44);
  r6  = VSHUF(a6 , 0x44); r7  = VSHUF(a7 , 0x44);
  r8  = VSHUF(a8 , 0x44); r9  = VSHUF(a9 , 0x44);
  r10 = VSHUF(a10, 0x44); r11 = VSHUF(a11, 0x44);
  r12 = VSHUF(a12, 0x44); r13 = VSHUF(a13, 0x44);
  r14 = VSHUF(a14, 0x44); r15 = VSHUF(a15, 0x44);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

static void shuf_00_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a)
{
  shuf_00_dl(r[0], a[0]);
  shuf_00_dl(r[1], a[1]);
}

// a = < H | G | F | E | D | C | B | A >
// r = < G | H | E | F | C | D | A | B >
static void shuf_01_dl(__m512i *r, const __m512i *a)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VSHUF(a0 , 0x4E); r1  = VSHUF(a1 , 0x4E);
  r2  = VSHUF(a2 , 0x4E); r3  = VSHUF(a3 , 0x4E);
  r4  = VSHUF(a4 , 0x4E); r5  = VSHUF(a5 , 0x4E);
  r6  = VSHUF(a6 , 0x4E); r7  = VSHUF(a7 , 0x4E);
  r8  = VSHUF(a8 , 0x4E); r9  = VSHUF(a9 , 0x4E);
  r10 = VSHUF(a10, 0x4E); r11 = VSHUF(a11, 0x4E);
  r12 = VSHUF(a12, 0x4E); r13 = VSHUF(a13, 0x4E);
  r14 = VSHUF(a14, 0x4E); r15 = VSHUF(a15, 0x4E);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

static void shuf_01_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a)
{
  shuf_01_dl(r[0], a[0]);
  shuf_01_dl(r[1], a[1]);
}

// a = < H | G | F | E | D | C | B | A >
// r = < H | H | F | F | D | D | B | B >
static void shuf_11_dl(__m512i *r, const __m512i *a)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VSHUF(a0 , 0xEE); r1  = VSHUF(a1 , 0xEE);
  r2  = VSHUF(a2 , 0xEE); r3  = VSHUF(a3 , 0xEE);
  r4  = VSHUF(a4 , 0xEE); r5  = VSHUF(a5 , 0xEE);
  r6  = VSHUF(a6 , 0xEE); r7  = VSHUF(a7 , 0xEE);
  r8  = VSHUF(a8 , 0xEE); r9  = VSHUF(a9 , 0xEE);
  r10 = VSHUF(a10, 0xEE); r11 = VSHUF(a11, 0xEE);
  r12 = VSHUF(a12, 0xEE); r13 = VSHUF(a13, 0xEE);
  r14 = VSHUF(a14, 0xEE); r15 = VSHUF(a15, 0xEE);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

static void shuf_11_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a)
{
  shuf_11_dl(r[0], a[0]);
  shuf_11_dl(r[1], a[1]);  
}

// a = < H | G | F | E | D | C | B | A >
// r = < 0 | 0 | H | G | 0 | 0 | D | C >
static void perm_zz32_dl(__m512i *r, const __m512i *a)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VZPERM(0x33, a0 , 0xEE); r1  = VZPERM(0x33, a1 , 0xEE);
  r2  = VZPERM(0x33, a2 , 0xEE); r3  = VZPERM(0x33, a3 , 0xEE);
  r4  = VZPERM(0x33, a4 , 0xEE); r5  = VZPERM(0x33, a5 , 0xEE);
  r6  = VZPERM(0x33, a6 , 0xEE); r7  = VZPERM(0x33, a7 , 0xEE);
  r8  = VZPERM(0x33, a8 , 0xEE); r9  = VZPERM(0x33, a9 , 0xEE);
  r10 = VZPERM(0x33, a10, 0xEE); r11 = VZPERM(0x33, a11, 0xEE);
  r12 = VZPERM(0x33, a12, 0xEE); r13 = VZPERM(0x33, a13, 0xEE);
  r14 = VZPERM(0x33, a14, 0xEE); r15 = VZPERM(0x33, a15, 0xEE);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

// a = < H | G | F | E | D | C | B | A >
// r = < F | E | 0 | 0 | B | A | 0 | 0 >
static void perm_10zz_dl(__m512i *r, const __m512i *a)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VZPERM(0xCC, a0 , 0x44); r1  = VZPERM(0xCC, a1 , 0x44);
  r2  = VZPERM(0xCC, a2 , 0x44); r3  = VZPERM(0xCC, a3 , 0x44);
  r4  = VZPERM(0xCC, a4 , 0x44); r5  = VZPERM(0xCC, a5 , 0x44);
  r6  = VZPERM(0xCC, a6 , 0x44); r7  = VZPERM(0xCC, a7 , 0x44);
  r8  = VZPERM(0xCC, a8 , 0x44); r9  = VZPERM(0xCC, a9 , 0x44);
  r10 = VZPERM(0xCC, a10, 0x44); r11 = VZPERM(0xCC, a11, 0x44);
  r12 = VZPERM(0xCC, a12, 0x44); r13 = VZPERM(0xCC, a13, 0x44);
  r14 = VZPERM(0xCC, a14, 0x44); r15 = VZPERM(0xCC, a15, 0x44);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

// a = < H | G | F | E | D | C | B | A >
// r = < F | E | H | G | B | A | D | C >
static void perm_1032_dl(__m512i *r, const __m512i *a)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VPERM(a0 , 0x4E); r1  = VPERM(a1 , 0x4E);
  r2  = VPERM(a2 , 0x4E); r3  = VPERM(a3 , 0x4E);
  r4  = VPERM(a4 , 0x4E); r5  = VPERM(a5 , 0x4E);
  r6  = VPERM(a6 , 0x4E); r7  = VPERM(a7 , 0x4E);
  r8  = VPERM(a8 , 0x4E); r9  = VPERM(a9 , 0x4E);
  r10 = VPERM(a10, 0x4E); r11 = VPERM(a11, 0x4E);
  r12 = VPERM(a12, 0x4E); r13 = VPERM(a13, 0x4E);
  r14 = VPERM(a14, 0x4E); r15 = VPERM(a15, 0x4E);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

static void perm_var_dl(__m512i *r, const __m512i *a, const __m512i mask)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VPERMV(mask, a0 ); r1  = VPERMV(mask, a1 );
  r2  = VPERMV(mask, a2 ); r3  = VPERMV(mask, a3 );
  r4  = VPERMV(mask, a4 ); r5  = VPERMV(mask, a5 );
  r6  = VPERMV(mask, a6 ); r7  = VPERMV(mask, a7 );
  r8  = VPERMV(mask, a8 ); r9  = VPERMV(mask, a9 );
  r10 = VPERMV(mask, a10); r11 = VPERMV(mask, a11);
  r12 = VPERMV(mask, a12); r13 = VPERMV(mask, a13);
  r14 = VPERMV(mask, a14); r15 = VPERMV(mask, a15);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H | G | F | E | D | C | B | I >
static void blend_0x01_dl(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  const __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  const __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VMBLEND(0x01, a0 , b0 ); r1  = VMBLEND(0x01, a1 , b1 );
  r2  = VMBLEND(0x01, a2 , b2 ); r3  = VMBLEND(0x01, a3 , b3 );
  r4  = VMBLEND(0x01, a4 , b4 ); r5  = VMBLEND(0x01, a5 , b5 );
  r6  = VMBLEND(0x01, a6 , b6 ); r7  = VMBLEND(0x01, a7 , b7 );
  r8  = VMBLEND(0x01, a8 , b8 ); r9  = VMBLEND(0x01, a9 , b9 );
  r10 = VMBLEND(0x01, a10, b10); r11 = VMBLEND(0x01, a11, b11);
  r12 = VMBLEND(0x01, a12, b12); r13 = VMBLEND(0x01, a13, b13);
  r14 = VMBLEND(0x01, a14, b14); r15 = VMBLEND(0x01, a15, b15);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

static void blend_0x01_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a, const fp2x2_8x1x1w b)
{
  blend_0x01_dl(r[0], a[0], b[0]);
  blend_0x01_dl(r[1], a[1], b[1]);
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H | G | F | E | D | C | J | I >
static void blend_0x03_dl(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  const __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  const __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VMBLEND(0x03, a0 , b0 ); r1  = VMBLEND(0x03, a1 , b1 );
  r2  = VMBLEND(0x03, a2 , b2 ); r3  = VMBLEND(0x03, a3 , b3 );
  r4  = VMBLEND(0x03, a4 , b4 ); r5  = VMBLEND(0x03, a5 , b5 );
  r6  = VMBLEND(0x03, a6 , b6 ); r7  = VMBLEND(0x03, a7 , b7 );
  r8  = VMBLEND(0x03, a8 , b8 ); r9  = VMBLEND(0x03, a9 , b9 );
  r10 = VMBLEND(0x03, a10, b10); r11 = VMBLEND(0x03, a11, b11);
  r12 = VMBLEND(0x03, a12, b12); r13 = VMBLEND(0x03, a13, b13);
  r14 = VMBLEND(0x03, a14, b14); r15 = VMBLEND(0x03, a15, b15);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

static void blend_0x03_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a, const fp2x2_8x1x1w b)
{
  blend_0x03_dl(r[0], a[0], b[0]);
  blend_0x03_dl(r[1], a[1], b[1]);
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H | G | N | M | D | C | J | I >
static void blend_0x33_dl(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  const __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  const __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VMBLEND(0x33, a0 , b0 ); r1  = VMBLEND(0x33, a1 , b1 );
  r2  = VMBLEND(0x33, a2 , b2 ); r3  = VMBLEND(0x33, a3 , b3 );
  r4  = VMBLEND(0x33, a4 , b4 ); r5  = VMBLEND(0x33, a5 , b5 );
  r6  = VMBLEND(0x33, a6 , b6 ); r7  = VMBLEND(0x33, a7 , b7 );
  r8  = VMBLEND(0x33, a8 , b8 ); r9  = VMBLEND(0x33, a9 , b9 );
  r10 = VMBLEND(0x33, a10, b10); r11 = VMBLEND(0x33, a11, b11);
  r12 = VMBLEND(0x33, a12, b12); r13 = VMBLEND(0x33, a13, b13);
  r14 = VMBLEND(0x33, a14, b14); r15 = VMBLEND(0x33, a15, b15);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H | O | F | M | D | K | B | I >
static void blend_0x55_dl(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  const __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  const __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VMBLEND(0x55, a0 , b0 ); r1  = VMBLEND(0x55, a1 , b1 );
  r2  = VMBLEND(0x55, a2 , b2 ); r3  = VMBLEND(0x55, a3 , b3 );
  r4  = VMBLEND(0x55, a4 , b4 ); r5  = VMBLEND(0x55, a5 , b5 );
  r6  = VMBLEND(0x55, a6 , b6 ); r7  = VMBLEND(0x55, a7 , b7 );
  r8  = VMBLEND(0x55, a8 , b8 ); r9  = VMBLEND(0x55, a9 , b9 );
  r10 = VMBLEND(0x55, a10, b10); r11 = VMBLEND(0x55, a11, b11);
  r12 = VMBLEND(0x55, a12, b12); r13 = VMBLEND(0x55, a13, b13);
  r14 = VMBLEND(0x55, a14, b14); r15 = VMBLEND(0x55, a15, b15);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

static void blend_0x55_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a, const fp2x2_8x1x1w b)
{
  blend_0x55_dl(r[0], a[0], b[0]);
  blend_0x55_dl(r[1], a[1], b[1]);
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H | G | N | M | D | C | J | I >
static void blend_0xC0_dl(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  const __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  const __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15;

  r0  = VMBLEND(0xC0, a0 , b0 ); r1  = VMBLEND(0xC0, a1 , b1 );
  r2  = VMBLEND(0xC0, a2 , b2 ); r3  = VMBLEND(0xC0, a3 , b3 );
  r4  = VMBLEND(0xC0, a4 , b4 ); r5  = VMBLEND(0xC0, a5 , b5 );
  r6  = VMBLEND(0xC0, a6 , b6 ); r7  = VMBLEND(0xC0, a7 , b7 );
  r8  = VMBLEND(0xC0, a8 , b8 ); r9  = VMBLEND(0xC0, a9 , b9 );
  r10 = VMBLEND(0xC0, a10, b10); r11 = VMBLEND(0xC0, a11, b11);
  r12 = VMBLEND(0xC0, a12, b12); r13 = VMBLEND(0xC0, a13, b13);
  r14 = VMBLEND(0xC0, a14, b14); r15 = VMBLEND(0xC0, a15, b15);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

// ----------------------------------------------------------------------------
// vector half-length transform 

// a = < H | G | F | E | D | C | B | A >
// r = < F | E | F | E | B | A | B | A >
static void perm_1010_hl(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  __m512i r0, r1, r2, r3;

  r0 = VPERM(a0, 0x44); r1 = VPERM(a1, 0x44);
  r2 = VPERM(a2, 0x44); r3 = VPERM(a3, 0x44);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
}

// a = < H | G | F | E | D | C | B | A >
// r = < F | E | H | G | B | A | D | C >
static void perm_1032_hl(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  __m512i r0, r1, r2, r3;

  r0 = VPERM(a0, 0x4E); r1 = VPERM(a1, 0x4E);
  r2 = VPERM(a2, 0x4E); r3 = VPERM(a3, 0x4E);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
}

// a = < H | G | F | E | D | C | B | A >
// r = < 0 | 0 | H | G | 0 | 0 | D | C >
static void perm_zz32_hl(__m512i *r, const __m512i *a)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  __m512i r0, r1, r2, r3;

  r0 = VZPERM(0x33, a0, 0xEE); r1 = VZPERM(0x33, a1, 0xEE);
  r2 = VZPERM(0x33, a2, 0xEE); r3 = VZPERM(0x33, a3, 0xEE);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
}

static void perm_var_hl(__m512i *r, const __m512i *a, const __m512i mask)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  __m512i r0, r1, r2, r3, r4, r5, r6, r7;

  r0 = VPERMV(mask, a0); r1 = VPERMV(mask, a1);
  r2 = VPERMV(mask, a2); r3 = VPERMV(mask, a3);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < P | O | N | M | D | C | B | A >
static void blend_0xF0_hl(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  __m512i r0, r1, r2, r3;

  r0 = VMBLEND(0xF0, a0, b0); r1 = VMBLEND(0xF0, a1, b1);
  r2 = VMBLEND(0xF0, a2, b2); r3 = VMBLEND(0xF0, a3, b3);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H | O | F | M | D | K | B | I >
static void blend_0x55_hl(__m512i *r, const __m512i *a, const __m512i *b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  __m512i r0, r1, r2, r3;

  r0 = VMBLEND(0x55, a0, b0); r1 = VMBLEND(0x55, a1, b1);
  r2 = VMBLEND(0x55, a2, b2); r3 = VMBLEND(0x55, a3, b3);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
}

// ----------------------------------------------------------------------------
// Fp single-length operations

static void add_fp_8x1w(fp_8x1w r, const fp_8x1w a, const fp_8x1w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask, t0;

  // r = a + b
  r0 = VADD(a0, b0); r1 = VADD(a1, b1); r2 = VADD(a2, b2); r3 = VADD(a3, b3);
  r4 = VADD(a4, b4); r5 = VADD(a5, b5); r6 = VADD(a6, b6); r7 = VADD(a7, b7);

  // r = r - p
  r0 = VSUB(r0, p0); r1 = VSUB(r1, p1); r2 = VSUB(r2, p2); r3 = VSUB(r3, p3);
  r4 = VSUB(r4, p4); r5 = VSUB(r5, p5); r6 = VSUB(r6, p6); r7 = VSUB(r7, p7);

  // get sign mask 
  t0 = VADD(r1, VSRA(r0, BRADIX));
  t0 = VADD(r2, VSRA(t0, BRADIX));
  t0 = VADD(r3, VSRA(t0, BRADIX));
  t0 = VADD(r4, VSRA(t0, BRADIX));
  t0 = VADD(r5, VSRA(t0, BRADIX));
  t0 = VADD(r6, VSRA(t0, BRADIX));
  t0 = VADD(r7, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 
  r4 = VADD(r4, VAND(p4, smask)); r5 = VADD(r5, VAND(p5, smask)); 
  r6 = VADD(r6, VAND(p6, smask)); r7 = VADD(r7, VAND(p7, smask)); 

  // carry propagation 
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4, VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5, VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6, VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7, VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

static void sub_fp_8x1w(fp_8x1w r, const fp_8x1w a, const fp_8x1w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask, t0;

  // r = a - b
  r0 = VSUB(a0, b0); r1 = VSUB(a1, b1); r2 = VSUB(a2, b2); r3 = VSUB(a3, b3);
  r4 = VSUB(a4, b4); r5 = VSUB(a5, b5); r6 = VSUB(a6, b6); r7 = VSUB(a7, b7);

  // get sign mask 
  t0 = VADD(r1, VSRA(r0, BRADIX));
  t0 = VADD(r2, VSRA(t0, BRADIX));
  t0 = VADD(r3, VSRA(t0, BRADIX));
  t0 = VADD(r4, VSRA(t0, BRADIX));
  t0 = VADD(r5, VSRA(t0, BRADIX));
  t0 = VADD(r6, VSRA(t0, BRADIX));
  t0 = VADD(r7, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 
  r4 = VADD(r4, VAND(p4, smask)); r5 = VADD(r5, VAND(p5, smask)); 
  r6 = VADD(r6, VAND(p6, smask)); r7 = VADD(r7, VAND(p7, smask)); 

  // carry propagation 
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4, VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5, VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6, VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7, VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H+P | G-O | F+N | E-M | D+L | C-K | B+J | A-I >
static void asx4_fp_8x1w(fp_8x1w r, const fp_8x1w a, const fp_8x1w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask;
  __m512i t0, t1, t2, t3, t4, t5, t6, t7;

  // r = H+P | G | F+N | E | D+L | C | B+J | A
  r0  = VMADD(a0, 0xAA, a0, b0); r1  = VMADD(a1, 0xAA, a1, b1);
  r2  = VMADD(a2, 0xAA, a2, b2); r3  = VMADD(a3, 0xAA, a3, b3);
  r4  = VMADD(a4, 0xAA, a4, b4); r5  = VMADD(a5, 0xAA, a5, b5);
  r6  = VMADD(a6, 0xAA, a6, b6); r7  = VMADD(a7, 0xAA, a7, b7);

  // t = p | O | p | M | p | K | p | I
  t0 = VMBLEND(0xAA, b0, p0); t1 = VMBLEND(0xAA, b1, p1);
  t2 = VMBLEND(0xAA, b2, p2); t3 = VMBLEND(0xAA, b3, p3);
  t4 = VMBLEND(0xAA, b4, p4); t5 = VMBLEND(0xAA, b5, p5);
  t6 = VMBLEND(0xAA, b6, p6); t7 = VMBLEND(0xAA, b7, p7); 

  // r = H+P-p | G-O | F+N-p | E-M | D+L-p | C-K | B+J-p | A-I
  r0 = VSUB(r0, t0); r1 = VSUB(r1, t1);
  r2 = VSUB(r2, t2); r3 = VSUB(r3, t3);
  r4 = VSUB(r4, t4); r5 = VSUB(r5, t5);
  r6 = VSUB(r6, t6); r7 = VSUB(r7, t7);

  // get sign mask 
  t0 = VADD(r1, VSRA(r0, BRADIX));
  t0 = VADD(r2, VSRA(t0, BRADIX));
  t0 = VADD(r3, VSRA(t0, BRADIX));
  t0 = VADD(r4, VSRA(t0, BRADIX));
  t0 = VADD(r5, VSRA(t0, BRADIX));
  t0 = VADD(r6, VSRA(t0, BRADIX));
  t0 = VADD(r7, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 
  r4 = VADD(r4, VAND(p4, smask)); r5 = VADD(r5, VAND(p5, smask)); 
  r6 = VADD(r6, VAND(p6, smask)); r7 = VADD(r7, VAND(p7, smask)); 

  // carry propagation 
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4, VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5, VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6, VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7, VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

static void add_fp_4x2w(fp_4x2w r, const fp_4x2w a, const fp_4x2w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask = VSET1(BMASK); 
  __m512i r0, r1, r2, r3, smask, t0;

  // r = a + b
  r0 = VADD(a0, b0); r1 = VADD(a1, b1); r2 = VADD(a2, b2); r3 = VADD(a3, b3);

  // r = r - p
  r0 = VSUB(r0, p0); r1 = VSUB(r1, p1); r2 = VSUB(r2, p2); r3 = VSUB(r3, p3);

  // get sign mask
  t0 = VMADD(r1, 0x55, r1, VSRA(r0, BRADIX));
  t0 = VMADD(r2, 0x55, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0x55, r3, VSRA(t0, BRADIX));
  t0 = VMADD(r0, 0xAA, r0, VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0 = VMADD(r1, 0xAA, r1, VSRA(t0, BRADIX));
  t0 = VMADD(r2, 0xAA, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0xAA, r3, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 

  // carry propagation 
  // r0 is finally 49-bit not 48-bit
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r0 = VMADD(r0, 0xAA, r0, VSHUF(VSRA(r3, BRADIX), 0x4E)); r3 = VAND(r3, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
}

static void sub_fp_4x2w(fp_4x2w r, const fp_4x2w a, const fp_4x2w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask = VSET1(BMASK); 
  __m512i r0, r1, r2, r3, smask, t0;

  // r = a - b
  r0 = VSUB(a0, b0); r1 = VSUB(a1, b1); r2 = VSUB(a2, b2); r3 = VSUB(a3, b3);

  // get sign mask
  t0 = VMADD(r1, 0x55, r1, VSRA(r0, BRADIX));
  t0 = VMADD(r2, 0x55, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0x55, r3, VSRA(t0, BRADIX));
  t0 = VMADD(r0, 0xAA, r0, VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0 = VMADD(r1, 0xAA, r1, VSRA(t0, BRADIX));
  t0 = VMADD(r2, 0xAA, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0xAA, r3, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 

  // carry propagation 
  // r0 is finally 49-bit not 48-bit
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r0 = VMADD(r0, 0xAA, r0, VSHUF(VSRA(r3, BRADIX), 0x4E)); r3 = VAND(r3, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
}

// a = < D' | D | C' | C | B' | B | A' | A >
// b = < H' | H | G' | G | F' | F | E' | E >
// r = < D'+H' | D+H | C'-G' | C+G | B'+F' | B+F | A'-E' | A-E >
static void asx2_fp_4x2w(fp_4x2w r, const fp_4x2w a, const fp_4x2w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask = VSET1(BMASK); 
  __m512i r0, r1, r2, r3, smask, t0, t1, t2, t3; 

  // r = D'+H' | D+H | C' | C | B'+F' | B+F | A' | A 
  r0 = VMADD(a0, 0xCC, a0, b0); r1 = VMADD(a1, 0xCC, a1, b1);
  r2 = VMADD(a2, 0xCC, a2, b2); r3 = VMADD(a3, 0xCC, a3, b3);

  // t = p' | p | G' | G | p' | p | E' | E
  t0 = VMBLEND(0xCC, b0, p0); t1 = VMBLEND(0xCC, b1, p1);
  t2 = VMBLEND(0xCC, b2, p2); t3 = VMBLEND(0xCC, b3, p3);

  // r = D'+H'-p | D+H-p | C'-G' | C-G | B'+F'-p' | B+F-p | A'-E' | A-E
  r0 = VSUB(r0, t0); r1 = VSUB(r1, t1);
  r2 = VSUB(r2, t2); r3 = VSUB(r3, t3);

  // get sign mask
  t0 = VMADD(r1, 0x55, r1, VSRA(r0, BRADIX));
  t0 = VMADD(r2, 0x55, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0x55, r3, VSRA(t0, BRADIX));
  t0 = VMADD(r0, 0xAA, r0, VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0 = VMADD(r1, 0xAA, r1, VSRA(t0, BRADIX));
  t0 = VMADD(r2, 0xAA, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0xAA, r3, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 

  // carry propagation 
  // r0 is finally 49-bit not 48-bit
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r0 = VMADD(r0, 0xAA, r0, VSHUF(VSRA(r3, BRADIX), 0x4E)); r3 = VAND(r3, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
}

// ----------------------------------------------------------------------------
// Fp double-length operations

static void add_fpx2_8x1w(fpx2_8x1w r, const fpx2_8x1w a, const fpx2_8x1w b)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  const __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  const __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15, smask, t0;

  // r = a + b
  r0  = VADD(a0 , b0 ); r1  = VADD(a1 , b1 );
  r2  = VADD(a2 , b2 ); r3  = VADD(a3 , b3 );
  r4  = VADD(a4 , b4 ); r5  = VADD(a5 , b5 );
  r6  = VADD(a6 , b6 ); r7  = VADD(a7 , b7 );
  r8  = VADD(a8 , b8 ); r9  = VADD(a9 , b9 );
  r10 = VADD(a10, b10); r11 = VADD(a11, b11);
  r12 = VADD(a12, b12); r13 = VADD(a13, b13);
  r14 = VADD(a14, b14); r15 = VADD(a15, b15);

  // r = r - p*2^384
  r8  = VSUB(r8 , p0); r9  = VSUB(r9 , p1);
  r10 = VSUB(r10, p2); r11 = VSUB(r11, p3);
  r12 = VSUB(r12, p4); r13 = VSUB(r13, p5);
  r14 = VSUB(r14, p6); r15 = VSUB(r15, p7);

  // get sign mask
  r1 = VADD(r1 , VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2 , VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3 , VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4 , VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5 , VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6 , VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7 , VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);
  r8 = VADD(r8 , VSRA(r7, BRADIX)); r7 = VAND(r7, bmask);
  t0 = VADD(r9 , VSRA(r8, BRADIX));
  t0 = VADD(r10, VSRA(t0, BRADIX));
  t0 = VADD(r11, VSRA(t0, BRADIX));
  t0 = VADD(r12, VSRA(t0, BRADIX));
  t0 = VADD(r13, VSRA(t0, BRADIX));
  t0 = VADD(r14, VSRA(t0, BRADIX));
  t0 = VADD(r15, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  // r = r + (p & smask)*2^384
  r8  = VADD(r8 , VAND(p0, smask)); r9  = VADD(r9 , VAND(p1, smask));
  r10 = VADD(r10, VAND(p2, smask)); r11 = VADD(r11, VAND(p3, smask));
  r12 = VADD(r12, VAND(p4, smask)); r13 = VADD(r13, VAND(p5, smask));
  r14 = VADD(r14, VAND(p6, smask)); r15 = VADD(r15, VAND(p7, smask));

  // carry propagation 
  r9  = VADD(r9 , VSRA(r8 , BRADIX)); r8  = VAND(r8 , bmask);
  r10 = VADD(r10, VSRA(r9 , BRADIX)); r9  = VAND(r9 , bmask);
  r11 = VADD(r11, VSRA(r10, BRADIX)); r10 = VAND(r10, bmask);
  r12 = VADD(r12, VSRA(r11, BRADIX)); r11 = VAND(r11, bmask);
  r13 = VADD(r13, VSRA(r12, BRADIX)); r12 = VAND(r12, bmask);
  r14 = VADD(r14, VSRA(r13, BRADIX)); r13 = VAND(r13, bmask);
  r15 = VADD(r15, VSRA(r14, BRADIX)); r14 = VAND(r14, bmask);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

static void sub_fpx2_8x1w(fpx2_8x1w r, const fpx2_8x1w a, const fpx2_8x1w b)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  const __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  const __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15, smask, t0;

  // r = a - b
  r0  = VSUB(a0 , b0 ); r1  = VSUB(a1 , b1 );
  r2  = VSUB(a2 , b2 ); r3  = VSUB(a3 , b3 );
  r4  = VSUB(a4 , b4 ); r5  = VSUB(a5 , b5 );
  r6  = VSUB(a6 , b6 ); r7  = VSUB(a7 , b7 );
  r8  = VSUB(a8 , b8 ); r9  = VSUB(a9 , b9 );
  r10 = VSUB(a10, b10); r11 = VSUB(a11, b11);
  r12 = VSUB(a12, b12); r13 = VSUB(a13, b13);
  r14 = VSUB(a14, b14); r15 = VSUB(a15, b15);

  // get sign mask
  r1 = VADD(r1 , VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2 , VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3 , VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4 , VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5 , VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6 , VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7 , VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);
  r8 = VADD(r8 , VSRA(r7, BRADIX)); r7 = VAND(r7, bmask);
  t0 = VADD(r9 , VSRA(r8, BRADIX));
  t0 = VADD(r10, VSRA(t0, BRADIX));
  t0 = VADD(r11, VSRA(t0, BRADIX));
  t0 = VADD(r12, VSRA(t0, BRADIX));
  t0 = VADD(r13, VSRA(t0, BRADIX));
  t0 = VADD(r14, VSRA(t0, BRADIX));
  t0 = VADD(r15, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  // r = r + (p & smask)*2^384
  r8  = VADD(r8 , VAND(p0, smask)); r9  = VADD(r9 , VAND(p1, smask));
  r10 = VADD(r10, VAND(p2, smask)); r11 = VADD(r11, VAND(p3, smask));
  r12 = VADD(r12, VAND(p4, smask)); r13 = VADD(r13, VAND(p5, smask));
  r14 = VADD(r14, VAND(p6, smask)); r15 = VADD(r15, VAND(p7, smask));

  // carry propagation 
  r9  = VADD(r9 , VSRA(r8 , BRADIX)); r8  = VAND(r8 , bmask);
  r10 = VADD(r10, VSRA(r9 , BRADIX)); r9  = VAND(r9 , bmask);
  r11 = VADD(r11, VSRA(r10, BRADIX)); r10 = VAND(r10, bmask);
  r12 = VADD(r12, VSRA(r11, BRADIX)); r11 = VAND(r11, bmask);
  r13 = VADD(r13, VSRA(r12, BRADIX)); r12 = VAND(r12, bmask);
  r14 = VADD(r14, VSRA(r13, BRADIX)); r13 = VAND(r13, bmask);
  r15 = VADD(r15, VSRA(r14, BRADIX)); r14 = VAND(r14, bmask);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

// a = < H | G | F | E | D | C | B | A >
// b = < P | O | N | M | L | K | J | I >
// r = < H+P | G-O | F+N | E-M | D+L | C-K | B+J | A-I >
static void asx4_fpx2_8x1w(fpx2_8x1w r, const fpx2_8x1w a, const fpx2_8x1w b)
{
  const __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  const __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  const __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  const __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  const __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2 , r3 , r4 , r5 , r6 , r7 ;
  __m512i r8, r9, r10, r11, r12, r13, r14, r15, smask;
  __m512i t0, t1, t2 , t3 , t4 , t5 , t6 , t7 ;

  // r = H+P | G | F+N | E | D+L | C | B+J | A
  r0  = VMADD(a0 , 0xAA, a0 , b0 ); r1  = VMADD(a1 , 0xAA, a1 , b1 );
  r2  = VMADD(a2 , 0xAA, a2 , b2 ); r3  = VMADD(a3 , 0xAA, a3 , b3 );
  r4  = VMADD(a4 , 0xAA, a4 , b4 ); r5  = VMADD(a5 , 0xAA, a5 , b5 );
  r6  = VMADD(a6 , 0xAA, a6 , b6 ); r7  = VMADD(a7 , 0xAA, a7 , b7 );
  r8  = VMADD(a8 , 0xAA, a8 , b8 ); r9  = VMADD(a9 , 0xAA, a9 , b9 );
  r10 = VMADD(a10, 0xAA, a10, b10); r11 = VMADD(a11, 0xAA, a11, b11);
  r12 = VMADD(a12, 0xAA, a12, b12); r13 = VMADD(a13, 0xAA, a13, b13);
  r14 = VMADD(a14, 0xAA, a14, b14); r15 = VMADD(a15, 0xAA, a15, b15);

  // t = p | O | p | M | p | K | p | I
  t0 = VMBLEND(0xAA, b8 , p0); t1 = VMBLEND(0xAA, b9 , p1);
  t2 = VMBLEND(0xAA, b10, p2); t3 = VMBLEND(0xAA, b11, p3);
  t4 = VMBLEND(0xAA, b12, p4); t5 = VMBLEND(0xAA, b13, p5);
  t6 = VMBLEND(0xAA, b14, p6); t7 = VMBLEND(0xAA, b15, p7); 

  // r = H+P | G-O | F+N | E-M | D+L | C-K | B+J | A-I
  r0 = VMSUB(r0, 0x55, r0, b0); r1 = VMSUB(r1, 0x55, r1, b1);
  r2 = VMSUB(r2, 0x55, r2, b2); r3 = VMSUB(r3, 0x55, r3, b3);
  r4 = VMSUB(r4, 0x55, r4, b4); r5 = VMSUB(r5, 0x55, r5, b5);
  r6 = VMSUB(r6, 0x55, r6, b6); r7 = VMSUB(r7, 0x55, r7, b7);
  // r = H+P-p | G-O | F+N-p | E-M | D+L-p | C-K | B+J-p | A-I
  r8  = VSUB(r8 , t0); r9  = VSUB(r9 , t1);
  r10 = VSUB(r10, t2); r11 = VSUB(r11, t3);
  r12 = VSUB(r12, t4); r13 = VSUB(r13, t5);
  r14 = VSUB(r14, t6); r15 = VSUB(r15, t7);

  // get sign mask
  r1 = VADD(r1 , VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2 , VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3 , VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4 , VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5 , VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6 , VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7 , VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);
  r8 = VADD(r8 , VSRA(r7, BRADIX)); r7 = VAND(r7, bmask);
  t0 = VADD(r9 , VSRA(r8, BRADIX));
  t0 = VADD(r10, VSRA(t0, BRADIX));
  t0 = VADD(r11, VSRA(t0, BRADIX));
  t0 = VADD(r12, VSRA(t0, BRADIX));
  t0 = VADD(r13, VSRA(t0, BRADIX));
  t0 = VADD(r14, VSRA(t0, BRADIX));
  t0 = VADD(r15, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  // r = r + (p & smask)*2^384
  r8  = VADD(r8 , VAND(p0, smask)); r9  = VADD(r9 , VAND(p1, smask));
  r10 = VADD(r10, VAND(p2, smask)); r11 = VADD(r11, VAND(p3, smask));
  r12 = VADD(r12, VAND(p4, smask)); r13 = VADD(r13, VAND(p5, smask));
  r14 = VADD(r14, VAND(p6, smask)); r15 = VADD(r15, VAND(p7, smask));

  // carry propagation 
  r9  = VADD(r9 , VSRA(r8 , BRADIX)); r8  = VAND(r8 , bmask);
  r10 = VADD(r10, VSRA(r9 , BRADIX)); r9  = VAND(r9 , bmask);
  r11 = VADD(r11, VSRA(r10, BRADIX)); r10 = VAND(r10, bmask);
  r12 = VADD(r12, VSRA(r11, BRADIX)); r11 = VAND(r11, bmask);
  r13 = VADD(r13, VSRA(r12, BRADIX)); r12 = VAND(r12, bmask);
  r14 = VADD(r14, VSRA(r13, BRADIX)); r13 = VAND(r13, bmask);
  r15 = VADD(r15, VSRA(r14, BRADIX)); r14 = VAND(r14, bmask);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
  r[12] = r12; r[13] = r13; r[14] = r14; r[15] = r15;
}

// Montgomery reduction
static void redc_fpx2_8x1w(fp_8x1w r, const fpx2_8x1w a)
{
  __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO;
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK), montw = VSET1(MONT_W_R48), zero = VZERO;
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask, u, t0;

  u  = VAND(VMACLO(zero, a0, montw), bmask);
  a0 = VMACLO(a0, u, p0); a1 = VMACLO(a1, u, p1); a2 = VMACLO(a2, u, p2);
  a3 = VMACLO(a3, u, p3); a4 = VMACLO(a4, u, p4); a5 = VMACLO(a5, u, p5);
  a6 = VMACLO(a6, u, p6); a7 = VMACLO(a7, u, p7); 
  y0 = VMACHI(y0, u, p0); y1 = VMACHI(y1, u, p1); y2 = VMACHI(y2, u, p2);
  y3 = VMACHI(y3, u, p3); y4 = VMACHI(y4, u, p4); y5 = VMACHI(y5, u, p5);
  y6 = VMACHI(y6, u, p6); y7 = VMACHI(y7, u, p7); 
  a1 = VADD(VADD(a1, VSRA(a0, BRADIX)), VSHL(y0, BALIGN));

  u  = VAND(VMACLO(zero, a1, montw), bmask);
  a1 = VMACLO(a1, u, p0); a2 = VMACLO(a2, u, p1); a3 = VMACLO(a3, u, p2);
  a4 = VMACLO(a4, u, p3); a5 = VMACLO(a5, u, p4); a6 = VMACLO(a6, u, p5);
  a7 = VMACLO(a7, u, p6); a8 = VMACLO(a8, u, p7); 
  y1 = VMACHI(y1, u, p0); y2 = VMACHI(y2, u, p1); y3 = VMACHI(y3, u, p2);
  y4 = VMACHI(y4, u, p3); y5 = VMACHI(y5, u, p4); y6 = VMACHI(y6, u, p5);
  y7 = VMACHI(y7, u, p6); y8 = VMACHI(y8, u, p7); 
  a2 = VADD(VADD(a2, VSRA(a1, BRADIX)), VSHL(y1, BALIGN));

  u  = VAND(VMACLO(zero, a2, montw), bmask);
  a2 = VMACLO(a2, u, p0); a3 = VMACLO(a3, u, p1); a4 = VMACLO(a4, u, p2);
  a5 = VMACLO(a5, u, p3); a6 = VMACLO(a6, u, p4); a7 = VMACLO(a7, u, p5);
  a8 = VMACLO(a8, u, p6); a9 = VMACLO(a9, u, p7); 
  y2 = VMACHI(y2, u, p0); y3 = VMACHI(y3, u, p1); y4 = VMACHI(y4, u, p2);
  y5 = VMACHI(y5, u, p3); y6 = VMACHI(y6, u, p4); y7 = VMACHI(y7, u, p5);
  y8 = VMACHI(y8, u, p6); y9 = VMACHI(y9, u, p7); 
  a3 = VADD(VADD(a3, VSRA(a2, BRADIX)), VSHL(y2, BALIGN));

  u  = VAND(VMACLO(zero, a3, montw), bmask);
  a3 = VMACLO(a3, u, p0); a4  = VMACLO(a4,  u, p1); a5 = VMACLO(a5, u, p2);
  a6 = VMACLO(a6, u, p3); a7  = VMACLO(a7,  u, p4); a8 = VMACLO(a8, u, p5);
  a9 = VMACLO(a9, u, p6); a10 = VMACLO(a10, u, p7); 
  y3 = VMACHI(y3, u, p0); y4  = VMACHI(y4 , u, p1); y5 = VMACHI(y5, u, p2);
  y6 = VMACHI(y6, u, p3); y7  = VMACHI(y7 , u, p4); y8 = VMACHI(y8, u, p5);
  y9 = VMACHI(y9, u, p6); y10 = VMACHI(y10, u, p7); 
  a4 = VADD(VADD(a4, VSRA(a3, BRADIX)), VSHL(y3, BALIGN));

  u   = VAND(VMACLO(zero, a4, montw), bmask);
  a4  = VMACLO(a4 , u, p0); a5  = VMACLO(a5 , u, p1); a6 = VMACLO(a6, u, p2);
  a7  = VMACLO(a7 , u, p3); a8  = VMACLO(a8 , u, p4); a9 = VMACLO(a9, u, p5);
  a10 = VMACLO(a10, u, p6); a11 = VMACLO(a11, u, p7); 
  y4  = VMACHI(y4 , u, p0); y5  = VMACHI(y5 , u, p1); y6 = VMACHI(y6, u, p2);
  y7  = VMACHI(y7 , u, p3); y8  = VMACHI(y8 , u, p4); y9 = VMACHI(y9, u, p5);
  y10 = VMACHI(y10, u, p6); y11 = VMACHI(y11, u, p7); 
  a5  = VADD(VADD(a5, VSRA(a4, BRADIX)), VSHL(y4, BALIGN));

  u   = VAND(VMACLO(zero, a5, montw), bmask);
  a5  = VMACLO(a5 , u, p0); a6  = VMACLO(a6 , u, p1); a7  = VMACLO(a7 , u, p2);
  a8  = VMACLO(a8 , u, p3); a9  = VMACLO(a9 , u, p4); a10 = VMACLO(a10, u, p5);
  a11 = VMACLO(a11, u, p6); a12 = VMACLO(a12, u, p7); 
  y5  = VMACHI(y5 , u, p0); y6  = VMACHI(y6 , u, p1); y7  = VMACHI(y7 , u, p2);
  y8  = VMACHI(y8 , u, p3); y9  = VMACHI(y9 , u, p4); y10 = VMACHI(y10, u, p5);
  y11 = VMACHI(y11, u, p6); y12 = VMACHI(y12, u, p7); 
  a6  = VADD(VADD(a6, VSRA(a5, BRADIX)), VSHL(y5, BALIGN));  

  u   = VAND(VMACLO(zero, a6, montw), bmask);
  a6  = VMACLO(a6 , u, p0); a7  = VMACLO(a7 , u, p1); a8  = VMACLO(a8 , u, p2);
  a9  = VMACLO(a9 , u, p3); a10 = VMACLO(a10, u, p4); a11 = VMACLO(a11, u, p5);
  a12 = VMACLO(a12, u, p6); a13 = VMACLO(a13, u, p7); 
  y6  = VMACHI(y6 , u, p0); y7  = VMACHI(y7 , u, p1); y8  = VMACHI(y8 , u, p2);
  y9  = VMACHI(y9 , u, p3); y10 = VMACHI(y10, u, p4); y11 = VMACHI(y11, u, p5);
  y12 = VMACHI(y12, u, p6); y13 = VMACHI(y13, u, p7); 
  a7  = VADD(VADD(a7, VSRA(a6, BRADIX)), VSHL(y6, BALIGN));  

  u   = VAND(VMACLO(zero, a7, montw), bmask);
  a7  = VMACLO(a7 , u, p0); a8  = VMACLO(a8 , u, p1); a9  = VMACLO(a9 , u, p2);
  a10 = VMACLO(a10, u, p3); a11 = VMACLO(a11, u, p4); a12 = VMACLO(a12, u, p5);
  a13 = VMACLO(a13, u, p6); a14 = VMACLO(a14, u, p7); 
  y7  = VMACHI(y7 , u, p0); y8  = VMACHI(y8 , u, p1); y9  = VMACHI(y9 , u, p2);
  y10 = VMACHI(y10, u, p3); y11 = VMACHI(y11, u, p4); y12 = VMACHI(y12, u, p5);
  y13 = VMACHI(y13, u, p6); y14 = VMACHI(y14, u, p7); 
  a8  = VADD(VADD(a8, VSRA(a7, BRADIX)), VSHL(y7, BALIGN)); 

  a9  = VADD(a9 , VSHL(y8 , BALIGN));
  a10 = VADD(a10, VSHL(y9 , BALIGN));
  a11 = VADD(a11, VSHL(y10, BALIGN));
  a12 = VADD(a12, VSHL(y11, BALIGN));
  a13 = VADD(a13, VSHL(y12, BALIGN));
  a14 = VADD(a14, VSHL(y13, BALIGN));
  a15 = VADD(a15, VSHL(y14, BALIGN));

  // final subtraction 
  r0 = a8 ; r1 = a9 ; r2 = a10; r3 = a11;
  r4 = a12; r5 = a13; r6 = a14; r7 = a15;

  // r = r - p
  r0 = VSUB(r0, p0); r1 = VSUB(r1, p1); r2 = VSUB(r2, p2); r3 = VSUB(r3, p3);
  r4 = VSUB(r4, p4); r5 = VSUB(r5, p5); r6 = VSUB(r6, p6); r7 = VSUB(r7, p7);

  // get sign mask 
  t0 = VADD(r1, VSRA(r0, BRADIX));
  t0 = VADD(r2, VSRA(t0, BRADIX));
  t0 = VADD(r3, VSRA(t0, BRADIX));
  t0 = VADD(r4, VSRA(t0, BRADIX));
  t0 = VADD(r5, VSRA(t0, BRADIX));
  t0 = VADD(r6, VSRA(t0, BRADIX));
  t0 = VADD(r7, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 
  r4 = VADD(r4, VAND(p4, smask)); r5 = VADD(r5, VAND(p5, smask)); 
  r6 = VADD(r6, VAND(p6, smask)); r7 = VADD(r7, VAND(p7, smask)); 

  // carry propagation 
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4, VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5, VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6, VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7, VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// Karatsuba (incl. carry prop.)
static void mul_fpx2_8x1w_v1(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i bmask = VSET1(BMASK);
  __m512i ta0, ta1, ta2, ta3;
  __m512i tb0, tb1, tb2, tb3;
  __m512i z0  = VZERO, z1  = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4  = VZERO, z5  = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8  = VZERO, z9  = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i z12 = VZERO, z13 = VZERO, z14 = VZERO, z15 = VZERO;
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO, y15 = VZERO;
  __m512i m0  = VZERO, m1  = VZERO, m2  = VZERO, m3  = VZERO;
  __m512i m4  = VZERO, m5  = VZERO, m6  = VZERO, m7  = VZERO;

  // compute zL(z0-z7) by aL(a0-a3) * bL(b0-b4)

  z0 = VMACLO(z0, a0, b0);
  y0 = VMACHI(y0, a0, b0);
  y0 = VSHL(y0, BALIGN);

  z1 = VMACLO(y0, a0, b1); z1 = VMACLO(z1, a1, b0);
  y1 = VMACHI(y1, a0, b1); y1 = VMACHI(y1, a1, b0);
  y1 = VSHL(y1, BALIGN);

  z2 = VMACLO(y1, a0, b2); z2 = VMACLO(z2, a1, b1); z2 = VMACLO(z2, a2, b0);
  y2 = VMACHI(y2, a0, b2); y2 = VMACHI(y2, a1, b1); y2 = VMACHI(y2, a2, b0);
  y2 = VSHL(y2, BALIGN);

  z3 = VMACLO(y2, a0, b3); z3 = VMACLO(z3, a1, b2); z3 = VMACLO(z3, a2, b1); 
  z3 = VMACLO(z3, a3, b0);
  y3 = VMACHI(y3, a0, b3); y3 = VMACHI(y3, a1, b2); y3 = VMACHI(y3, a2, b1); 
  y3 = VMACHI(y3, a3, b0);
  y3 = VSHL(y3, BALIGN);

  z4 = VMACLO(y3, a1, b3); z4 = VMACLO(z4, a2, b2); z4 = VMACLO(z4, a3, b1);
  y4 = VMACHI(y4, a1, b3); y4 = VMACHI(y4, a2, b2); y4 = VMACHI(y4, a3, b1);
  y4 = VSHL(y4, BALIGN);

  z5 = VMACLO(y4, a2, b3); z5 = VMACLO(z5, a3, b2);
  y5 = VMACHI(y5, a2, b3); y5 = VMACHI(y5, a3, b2);
  y5 = VSHL(y5, BALIGN);

  z6 = VMACLO(y5, a3, b3);
  y6 = VMACHI(y6, a3, b3);
  y6 = VSHL(y6, BALIGN);

  z7 = y6;

  // compute zH(z8-z15) by aH(a4-a7) * bH(b4-b7)

  z8 = VMACLO(z8, a4, b4);
  y8 = VMACHI(y8, a4, b4);
  y8 = VSHL(y8, BALIGN);

  z9 = VMACLO(y8, a4, b5); z9 = VMACLO(z9, a5, b4);
  y9 = VMACHI(y9, a4, b5); y9 = VMACHI(y9, a5, b4);
  y9 = VSHL(y9, BALIGN);

  z10 = VMACLO(y9,  a4, b6); z10 = VMACLO(z10, a5, b5); 
  z10 = VMACLO(z10, a6, b4);
  y10 = VMACHI(y10, a4, b6); y10 = VMACHI(y10, a5, b5); 
  y10 = VMACHI(y10, a6, b4);
  y10 = VSHL(y10, BALIGN);

  z11 = VMACLO(y10, a4, b7); z11 = VMACLO(z11, a5, b6);
  z11 = VMACLO(z11, a6, b5); z11 = VMACLO(z11, a7, b4);
  y11 = VMACHI(y11, a4, b7); y11 = VMACHI(y11, a5, b6);
  y11 = VMACHI(y11, a6, b5); y11 = VMACHI(y11, a7, b4);
  y11 = VSHL(y11, BALIGN);

  z12 = VMACLO(y11, a5, b7); z12 = VMACLO(z12, a6, b6);
  z12 = VMACLO(z12, a7, b5);
  y12 = VMACHI(y12, a5, b7); y12 = VMACHI(y12, a6, b6);
  y12 = VMACHI(y12, a7, b5);
  y12 = VSHL(y12, BALIGN);

  z13 = VMACLO(y12, a6, b7); z13 = VMACLO(z13, a7, b6);
  y13 = VMACHI(y13, a6, b7); y13 = VMACHI(y13, a7, b6);
  y13 = VSHL(y13, BALIGN);

  z14 = VMACLO(y13, a7, b7);
  y14 = VMACHI(y14, a7, b7);
  y14 = VSHL(y14, BALIGN);

  z15 = y14;

  // ta(ta0-ta3) = aL(a0-a3) + aH(a4-a7)
  ta0 = VADD(a0, a4); ta1 = VADD(a1, a5);
  ta2 = VADD(a2, a6); ta3 = VADD(a3, a7);

  // tb(tb0-tb3) = bL(b0-b3) + bH(b4-b7)
  tb0 = VADD(b0, b4); tb1 = VADD(b1, b5); 
  tb2 = VADD(b2, b6); tb3 = VADD(b3, b7);

  // zM = ta * tb - zL - zH 
  
  y0 = y1 = y2 = y3 = y4 = y5 = y6 = y7 = VZERO;

  m0 = VMACLO(m0, ta0, tb0);
  y0 = VMACHI(y0, ta0, tb0);
  y0 = VSHL(y0, BALIGN);

  m1 = VMACLO(y0, ta0, tb1); m1 = VMACLO(m1, ta1, tb0);
  y1 = VMACHI(y1, ta0, tb1); y1 = VMACHI(y1, ta1, tb0);
  y1 = VSHL(y1, BALIGN);

  m2 = VMACLO(y1, ta0, tb2); m2 = VMACLO(m2, ta1, tb1); 
  m2 = VMACLO(m2, ta2, tb0);
  y2 = VMACHI(y2, ta0, tb2); y2 = VMACHI(y2, ta1, tb1); 
  y2 = VMACHI(y2, ta2, tb0);
  y2 = VSHL(y2, BALIGN);

  m3 = VMACLO(y2, ta0, tb3); m3 = VMACLO(m3, ta1, tb2); 
  m3 = VMACLO(m3, ta2, tb1); m3 = VMACLO(m3, ta3, tb0);
  y3 = VMACHI(y3, ta0, tb3); y3 = VMACHI(y3, ta1, tb2); 
  y3 = VMACHI(y3, ta2, tb1); y3 = VMACHI(y3, ta3, tb0);
  y3 = VSHL(y3, BALIGN);

  m4 = VMACLO(y3, ta1, tb3); m4 = VMACLO(m4, ta2, tb2); 
  m4 = VMACLO(m4, ta3, tb1);
  y4 = VMACHI(y4, ta1, tb3); y4 = VMACHI(y4, ta2, tb2); 
  y4 = VMACHI(y4, ta3, tb1);
  y4 = VSHL(y4, BALIGN);

  m5 = VMACLO(y4, ta2, tb3); m5 = VMACLO(m5, ta3, tb2);
  y5 = VMACHI(y5, ta2, tb3); y5 = VMACHI(y5, ta3, tb2);
  y5 = VSHL(y5, BALIGN);

  m6 = VMACLO(y5, ta3, tb3);
  y6 = VMACHI(y6, ta3, tb3);
  y6 = VSHL(y6, BALIGN);

  m7 = y6;

  m0 = VSUB(m0, VADD(z0, z8 )); m1 = VSUB(m1, VADD(z1, z9 ));
  m2 = VSUB(m2, VADD(z2, z10)); m3 = VSUB(m3, VADD(z3, z11));
  m4 = VSUB(m4, VADD(z4, z12)); m5 = VSUB(m5, VADD(z5, z13));
  m6 = VSUB(m6, VADD(z6, z14)); m7 = VSUB(m7, VADD(z7, z15));

  // z = z + zM
  z4  = VADD(z4 , m0); z5  = VADD(z5 , m1);
  z6  = VADD(z6 , m2); z7  = VADD(z7 , m3);
  z8  = VADD(z8 , m4); z9  = VADD(z9 , m5);
  z10 = VADD(z10, m6); z11 = VADD(z11, m7);

  // carry propagation 
  z1  = VADD(z1,  VSRA(z0,  BRADIX)); z0  = VAND(z0,  bmask);
  z2  = VADD(z2,  VSRA(z1,  BRADIX)); z1  = VAND(z1,  bmask);
  z3  = VADD(z3,  VSRA(z2,  BRADIX)); z2  = VAND(z2,  bmask);
  z4  = VADD(z4,  VSRA(z3,  BRADIX)); z3  = VAND(z3,  bmask);
  z5  = VADD(z5,  VSRA(z4,  BRADIX)); z4  = VAND(z4,  bmask);
  z6  = VADD(z6,  VSRA(z5,  BRADIX)); z5  = VAND(z5,  bmask);
  z7  = VADD(z7,  VSRA(z6,  BRADIX)); z6  = VAND(z6,  bmask);
  z8  = VADD(z8,  VSRA(z7,  BRADIX)); z7  = VAND(z7,  bmask);
  z9  = VADD(z9,  VSRA(z8,  BRADIX)); z8  = VAND(z8,  bmask);
  z10 = VADD(z10, VSRA(z9,  BRADIX)); z9  = VAND(z9,  bmask);
  z11 = VADD(z11, VSRA(z10, BRADIX)); z10 = VAND(z10, bmask);
  z12 = VADD(z12, VSRA(z11, BRADIX)); z11 = VAND(z11, bmask);
  z13 = VADD(z13, VSRA(z12, BRADIX)); z12 = VAND(z12, bmask);
  z14 = VADD(z14, VSRA(z13, BRADIX)); z13 = VAND(z13, bmask);
  z15 = VADD(z15, VSRA(z14, BRADIX)); z14 = VAND(z14, bmask);

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
  r[12] = z12; r[13] = z13; r[14] = z14; r[15] = z15; 
}

// Karatsuba (excl. carry prop.)
static void mul_fpx2_8x1w_v2(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i bmask = VSET1(BMASK);
  __m512i ta0, ta1, ta2, ta3;
  __m512i tb0, tb1, tb2, tb3;
  __m512i z0  = VZERO, z1  = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4  = VZERO, z5  = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8  = VZERO, z9  = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i z12 = VZERO, z13 = VZERO, z14 = VZERO, z15 = VZERO;
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO, y15 = VZERO;
  __m512i m0  = VZERO, m1  = VZERO, m2  = VZERO, m3  = VZERO;
  __m512i m4  = VZERO, m5  = VZERO, m6  = VZERO, m7  = VZERO;

  // compute zL(z0-z7) by aL(a0-a3) * bL(b0-b4)

  z0 = VMACLO(z0, a0, b0);
  y0 = VMACHI(y0, a0, b0);
  y0 = VSHL(y0, BALIGN);

  z1 = VMACLO(y0, a0, b1); z1 = VMACLO(z1, a1, b0);
  y1 = VMACHI(y1, a0, b1); y1 = VMACHI(y1, a1, b0);
  y1 = VSHL(y1, BALIGN);

  z2 = VMACLO(y1, a0, b2); z2 = VMACLO(z2, a1, b1); z2 = VMACLO(z2, a2, b0);
  y2 = VMACHI(y2, a0, b2); y2 = VMACHI(y2, a1, b1); y2 = VMACHI(y2, a2, b0);
  y2 = VSHL(y2, BALIGN);

  z3 = VMACLO(y2, a0, b3); z3 = VMACLO(z3, a1, b2); z3 = VMACLO(z3, a2, b1); 
  z3 = VMACLO(z3, a3, b0);
  y3 = VMACHI(y3, a0, b3); y3 = VMACHI(y3, a1, b2); y3 = VMACHI(y3, a2, b1); 
  y3 = VMACHI(y3, a3, b0);
  y3 = VSHL(y3, BALIGN);

  z4 = VMACLO(y3, a1, b3); z4 = VMACLO(z4, a2, b2); z4 = VMACLO(z4, a3, b1);
  y4 = VMACHI(y4, a1, b3); y4 = VMACHI(y4, a2, b2); y4 = VMACHI(y4, a3, b1);
  y4 = VSHL(y4, BALIGN);

  z5 = VMACLO(y4, a2, b3); z5 = VMACLO(z5, a3, b2);
  y5 = VMACHI(y5, a2, b3); y5 = VMACHI(y5, a3, b2);
  y5 = VSHL(y5, BALIGN);

  z6 = VMACLO(y5, a3, b3);
  y6 = VMACHI(y6, a3, b3);
  y6 = VSHL(y6, BALIGN);

  z7 = y6;

  // compute zH(z8-z15) by aH(a4-a7) * bH(b4-b7)

  z8 = VMACLO(z8, a4, b4);
  y8 = VMACHI(y8, a4, b4);
  y8 = VSHL(y8, BALIGN);

  z9 = VMACLO(y8, a4, b5); z9 = VMACLO(z9, a5, b4);
  y9 = VMACHI(y9, a4, b5); y9 = VMACHI(y9, a5, b4);
  y9 = VSHL(y9, BALIGN);

  z10 = VMACLO(y9,  a4, b6); z10 = VMACLO(z10, a5, b5); 
  z10 = VMACLO(z10, a6, b4);
  y10 = VMACHI(y10, a4, b6); y10 = VMACHI(y10, a5, b5); 
  y10 = VMACHI(y10, a6, b4);
  y10 = VSHL(y10, BALIGN);

  z11 = VMACLO(y10, a4, b7); z11 = VMACLO(z11, a5, b6);
  z11 = VMACLO(z11, a6, b5); z11 = VMACLO(z11, a7, b4);
  y11 = VMACHI(y11, a4, b7); y11 = VMACHI(y11, a5, b6);
  y11 = VMACHI(y11, a6, b5); y11 = VMACHI(y11, a7, b4);
  y11 = VSHL(y11, BALIGN);

  z12 = VMACLO(y11, a5, b7); z12 = VMACLO(z12, a6, b6);
  z12 = VMACLO(z12, a7, b5);
  y12 = VMACHI(y12, a5, b7); y12 = VMACHI(y12, a6, b6);
  y12 = VMACHI(y12, a7, b5);
  y12 = VSHL(y12, BALIGN);

  z13 = VMACLO(y12, a6, b7); z13 = VMACLO(z13, a7, b6);
  y13 = VMACHI(y13, a6, b7); y13 = VMACHI(y13, a7, b6);
  y13 = VSHL(y13, BALIGN);

  z14 = VMACLO(y13, a7, b7);
  y14 = VMACHI(y14, a7, b7);
  y14 = VSHL(y14, BALIGN);

  z15 = y14;

  // ta(ta0-ta3) = aL(a0-a3) + aH(a4-a7)
  ta0 = VADD(a0, a4); ta1 = VADD(a1, a5);
  ta2 = VADD(a2, a6); ta3 = VADD(a3, a7);

  // tb(tb0-tb3) = bL(b0-b3) + bH(b4-b7)
  tb0 = VADD(b0, b4); tb1 = VADD(b1, b5); 
  tb2 = VADD(b2, b6); tb3 = VADD(b3, b7);

  // zM = ta * tb - zL - zH 
  
  y0 = y1 = y2 = y3 = y4 = y5 = y6 = y7 = VZERO;

  m0 = VMACLO(m0, ta0, tb0);
  y0 = VMACHI(y0, ta0, tb0);
  y0 = VSHL(y0, BALIGN);

  m1 = VMACLO(y0, ta0, tb1); m1 = VMACLO(m1, ta1, tb0);
  y1 = VMACHI(y1, ta0, tb1); y1 = VMACHI(y1, ta1, tb0);
  y1 = VSHL(y1, BALIGN);

  m2 = VMACLO(y1, ta0, tb2); m2 = VMACLO(m2, ta1, tb1); 
  m2 = VMACLO(m2, ta2, tb0);
  y2 = VMACHI(y2, ta0, tb2); y2 = VMACHI(y2, ta1, tb1); 
  y2 = VMACHI(y2, ta2, tb0);
  y2 = VSHL(y2, BALIGN);

  m3 = VMACLO(y2, ta0, tb3); m3 = VMACLO(m3, ta1, tb2); 
  m3 = VMACLO(m3, ta2, tb1); m3 = VMACLO(m3, ta3, tb0);
  y3 = VMACHI(y3, ta0, tb3); y3 = VMACHI(y3, ta1, tb2); 
  y3 = VMACHI(y3, ta2, tb1); y3 = VMACHI(y3, ta3, tb0);
  y3 = VSHL(y3, BALIGN);

  m4 = VMACLO(y3, ta1, tb3); m4 = VMACLO(m4, ta2, tb2); 
  m4 = VMACLO(m4, ta3, tb1);
  y4 = VMACHI(y4, ta1, tb3); y4 = VMACHI(y4, ta2, tb2); 
  y4 = VMACHI(y4, ta3, tb1);
  y4 = VSHL(y4, BALIGN);

  m5 = VMACLO(y4, ta2, tb3); m5 = VMACLO(m5, ta3, tb2);
  y5 = VMACHI(y5, ta2, tb3); y5 = VMACHI(y5, ta3, tb2);
  y5 = VSHL(y5, BALIGN);

  m6 = VMACLO(y5, ta3, tb3);
  y6 = VMACHI(y6, ta3, tb3);
  y6 = VSHL(y6, BALIGN);

  m7 = y6;

  m0 = VSUB(m0, VADD(z0, z8 )); m1 = VSUB(m1, VADD(z1, z9 ));
  m2 = VSUB(m2, VADD(z2, z10)); m3 = VSUB(m3, VADD(z3, z11));
  m4 = VSUB(m4, VADD(z4, z12)); m5 = VSUB(m5, VADD(z5, z13));
  m6 = VSUB(m6, VADD(z6, z14)); m7 = VSUB(m7, VADD(z7, z15));

  // z = z + zM
  z4  = VADD(z4 , m0); z5  = VADD(z5 , m1);
  z6  = VADD(z6 , m2); z7  = VADD(z7 , m3);
  z8  = VADD(z8 , m4); z9  = VADD(z9 , m5);
  z10 = VADD(z10, m6); z11 = VADD(z11, m7);

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
  r[12] = z12; r[13] = z13; r[14] = z14; r[15] = z15; 
}

// product-scanning (incl. carry prop.)
static void mul_fpx2_8x1w_v3(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i bmask = VSET1(BMASK);
  __m512i z0  = VZERO, z1  = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4  = VZERO, z5  = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8  = VZERO, z9  = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i z12 = VZERO, z13 = VZERO, z14 = VZERO, z15 = VZERO;
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO, y15 = VZERO;


  z0 = VMACLO(z0, a0, b0);
  y0 = VMACHI(y0, a0, b0);
  y0 = VSHL(y0, BALIGN);

  z1 = VMACLO(y0, a0, b1); z1 = VMACLO(z1, a1, b0);
  y1 = VMACHI(y1, a0, b1); y1 = VMACHI(y1, a1, b0);
  y1 = VSHL(y1, BALIGN);

  z2 = VMACLO(y1, a0, b2); z2 = VMACLO(z2, a1, b1); z2 = VMACLO(z2, a2, b0);
  y2 = VMACHI(y2, a0, b2); y2 = VMACHI(y2, a1, b1); y2 = VMACHI(y2, a2, b0);
  y2 = VSHL(y2, BALIGN);

  z3 = VMACLO(y2, a0, b3); z3 = VMACLO(z3, a1, b2); z3 = VMACLO(z3, a2, b1); 
  z3 = VMACLO(z3, a3, b0);
  y3 = VMACHI(y3, a0, b3); y3 = VMACHI(y3, a1, b2); y3 = VMACHI(y3, a2, b1); 
  y3 = VMACHI(y3, a3, b0);
  y3 = VSHL(y3, BALIGN);

  z4 = VMACLO(y3, a0, b4); z4 = VMACLO(z4, a1, b3); z4 = VMACLO(z4, a2, b2); 
  z4 = VMACLO(z4, a3, b1); z4 = VMACLO(z4, a4, b0);
  y4 = VMACHI(y4, a0, b4); y4 = VMACHI(y4, a1, b3); y4 = VMACHI(y4, a2, b2); 
  y4 = VMACHI(y4, a3, b1); y4 = VMACHI(y4, a4, b0);
  y4 = VSHL(y4, BALIGN);

  z5 = VMACLO(y4, a0, b5); z5 = VMACLO(z5, a1, b4); z5 = VMACLO(z5, a2, b3);
  z5 = VMACLO(z5, a3, b2); z5 = VMACLO(z5, a4, b1); z5 = VMACLO(z5, a5, b0);
  y5 = VMACHI(y5, a0, b5); y5 = VMACHI(y5, a1, b4); y5 = VMACHI(y5, a2, b3);
  y5 = VMACHI(y5, a3, b2); y5 = VMACHI(y5, a4, b1); y5 = VMACHI(y5, a5, b0);
  y5 = VSHL(y5, BALIGN);

  z6 = VMACLO(y5, a0, b6); z6 = VMACLO(z6, a1, b5); z6 = VMACLO(z6, a2, b4);
  z6 = VMACLO(z6, a3, b3); z6 = VMACLO(z6, a4, b2); z6 = VMACLO(z6, a5, b1);
  z6 = VMACLO(z6, a6, b0);
  y6 = VMACHI(y6, a0, b6); y6 = VMACHI(y6, a1, b5); y6 = VMACHI(y6, a2, b4);
  y6 = VMACHI(y6, a3, b3); y6 = VMACHI(y6, a4, b2); y6 = VMACHI(y6, a5, b1);
  y6 = VMACHI(y6, a6, b0);
  y6 = VSHL(y6, BALIGN);

  z7 = VMACLO(y6, a0, b7); z7 = VMACLO(z7, a1, b6); z7 = VMACLO(z7, a2, b5);
  z7 = VMACLO(z7, a3, b4); z7 = VMACLO(z7, a4, b3); z7 = VMACLO(z7, a5, b2);
  z7 = VMACLO(z7, a6, b1); z7 = VMACLO(z7, a7, b0);
  y7 = VMACHI(y7, a0, b7); y7 = VMACHI(y7, a1, b6); y7 = VMACHI(y7, a2, b5);
  y7 = VMACHI(y7, a3, b4); y7 = VMACHI(y7, a4, b3); y7 = VMACHI(y7, a5, b2);
  y7 = VMACHI(y7, a6, b1); y7 = VMACHI(y7, a7, b0);
  y7 = VSHL(y7, BALIGN);

  z8 = VMACLO(y7, a1, b7); z8 = VMACLO(z8, a2, b6); z8 = VMACLO(z8, a3, b5);
  z8 = VMACLO(z8, a4, b4); z8 = VMACLO(z8, a5, b3); z8 = VMACLO(z8, a6, b2);
  z8 = VMACLO(z8, a7, b1);
  y8 = VMACHI(y8, a1, b7); y8 = VMACHI(y8, a2, b6); y8 = VMACHI(y8, a3, b5);
  y8 = VMACHI(y8, a4, b4); y8 = VMACHI(y8, a5, b3); y8 = VMACHI(y8, a6, b2);
  y8 = VMACHI(y8, a7, b1);
  y8 = VSHL(y8, BALIGN);

  z9 = VMACLO(y8, a2, b7); z9 = VMACLO(z9, a3, b6); z9 = VMACLO(z9, a4, b5);
  z9 = VMACLO(z9, a5, b4); z9 = VMACLO(z9, a6, b3); z9 = VMACLO(z9, a7, b2);
  y9 = VMACHI(y9, a2, b7); y9 = VMACHI(y9, a3, b6); y9 = VMACHI(y9, a4, b5);
  y9 = VMACHI(y9, a5, b4); y9 = VMACHI(y9, a6, b3); y9 = VMACHI(y9, a7, b2);
  y9 = VSHL(y9, BALIGN);

  z10 = VMACLO(y9,  a3, b7); z10 = VMACLO(z10, a4, b6);
  z10 = VMACLO(z10, a5, b5); z10 = VMACLO(z10, a6, b4);
  z10 = VMACLO(z10, a7, b3);
  y10 = VMACHI(y10, a3, b7); y10 = VMACHI(y10, a4, b6);
  y10 = VMACHI(y10, a5, b5); y10 = VMACHI(y10, a6, b4);
  y10 = VMACHI(y10, a7, b3);
  y10 = VSHL(y10, BALIGN);

  z11 = VMACLO(y10, a4, b7); z11 = VMACLO(z11, a5, b6);
  z11 = VMACLO(z11, a6, b5); z11 = VMACLO(z11, a7, b4);
  y11 = VMACHI(y11, a4, b7); y11 = VMACHI(y11, a5, b6);
  y11 = VMACHI(y11, a6, b5); y11 = VMACHI(y11, a7, b4);
  y11 = VSHL(y11, BALIGN);

  z12 = VMACLO(y11, a5, b7); z12 = VMACLO(z12, a6, b6);
  z12 = VMACLO(z12, a7, b5);
  y12 = VMACHI(y12, a5, b7); y12 = VMACHI(y12, a6, b6);
  y12 = VMACHI(y12, a7, b5);
  y12 = VSHL(y12, BALIGN);

  z13 = VMACLO(y12, a6, b7); z13 = VMACLO(z13, a7, b6);
  y13 = VMACHI(y13, a6, b7); y13 = VMACHI(y13, a7, b6);
  y13 = VSHL(y13, BALIGN);

  z14 = VMACLO(y13, a7, b7);
  y14 = VMACHI(y14, a7, b7);
  y14 = VSHL(y14, BALIGN);

  z15 = y14;

  // carry propagation 
  z1  = VADD(z1,  VSRA(z0,  BRADIX)); z0  = VAND(z0,  bmask);
  z2  = VADD(z2,  VSRA(z1,  BRADIX)); z1  = VAND(z1,  bmask);
  z3  = VADD(z3,  VSRA(z2,  BRADIX)); z2  = VAND(z2,  bmask);
  z4  = VADD(z4,  VSRA(z3,  BRADIX)); z3  = VAND(z3,  bmask);
  z5  = VADD(z5,  VSRA(z4,  BRADIX)); z4  = VAND(z4,  bmask);
  z6  = VADD(z6,  VSRA(z5,  BRADIX)); z5  = VAND(z5,  bmask);
  z7  = VADD(z7,  VSRA(z6,  BRADIX)); z6  = VAND(z6,  bmask);
  z8  = VADD(z8,  VSRA(z7,  BRADIX)); z7  = VAND(z7,  bmask);
  z9  = VADD(z9,  VSRA(z8,  BRADIX)); z8  = VAND(z8,  bmask);
  z10 = VADD(z10, VSRA(z9,  BRADIX)); z9  = VAND(z9,  bmask);
  z11 = VADD(z11, VSRA(z10, BRADIX)); z10 = VAND(z10, bmask);
  z12 = VADD(z12, VSRA(z11, BRADIX)); z11 = VAND(z11, bmask);
  z13 = VADD(z13, VSRA(z12, BRADIX)); z12 = VAND(z12, bmask);
  z14 = VADD(z14, VSRA(z13, BRADIX)); z13 = VAND(z13, bmask);
  z15 = VADD(z15, VSRA(z14, BRADIX)); z14 = VAND(z14, bmask);


  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
  r[12] = z12; r[13] = z13; r[14] = z14; r[15] = z15; 
}

// product-scanning (excl. carry prop.)
static void mul_fpx2_8x1w_v4(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i bmask = VSET1(BMASK);
  __m512i z0  = VZERO, z1  = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4  = VZERO, z5  = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8  = VZERO, z9  = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i z12 = VZERO, z13 = VZERO, z14 = VZERO, z15 = VZERO;
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO, y11 = VZERO;
  __m512i y12 = VZERO, y13 = VZERO, y14 = VZERO, y15 = VZERO;


  z0 = VMACLO(z0, a0, b0);
  y0 = VMACHI(y0, a0, b0);
  y0 = VSHL(y0, BALIGN);

  z1 = VMACLO(y0, a0, b1); z1 = VMACLO(z1, a1, b0);
  y1 = VMACHI(y1, a0, b1); y1 = VMACHI(y1, a1, b0);
  y1 = VSHL(y1, BALIGN);

  z2 = VMACLO(y1, a0, b2); z2 = VMACLO(z2, a1, b1); z2 = VMACLO(z2, a2, b0);
  y2 = VMACHI(y2, a0, b2); y2 = VMACHI(y2, a1, b1); y2 = VMACHI(y2, a2, b0);
  y2 = VSHL(y2, BALIGN);

  z3 = VMACLO(y2, a0, b3); z3 = VMACLO(z3, a1, b2); z3 = VMACLO(z3, a2, b1); 
  z3 = VMACLO(z3, a3, b0);
  y3 = VMACHI(y3, a0, b3); y3 = VMACHI(y3, a1, b2); y3 = VMACHI(y3, a2, b1); 
  y3 = VMACHI(y3, a3, b0);
  y3 = VSHL(y3, BALIGN);

  z4 = VMACLO(y3, a0, b4); z4 = VMACLO(z4, a1, b3); z4 = VMACLO(z4, a2, b2); 
  z4 = VMACLO(z4, a3, b1); z4 = VMACLO(z4, a4, b0);
  y4 = VMACHI(y4, a0, b4); y4 = VMACHI(y4, a1, b3); y4 = VMACHI(y4, a2, b2); 
  y4 = VMACHI(y4, a3, b1); y4 = VMACHI(y4, a4, b0);
  y4 = VSHL(y4, BALIGN);

  z5 = VMACLO(y4, a0, b5); z5 = VMACLO(z5, a1, b4); z5 = VMACLO(z5, a2, b3);
  z5 = VMACLO(z5, a3, b2); z5 = VMACLO(z5, a4, b1); z5 = VMACLO(z5, a5, b0);
  y5 = VMACHI(y5, a0, b5); y5 = VMACHI(y5, a1, b4); y5 = VMACHI(y5, a2, b3);
  y5 = VMACHI(y5, a3, b2); y5 = VMACHI(y5, a4, b1); y5 = VMACHI(y5, a5, b0);
  y5 = VSHL(y5, BALIGN);

  z6 = VMACLO(y5, a0, b6); z6 = VMACLO(z6, a1, b5); z6 = VMACLO(z6, a2, b4);
  z6 = VMACLO(z6, a3, b3); z6 = VMACLO(z6, a4, b2); z6 = VMACLO(z6, a5, b1);
  z6 = VMACLO(z6, a6, b0);
  y6 = VMACHI(y6, a0, b6); y6 = VMACHI(y6, a1, b5); y6 = VMACHI(y6, a2, b4);
  y6 = VMACHI(y6, a3, b3); y6 = VMACHI(y6, a4, b2); y6 = VMACHI(y6, a5, b1);
  y6 = VMACHI(y6, a6, b0);
  y6 = VSHL(y6, BALIGN);

  z7 = VMACLO(y6, a0, b7); z7 = VMACLO(z7, a1, b6); z7 = VMACLO(z7, a2, b5);
  z7 = VMACLO(z7, a3, b4); z7 = VMACLO(z7, a4, b3); z7 = VMACLO(z7, a5, b2);
  z7 = VMACLO(z7, a6, b1); z7 = VMACLO(z7, a7, b0);
  y7 = VMACHI(y7, a0, b7); y7 = VMACHI(y7, a1, b6); y7 = VMACHI(y7, a2, b5);
  y7 = VMACHI(y7, a3, b4); y7 = VMACHI(y7, a4, b3); y7 = VMACHI(y7, a5, b2);
  y7 = VMACHI(y7, a6, b1); y7 = VMACHI(y7, a7, b0);
  y7 = VSHL(y7, BALIGN);

  z8 = VMACLO(y7, a1, b7); z8 = VMACLO(z8, a2, b6); z8 = VMACLO(z8, a3, b5);
  z8 = VMACLO(z8, a4, b4); z8 = VMACLO(z8, a5, b3); z8 = VMACLO(z8, a6, b2);
  z8 = VMACLO(z8, a7, b1);
  y8 = VMACHI(y8, a1, b7); y8 = VMACHI(y8, a2, b6); y8 = VMACHI(y8, a3, b5);
  y8 = VMACHI(y8, a4, b4); y8 = VMACHI(y8, a5, b3); y8 = VMACHI(y8, a6, b2);
  y8 = VMACHI(y8, a7, b1);
  y8 = VSHL(y8, BALIGN);

  z9 = VMACLO(y8, a2, b7); z9 = VMACLO(z9, a3, b6); z9 = VMACLO(z9, a4, b5);
  z9 = VMACLO(z9, a5, b4); z9 = VMACLO(z9, a6, b3); z9 = VMACLO(z9, a7, b2);
  y9 = VMACHI(y9, a2, b7); y9 = VMACHI(y9, a3, b6); y9 = VMACHI(y9, a4, b5);
  y9 = VMACHI(y9, a5, b4); y9 = VMACHI(y9, a6, b3); y9 = VMACHI(y9, a7, b2);
  y9 = VSHL(y9, BALIGN);

  z10 = VMACLO(y9,  a3, b7); z10 = VMACLO(z10, a4, b6);
  z10 = VMACLO(z10, a5, b5); z10 = VMACLO(z10, a6, b4);
  z10 = VMACLO(z10, a7, b3);
  y10 = VMACHI(y10, a3, b7); y10 = VMACHI(y10, a4, b6);
  y10 = VMACHI(y10, a5, b5); y10 = VMACHI(y10, a6, b4);
  y10 = VMACHI(y10, a7, b3);
  y10 = VSHL(y10, BALIGN);

  z11 = VMACLO(y10, a4, b7); z11 = VMACLO(z11, a5, b6);
  z11 = VMACLO(z11, a6, b5); z11 = VMACLO(z11, a7, b4);
  y11 = VMACHI(y11, a4, b7); y11 = VMACHI(y11, a5, b6);
  y11 = VMACHI(y11, a6, b5); y11 = VMACHI(y11, a7, b4);
  y11 = VSHL(y11, BALIGN);

  z12 = VMACLO(y11, a5, b7); z12 = VMACLO(z12, a6, b6);
  z12 = VMACLO(z12, a7, b5);
  y12 = VMACHI(y12, a5, b7); y12 = VMACHI(y12, a6, b6);
  y12 = VMACHI(y12, a7, b5);
  y12 = VSHL(y12, BALIGN);

  z13 = VMACLO(y12, a6, b7); z13 = VMACLO(z13, a7, b6);
  y13 = VMACHI(y13, a6, b7); y13 = VMACHI(y13, a7, b6);
  y13 = VSHL(y13, BALIGN);

  z14 = VMACLO(y13, a7, b7);
  y14 = VMACHI(y14, a7, b7);
  y14 = VSHL(y14, BALIGN);

  z15 = y14;

  // carry propagation 
  z1  = VADD(z1,  VSRA(z0,  BRADIX)); z0  = VAND(z0,  bmask);
  z2  = VADD(z2,  VSRA(z1,  BRADIX)); z1  = VAND(z1,  bmask);
  z3  = VADD(z3,  VSRA(z2,  BRADIX)); z2  = VAND(z2,  bmask);
  z4  = VADD(z4,  VSRA(z3,  BRADIX)); z3  = VAND(z3,  bmask);
  z5  = VADD(z5,  VSRA(z4,  BRADIX)); z4  = VAND(z4,  bmask);
  z6  = VADD(z6,  VSRA(z5,  BRADIX)); z5  = VAND(z5,  bmask);
  z7  = VADD(z7,  VSRA(z6,  BRADIX)); z6  = VAND(z6,  bmask);
  z8  = VADD(z8,  VSRA(z7,  BRADIX)); z7  = VAND(z7,  bmask);
  z9  = VADD(z9,  VSRA(z8,  BRADIX)); z8  = VAND(z8,  bmask);
  z10 = VADD(z10, VSRA(z9,  BRADIX)); z9  = VAND(z9,  bmask);
  z11 = VADD(z11, VSRA(z10, BRADIX)); z10 = VAND(z10, bmask);
  z12 = VADD(z12, VSRA(z11, BRADIX)); z11 = VAND(z11, bmask);
  z13 = VADD(z13, VSRA(z12, BRADIX)); z12 = VAND(z12, bmask);
  z14 = VADD(z14, VSRA(z13, BRADIX)); z13 = VAND(z13, bmask);
  z15 = VADD(z15, VSRA(z14, BRADIX)); z14 = VAND(z14, bmask);


  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
  r[12] = z12; r[13] = z13; r[14] = z14; r[15] = z15; 
}

// operand-scanning (excl. carry prop.)
static void mul_fpx2_4x2w_v1(fpx2_4x2w r, const fp_4x2w a, const fp_4x2w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  __m512i z0 = VZERO, z1 = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4 = VZERO, z5 = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8 = VZERO, z9 = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i y0 = VZERO, y1 = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4 = VZERO, y5 = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8 = VZERO, y9 = VZERO, y10 = VZERO, tb;

  tb = VSHUF(b0, 0x44);
  z0 = VMACLO(z0, tb, a0); z1 = VMACLO(z1, tb, a1);
  z2 = VMACLO(z2, tb, a2); z3 = VMACLO(z3, tb, a3);
  y0 = VMACHI(y0, tb, a0); y1 = VMACHI(y1, tb, a1);
  y2 = VMACHI(y2, tb, a2); y3 = VMACHI(y3, tb, a3);

  tb = VSHUF(b1, 0x44);
  z1 = VMACLO(z1, tb, a0); z2 = VMACLO(z2, tb, a1);
  z3 = VMACLO(z3, tb, a2); z4 = VMACLO(z4, tb, a3);
  y1 = VMACHI(y1, tb, a0); y2 = VMACHI(y2, tb, a1);
  y3 = VMACHI(y3, tb, a2); y4 = VMACHI(y4, tb, a3);

  tb = VSHUF(b2, 0x44);
  z2 = VMACLO(z2, tb, a0); z3 = VMACLO(z3, tb, a1);
  z4 = VMACLO(z4, tb, a2); z5 = VMACLO(z5, tb, a3);
  y2 = VMACHI(y2, tb, a0); y3 = VMACHI(y3, tb, a1);
  y4 = VMACHI(y4, tb, a2); y5 = VMACHI(y5, tb, a3);

  tb = VSHUF(b3, 0x44);
  z3 = VMACLO(z3, tb, a0); z4 = VMACLO(z4, tb, a1);
  z5 = VMACLO(z5, tb, a2); z6 = VMACLO(z6, tb, a3);
  y3 = VMACHI(y3, tb, a0); y4 = VMACHI(y4, tb, a1);
  y5 = VMACHI(y5, tb, a2); y6 = VMACHI(y6, tb, a3);

  tb = VSHUF(b0, 0xEE);
  z4 = VMACLO(z4, tb, a0); z5 = VMACLO(z5, tb, a1);
  z6 = VMACLO(z6, tb, a2); z7 = VMACLO(z7, tb, a3);
  y4 = VMACHI(y4, tb, a0); y5 = VMACHI(y5, tb, a1);
  y6 = VMACHI(y6, tb, a2); y7 = VMACHI(y7, tb, a3);

  tb = VSHUF(b1, 0xEE);
  z5 = VMACLO(z5, tb, a0); z6 = VMACLO(z6, tb, a1);
  z7 = VMACLO(z7, tb, a2); z8 = VMACLO(z8, tb, a3);
  y5 = VMACHI(y5, tb, a0); y6 = VMACHI(y6, tb, a1);
  y7 = VMACHI(y7, tb, a2); y8 = VMACHI(y8, tb, a3);

  tb = VSHUF(b2, 0xEE);
  z6 = VMACLO(z6, tb, a0); z7 = VMACLO(z7, tb, a1);
  z8 = VMACLO(z8, tb, a2); z9 = VMACLO(z9, tb, a3);
  y6 = VMACHI(y6, tb, a0); y7 = VMACHI(y7, tb, a1);
  y8 = VMACHI(y8, tb, a2); y9 = VMACHI(y9, tb, a3);

  tb = VSHUF(b3, 0xEE);
  z7 = VMACLO(z7, tb, a0); z8 = VMACLO(z8, tb, a1);
  z9 = VMACLO(z9, tb, a2); z10 = VMACLO(z10, tb, a3);
  y7 = VMACHI(y7, tb, a0); y8 = VMACHI(y8, tb, a1);
  y9 = VMACHI(y9, tb, a2); y10 = VMACHI(y10, tb, a3);

  z1  = VADD(z1 , VSHL(y0 , BALIGN));
  z2  = VADD(z2 , VSHL(y1 , BALIGN));
  z3  = VADD(z3 , VSHL(y2 , BALIGN));
  z4  = VADD(z4 , VSHL(y3 , BALIGN));
  z5  = VADD(z5 , VSHL(y4 , BALIGN));
  z6  = VADD(z6 , VSHL(y5 , BALIGN));
  z7  = VADD(z7 , VSHL(y6 , BALIGN));
  z8  = VADD(z8 , VSHL(y7 , BALIGN));
  z9  = VADD(z9 , VSHL(y8 , BALIGN));
  z10 = VADD(z10, VSHL(y9 , BALIGN));
  z11 = VADD(z11, VSHL(y10, BALIGN));

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
}

static void redc_fpx2_4x2w(fp_4x2w r, const fpx2_4x2w a)
{
  __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  __m512i y0  = VZERO, y1  = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4  = VZERO, y5  = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8  = VZERO, y9  = VZERO, y10 = VZERO;
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask = VSET1(BMASK), montw = VSET1(MONT_W_R48), zero = VZERO;
  __m512i r0, r1, r2, r3, smask, u, t0;

  u  = VSHUF(VAND(VMACLO(zero, a0, montw), bmask), 0x44);
  a0 = VMACLO(a0, u, p0); a1 = VMACLO(a1, u, p1);
  a2 = VMACLO(a2, u, p2); a3 = VMACLO(a3, u, p3);
  y0 = VMACHI(y0, u, p0); y1 = VMACHI(y1, u, p1);
  y2 = VMACHI(y2, u, p2); y3 = VMACHI(y3, u, p3);
  a4 = VMADD(a4, 0x55, a4, VSHUF(a0, 0x4E  ));
  a1 = VMADD(a1, 0x55, a1, VSRA (a0, BRADIX));
  a1 = VADD(a1, VSHL(y0, BALIGN));

  u  = VSHUF(VAND(VMACLO(zero, a1, montw), bmask), 0x44);
  a1 = VMACLO(a1, u, p0); a2 = VMACLO(a2, u, p1);
  a3 = VMACLO(a3, u, p2); a4 = VMACLO(a4, u, p3);
  y1 = VMACHI(y1, u, p0); y2 = VMACHI(y2, u, p1);
  y3 = VMACHI(y3, u, p2); y4 = VMACHI(y4, u, p3);
  a5 = VMADD(a5, 0x55, a5, VSHUF(a1, 0x4E  ));
  a2 = VMADD(a2, 0x55, a2, VSRA (a1, BRADIX));
  a2 = VADD(a2, VSHL(y1, BALIGN));

  u  = VSHUF(VAND(VMACLO(zero, a2, montw), bmask), 0x44);
  a2 = VMACLO(a2, u, p0); a3 = VMACLO(a3, u, p1);
  a4 = VMACLO(a4, u, p2); a5 = VMACLO(a5, u, p3);
  y2 = VMACHI(y2, u, p0); y3 = VMACHI(y3, u, p1);
  y4 = VMACHI(y4, u, p2); y5 = VMACHI(y5, u, p3);
  a6 = VMADD(a6, 0x55, a6, VSHUF(a2, 0x4E  ));
  a3 = VMADD(a3, 0x55, a3, VSRA (a2, BRADIX));
  a3 = VADD(a3, VSHL(y2, BALIGN));

  u  = VSHUF(VAND(VMACLO(zero, a3, montw), bmask), 0x44);
  a3 = VMACLO(a3, u, p0); a4 = VMACLO(a4, u, p1);
  a5 = VMACLO(a5, u, p2); a6 = VMACLO(a6, u, p3);
  y3 = VMACHI(y3, u, p0); y4 = VMACHI(y4, u, p1);
  y5 = VMACHI(y5, u, p2); y6 = VMACHI(y6, u, p3);
  a7 = VMADD(a7, 0x55, a7, VSHUF(a3, 0x4E  ));
  a4 = VMADD(a4, 0x55, a4, VSRA (a3, BRADIX));
  a4 = VADD(a4, VSHL(y3, BALIGN));

  u  = VSHUF(VAND(VMACLO(zero, a4, montw), bmask), 0x44);
  a4 = VMACLO(a4, u, p0); a5 = VMACLO(a5, u, p1);
  a6 = VMACLO(a6, u, p2); a7 = VMACLO(a7, u, p3);
  y4 = VMACHI(y4, u, p0); y5 = VMACHI(y5, u, p1);
  y6 = VMACHI(y6, u, p2); y7 = VMACHI(y7, u, p3);
  a8 = VMADD(a8, 0x55, a8, VSHUF(a4, 0x4E  ));
  a5 = VMADD(a5, 0x55, a5, VSRA (a4, BRADIX));
  a5 = VADD(a5, VSHL(y4, BALIGN));

  u  = VSHUF(VAND(VMACLO(zero, a5, montw), bmask), 0x44);
  a5 = VMACLO(a5, u, p0); a6 = VMACLO(a6, u, p1);
  a7 = VMACLO(a7, u, p2); a8 = VMACLO(a8, u, p3);
  y5 = VMACHI(y5, u, p0); y6 = VMACHI(y6, u, p1);
  y7 = VMACHI(y7, u, p2); y8 = VMACHI(y8, u, p3);
  a9 = VMADD(a9, 0x55, a9, VSHUF(a5, 0x4E  ));
  a6 = VMADD(a6, 0x55, a6, VSRA (a5, BRADIX));
  a6 = VADD(a6, VSHL(y5, BALIGN));

  u   = VSHUF(VAND(VMACLO(zero, a6, montw), bmask), 0x44);
  a6  = VMACLO(a6, u, p0); a7  = VMACLO(a7, u, p1);
  a8  = VMACLO(a8, u, p2); a9  = VMACLO(a9, u, p3);
  y6  = VMACHI(y6, u, p0); y7  = VMACHI(y7, u, p1);
  y8  = VMACHI(y8, u, p2); y9  = VMACHI(y9, u, p3);
  a10 = VMADD(a10, 0x55, a10, VSHUF(a6, 0x4E  ));
  a7  = VMADD(a7 , 0x55, a7 , VSRA (a6, BRADIX));
  a7  = VADD(a7, VSHL(y6, BALIGN));

  u   = VSHUF(VAND(VMACLO(zero, a7, montw), bmask), 0x44);
  a7  = VMACLO(a7, u, p0); a8   = VMACLO(a8 , u, p1);
  a9  = VMACLO(a9, u, p2); a10  = VMACLO(a10, u, p3);
  y7  = VMACHI(y7, u, p0); y8   = VMACHI(y8 , u, p1);
  y9  = VMACHI(y9, u, p2); y10  = VMACHI(y10, u, p3);
  a11 = VMADD(a11, 0x55, a11, VSHUF(a7, 0x4E  ));
  a8  = VMADD(a8 , 0x55, a8 , VSRA (a7, BRADIX));
  a8  = VADD(a8, VSHL(y7, BALIGN));

  a9  = VADD(a9 , VSHL(y8 , BALIGN));
  a10 = VADD(a10, VSHL(y9 , BALIGN));
  a11 = VADD(a11, VSHL(y10, BALIGN));

  // final subtraction
  r0 = a8; r1 = a9; r2 = a10; r3 = a11;

  // r = r - p
  r0 = VSUB(r0, p0); r1 = VSUB(r1, p1); r2 = VSUB(r2, p2); r3 = VSUB(r3, p3);

  // get sign mask
  t0 = VMADD(r1, 0x55, r1, VSRA(r0, BRADIX));
  t0 = VMADD(r2, 0x55, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0x55, r3, VSRA(t0, BRADIX));
  t0 = VMADD(r0, 0xAA, r0, VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0 = VMADD(r1, 0xAA, r1, VSRA(t0, BRADIX));
  t0 = VMADD(r2, 0xAA, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0xAA, r3, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 

  // carry propagation 
  // r0 is finally 49-bit not 48-bit
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r0 = VMADD(r0, 0xAA, r0, VSHUF(VSRA(r3, BRADIX), 0x4E)); r3 = VAND(r3, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
}

// a = < D' | D | C' | C | B' | B | A' | A >
// b = < H' | H | G' | G | F' | F | E' | E >
// r = < D'+H' | D+H | C'-G' | C+G | B'+F' | B+F | A'-E' | A-E >
static void asx2_fpx2_4x2w(fpx2_4x2w r, const fpx2_4x2w a, const fpx2_4x2w b)
{
  __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask0 = VSET(0, BMASK, 0, BMASK, 0, BMASK, 0, BMASK);
  const __m512i bmask1 = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11; 
  __m512i t0, t1, t2, t3, smask;

  // r = D'+H' | D+H | C' | C | B'+F' | B+F | A' | A 
  r0  = VMADD(a0 , 0xCC, a0 , b0 ); r1  = VMADD(a1 , 0xCC, a1 , b1 );
  r2  = VMADD(a2 , 0xCC, a2 , b2 ); r3  = VMADD(a3 , 0xCC, a3 , b3 );
  r4  = VMADD(a4 , 0xCC, a4 , b4 ); r5  = VMADD(a5 , 0xCC, a5 , b5 );
  r6  = VMADD(a6 , 0xCC, a6 , b6 ); r7  = VMADD(a7 , 0xCC, a7 , b7 );
  r8  = VMADD(a8 , 0xCC, a8 , b8 ); r9  = VMADD(a9 , 0xCC, a9 , b9 );
  r10 = VMADD(a10, 0xCC, a10, b10); r11 = VMADD(a11, 0xCC, a11, b11);

  // t = p' | p | G' | G | p' | p | E' | E
  t0 = VMBLEND(0xCC, b8 , p0); t1 = VMBLEND(0xCC, b9 , p1);
  t2 = VMBLEND(0xCC, b10, p2); t3 = VMBLEND(0xCC, b11, p3);

  // r = D'+H' | D+H | C'-G' | C-G | B'+F' | B+F | A'-E' | A-E
  r0 = VMSUB(r0, 0x33, r0, b0); r1 = VMSUB(r1, 0x33, r1, b1);
  r2 = VMSUB(r2, 0x33, r2, b2); r3 = VMSUB(r3, 0x33, r3, b3);
  r4 = VMSUB(r4, 0x33, r4, b4); r5 = VMSUB(r5, 0x33, r5, b5);
  r6 = VMSUB(r6, 0x33, r6, b6); r7 = VMSUB(r7, 0x33, r7, b7);
  // r = D'+H'-p | D+H-p | C'-G' | C-G | B'+F'-p' | B+F-p | A'-E' | A-E
  r8  = VSUB(r8 , t0); r9  = VSUB(r9 , t1);
  r10 = VSUB(r10, t2); r11 = VSUB(r11, t3);

  // get sign mask + carry propagation
  r1  = VMADD(r1 , 0x55, r1 , VSRA(r0, BRADIX)); 
  r4  = VMADD(r4 , 0x55, r4 , VSHUF(r0, 0x4E)); r0 = VAND(r0, bmask0);
  r2  = VMADD(r2 , 0x55, r2 , VSRA(r1, BRADIX)); 
  r5  = VMADD(r5 , 0x55, r5 , VSHUF(r1, 0x4E)); r1 = VAND(r1, bmask0);
  r3  = VMADD(r3 , 0x55, r3 , VSRA(r2, BRADIX)); 
  r6  = VMADD(r6 , 0x55, r6 , VSHUF(r2, 0x4E)); r2 = VAND(r2, bmask0);
  r4  = VMADD(r4 , 0x55, r4 , VSRA(r3, BRADIX)); 
  r7  = VMADD(r7 , 0x55, r7 , VSHUF(r3, 0x4E)); r3 = VAND(r3, bmask0);
  r5  = VMADD(r5 , 0x55, r5 , VSRA(r4, BRADIX)); 
  r8  = VMADD(r8 , 0x55, r8 , VSHUF(r4, 0x4E)); r4 = VAND(r4, bmask0);
  r6  = VMADD(r6 , 0x55, r6 , VSRA(r5, BRADIX)); 
  r9  = VMADD(r9 , 0x55, r9 , VSHUF(r5, 0x4E)); r5 = VAND(r5, bmask0);
  r7  = VMADD(r7 , 0x55, r7 , VSRA(r6, BRADIX)); 
  r10 = VMADD(r10, 0x55, r10, VSHUF(r6, 0x4E)); r6 = VAND(r6, bmask0);
  r8  = VMADD(r8 , 0x55, r8 , VSRA(r7, BRADIX)); 
  r11 = VMADD(r11, 0x55, r11, VSHUF(r7, 0x4E)); r7 = VAND(r7, bmask0);
  t0  = VMADD(r9 , 0x55, r9 , VSRA(r8, BRADIX));
  t0  = VMADD(r10, 0x55, r10, VSRA(t0, BRADIX));
  t0  = VMADD(r11, 0x55, r11, VSRA(t0, BRADIX));
  t0  = VMADD(r8 , 0xAA, r8 , VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0  = VMADD(r9 , 0xAA, r9 , VSRA(t0, BRADIX));
  t0  = VMADD(r10, 0xAA, r10, VSRA(t0, BRADIX));
  t0  = VMADD(r11, 0xAA, r11, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r8  = VADD(r8 , VAND(p0, smask)); r9  = VADD(r9 , VAND(p1, smask));
  r10 = VADD(r10, VAND(p2, smask)); r11 = VADD(r11, VAND(p3, smask)); 

  // carry propagation 
  // r8 is finally 49-bit not 48-bit
  r9  = VADD(r9 , VSRA(r8 , BRADIX)); r8  = VAND(r8 , bmask1);
  r10 = VADD(r10, VSRA(r9 , BRADIX)); r9  = VAND(r9 , bmask1);
  r11 = VADD(r11, VSRA(r10, BRADIX)); r10 = VAND(r10, bmask1);
  r8  = VMADD(r8, 0xAA, r8, VSHUF(VSRA(r11, BRADIX), 0x4E)); 
  r11 = VAND(r11, bmask1);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
}

static void add_fpx2_4x2w(fp_4x2w r, const fp_4x2w a, const fp_4x2w b)
{
  __m512i a0  = a[0 ], a1  = a[1 ], a2  = a[2 ], a3  = a[3 ];
  __m512i a4  = a[4 ], a5  = a[5 ], a6  = a[6 ], a7  = a[7 ];
  __m512i a8  = a[8 ], a9  = a[9 ], a10 = a[10], a11 = a[11];
  __m512i b0  = b[0 ], b1  = b[1 ], b2  = b[2 ], b3  = b[3 ];
  __m512i b4  = b[4 ], b5  = b[5 ], b6  = b[6 ], b7  = b[7 ];
  __m512i b8  = b[8 ], b9  = b[9 ], b10 = b[10], b11 = b[11];
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask0 = VSET(0, BMASK, 0, BMASK, 0, BMASK, 0, BMASK);
  const __m512i bmask1 = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11; 
  __m512i t0, t1, t2, t3, smask;

  // r = a + b
  r0  = VADD(a0 , b0 ); r1  = VADD(a1 , b1 );
  r2  = VADD(a2 , b2 ); r3  = VADD(a3 , b3 );
  r4  = VADD(a4 , b4 ); r5  = VADD(a5 , b5 );
  r6  = VADD(a6 , b6 ); r7  = VADD(a7 , b7 );
  r8  = VADD(a8 , b8 ); r9  = VADD(a9 , b9 );
  r10 = VADD(a10, b10); r11 = VADD(a11, b11);

  // r = r - p*2^384
  r8  = VSUB(r8 , p0); r9  = VSUB(r9 , p1);
  r10 = VSUB(r10, p2); r11 = VSUB(r11, p3);

  // get sign mask + carry propagation
  r1  = VMADD(r1 , 0x55, r1 , VSRA(r0, BRADIX)); 
  r4  = VMADD(r4 , 0x55, r4 , VSHUF(r0, 0x4E)); r0 = VAND(r0, bmask0);
  r2  = VMADD(r2 , 0x55, r2 , VSRA(r1, BRADIX)); 
  r5  = VMADD(r5 , 0x55, r5 , VSHUF(r1, 0x4E)); r1 = VAND(r1, bmask0);
  r3  = VMADD(r3 , 0x55, r3 , VSRA(r2, BRADIX)); 
  r6  = VMADD(r6 , 0x55, r6 , VSHUF(r2, 0x4E)); r2 = VAND(r2, bmask0);
  r4  = VMADD(r4 , 0x55, r4 , VSRA(r3, BRADIX)); 
  r7  = VMADD(r7 , 0x55, r7 , VSHUF(r3, 0x4E)); r3 = VAND(r3, bmask0);
  r5  = VMADD(r5 , 0x55, r5 , VSRA(r4, BRADIX)); 
  r8  = VMADD(r8 , 0x55, r8 , VSHUF(r4, 0x4E)); r4 = VAND(r4, bmask0);
  r6  = VMADD(r6 , 0x55, r6 , VSRA(r5, BRADIX)); 
  r9  = VMADD(r9 , 0x55, r9 , VSHUF(r5, 0x4E)); r5 = VAND(r5, bmask0);
  r7  = VMADD(r7 , 0x55, r7 , VSRA(r6, BRADIX)); 
  r10 = VMADD(r10, 0x55, r10, VSHUF(r6, 0x4E)); r6 = VAND(r6, bmask0);
  r8  = VMADD(r8 , 0x55, r8 , VSRA(r7, BRADIX)); 
  r11 = VMADD(r11, 0x55, r11, VSHUF(r7, 0x4E)); r7 = VAND(r7, bmask0);
  t0  = VMADD(r9 , 0x55, r9 , VSRA(r8, BRADIX));
  t0  = VMADD(r10, 0x55, r10, VSRA(t0, BRADIX));
  t0  = VMADD(r11, 0x55, r11, VSRA(t0, BRADIX));
  t0  = VMADD(r8 , 0xAA, r8 , VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0  = VMADD(r9 , 0xAA, r9 , VSRA(t0, BRADIX));
  t0  = VMADD(r10, 0xAA, r10, VSRA(t0, BRADIX));
  t0  = VMADD(r11, 0xAA, r11, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r8  = VADD(r8 , VAND(p0, smask)); r9  = VADD(r9 , VAND(p1, smask));
  r10 = VADD(r10, VAND(p2, smask)); r11 = VADD(r11, VAND(p3, smask)); 

  // carry propagation 
  // r8 is finally 49-bit not 48-bit
  r9  = VADD(r9 , VSRA(r8 , BRADIX)); r8  = VAND(r8 , bmask1);
  r10 = VADD(r10, VSRA(r9 , BRADIX)); r9  = VAND(r9 , bmask1);
  r11 = VADD(r11, VSRA(r10, BRADIX)); r10 = VAND(r10, bmask1);
  r8  = VMADD(r8, 0xAA, r8, VSHUF(VSRA(r11, BRADIX), 0x4E)); 
  r11 = VAND(r11, bmask1);

  r[0 ] = r0 ; r[1 ] = r1 ; r[2 ] = r2 ; r[3 ] = r3 ;
  r[4 ] = r4 ; r[5 ] = r5 ; r[6 ] = r6 ; r[7 ] = r7 ;
  r[8 ] = r8 ; r[9 ] = r9 ; r[10] = r10; r[11] = r11;
}

// ----------------------------------------------------------------------------
// Fp2 single-length operations

static void add_fp2_8x1x1w(fp2_8x1x1w r, const fp2_8x1x1w a, const fp2_8x1x1w b)
{
  add_fp_8x1w(r[0], a[0], b[0]);
  add_fp_8x1w(r[1], a[1], b[1]);
}

static void add_fp2_4x2x1w(fp2_4x2x1w r, const fp2_4x2x1w a, const fp2_4x2x1w b)
{
  add_fp_8x1w(r, a, b);
}

// a = < D1 | D0 | C1 | C0 | B1 | B0 | A1 | A0 >  
// b = < H1 | H0 | G1 | G0 | F1 | F0 | E1 | E0 >
// r = < D1+H1 | D0+H0 | C1-G1 | C0-G0 | B1-F1 | B0-F0 | A1+E1 | A0+E0 >
static void assa_fp2_4x2x1w(fp2_4x2x1w r, const fp2_4x2x1w a, const fp2_4x2x1w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const __m512i p0 = VSET1(P48[0]), p1 = VSET1(P48[1]), p2 = VSET1(P48[2]);
  const __m512i p3 = VSET1(P48[3]), p4 = VSET1(P48[4]), p5 = VSET1(P48[5]);
  const __m512i p6 = VSET1(P48[6]), p7 = VSET1(P48[7]);
  const __m512i bmask = VSET1(BMASK);
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, smask;
  __m512i t0, t1, t2, t3, t4, t5, t6, t7;

  // r = D1+H1 | D0+H0 | C1 | C0 | B1 | B0 | A1+E1 | A0+E0
  r0 = VMADD(a0, 0xC3, a0, b0); r1 = VMADD(a1, 0xC3, a1, b1);
  r2 = VMADD(a2, 0xC3, a2, b2); r3 = VMADD(a3, 0xC3, a3, b3);
  r4 = VMADD(a4, 0xC3, a4, b4); r5 = VMADD(a5, 0xC3, a5, b5);
  r6 = VMADD(a6, 0xC3, a6, b6); r7 = VMADD(a7, 0xC3, a7, b7);

  // t = p | p | G1 | G0 | F1 | F0 | p | p
  t0 = VMBLEND(0xC3, b0, p0); t1 = VMBLEND(0xC3, b1, p1);
  t2 = VMBLEND(0xC3, b2, p2); t3 = VMBLEND(0xC3, b3, p3);
  t4 = VMBLEND(0xC3, b4, p4); t5 = VMBLEND(0xC3, b5, p5);
  t6 = VMBLEND(0xC3, b6, p6); t7 = VMBLEND(0xC3, b7, p7); 

  // r = D1+H1-p | D0+H0-p | C1-G1 | C0-G0 | B1-F1 | B0-F0 | A1+E1-p | A0+E0-p
  r0 = VSUB(r0, t0); r1 = VSUB(r1, t1); r2 = VSUB(r2, t2); r3 = VSUB(r3, t3);
  r4 = VSUB(r4, t4); r5 = VSUB(r5, t5); r6 = VSUB(r6, t6); r7 = VSUB(r7, t7);

  // get sign mask 
  t0 = VADD(r1, VSRA(r0, BRADIX));
  t0 = VADD(r2, VSRA(t0, BRADIX));
  t0 = VADD(r3, VSRA(t0, BRADIX));
  t0 = VADD(r4, VSRA(t0, BRADIX));
  t0 = VADD(r5, VSRA(t0, BRADIX));
  t0 = VADD(r6, VSRA(t0, BRADIX));
  t0 = VADD(r7, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 
  r4 = VADD(r4, VAND(p4, smask)); r5 = VADD(r5, VAND(p5, smask)); 
  r6 = VADD(r6, VAND(p6, smask)); r7 = VADD(r7, VAND(p7, smask)); 

  // carry propagation 
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r4 = VADD(r4, VSRA(r3, BRADIX)); r3 = VAND(r3, bmask);
  r5 = VADD(r5, VSRA(r4, BRADIX)); r4 = VAND(r4, bmask);
  r6 = VADD(r6, VSRA(r5, BRADIX)); r5 = VAND(r5, bmask);
  r7 = VADD(r7, VSRA(r6, BRADIX)); r6 = VAND(r6, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
  r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

// r0 = a0 - a1
// r1 = a0 + a1
static void mul_by_u_plus_1_fp2_4x2x1w(fp2_4x2x1w r, const fp2_4x2x1w a)
{
  fp2_4x2x1w t0;

  // a = A1 | A0 at Fp layer 
  shuf_01(t0, a);                       //     A0 |    A1
  asx4_fp_8x1w(r, a, t0);               //  A0+A1 | A0-A1
}

// r0 = (a0 + a1)*(a0 - a1)
// r1 = 2*a0*a1
static void sqr_fp2_4x2x1w(fp2_4x2x1w r, const fp2_4x2x1w a)
{
  fp2_4x2x1w t0, t1, t2;
  fp2x2_4x2x1w tt0;

  // a = A1 | A0 at Fp layer 
  shuf_00(t0, a);                       //      A0 |              A0
  shuf_01(t1, a);                       //      A0 |              A1 
  shuf_z1(t2, a);                       //       0 |              A1
  add_fp_8x1w(t0, t0, t1);              //    2*A0 |           A0+A1
  sub_fp_8x1w(t2, a, t2);               //      A1 |           A0-A1
  mul_fpx2_8x1w(tt0, t0, t2);           // 2*A0*A1 | (A0+A1)*(A0-A1)
  redc_fpx2_8x1w(r, tt0);               // 2*A0*A1 | (A0+A1)*(A0-A1)
}

// r0 = a0*b0 - a1*b1
// r1 = a0*b1 + a1*b0
static void mul_fp2_2x4x1w(fp2_2x4x1w r, const fp2_2x4x1w a, const fp2_2x4x1w b)
{
  fp2_2x4x1w t0, t1;
  fp2x2_2x4x1w tt0;

  // a = A1 | A0 | ... | ... at Fp layer
  // b = B1 | B0 | ... | ... at Fp layer
  perm_3322(t0, a);                     //        A1 |        A1 |   A0 |   A0
  perm_2332(t1, b);                     //        B0 |        B1 |   B1 |   B0
  mul_fpx2_8x1w(tt0, t0, t1);           //      A1B0 |      A1B1 | A0B1 | A0B0
  redc_fpx2_8x1w(t0, tt0);              //      A1B0 |      A1B1 | A0B1 | A0B0
  perm_10zz(t1, t0);                    //      A0B1 |      A0B0 |    0 |    0 
  asx4_fp_8x1w(r, t1, t0);              // A0B1+A1B0 | A0B0-A1B1 |  ... |  ...
}

static void add_fp2_2x2x2w(fp2_2x2x2w r, const fp2_2x2x2w a, const fp2_2x2x2w b)
{
  add_fp_4x2w(r, a, b);
}

// a = < B1' | B1 | B0' | B0 | A1' | A1 | A0' | A0 >
// b = < D1' | D1 | D0' | D0 | C1' | C1 | C0' | C0 >
// r = < B1'+D1' | B1+D1 | B0'+D0' | B0+D0 | A1'-C1' | A1-C1 | A0'-C0' | A0-C0 >
static void as_fp2_2x2x2w(fp2_2x2x2w r, const fp2_2x2x2w a, const fp2_2x2x2w b)
{
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  const __m512i p0 = VSET(P48[4], P48[0], P48[4], P48[0], P48[4], P48[0], P48[4], P48[0]);
  const __m512i p1 = VSET(P48[5], P48[1], P48[5], P48[1], P48[5], P48[1], P48[5], P48[1]);
  const __m512i p2 = VSET(P48[6], P48[2], P48[6], P48[2], P48[6], P48[2], P48[6], P48[2]);
  const __m512i p3 = VSET(P48[7], P48[3], P48[7], P48[3], P48[7], P48[3], P48[7], P48[3]);
  const __m512i bmask = VSET1(BMASK); 
  __m512i r0, r1, r2, r3, smask;
  __m512i t0, t1, t2, t3;

  // r =  B1'+D1' | B1+D1 | B0'+D0' | B0+D0 | A1' | A1 | A0' | A0 
  r0 = VMADD(a0, 0xF0, a0, b0); r1 = VMADD(a1, 0xF0, a1, b1);
  r2 = VMADD(a2, 0xF0, a2, b2); r3 = VMADD(a3, 0xF0, a3, b3);

  // t = p' | p | p' | p | C1' | C1 | C0' | C0
  t0 = VMBLEND(0xF0, b0, p0); t1 = VMBLEND(0xF0, b1, p1);
  t2 = VMBLEND(0xF0, b2, p2); t3 = VMBLEND(0xF0, b3, p3);

  // r = B1'+D1'-p' | B1+D1-p | B0'+D0'-p' | B0+D0-p | A1'-C1' | A1-C1 | A0'-C0' | A0-C0
  r0 = VSUB(r0, t0); r1 = VSUB(r1, t1); r2 = VSUB(r2, t2); r3 = VSUB(r3, t3);

  // get sign mask
  t0 = VMADD(r1, 0x55, r1, VSRA(r0, BRADIX));
  t0 = VMADD(r2, 0x55, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0x55, r3, VSRA(t0, BRADIX));
  t0 = VMADD(r0, 0xAA, r0, VZSHUF(0xCCCC, VSRA(t0, BRADIX), 0x4E));
  t0 = VMADD(r1, 0xAA, r1, VSRA(t0, BRADIX));
  t0 = VMADD(r2, 0xAA, r2, VSRA(t0, BRADIX));
  t0 = VMADD(r3, 0xAA, r3, VSRA(t0, BRADIX));

  // if r is non-negative, smask = all-0 
  // if r is     negative, smask = all-1
  smask = VSRA(t0, 63);
  smask = VSHUF(smask, 0xEE);

  // r = r + (p & smask)
  r0 = VADD(r0, VAND(p0, smask)); r1 = VADD(r1, VAND(p1, smask)); 
  r2 = VADD(r2, VAND(p2, smask)); r3 = VADD(r3, VAND(p3, smask)); 

  // carry propagation 
  // r0 is finally 49-bit not 48-bit
  r1 = VADD(r1, VSRA(r0, BRADIX)); r0 = VAND(r0, bmask);
  r2 = VADD(r2, VSRA(r1, BRADIX)); r1 = VAND(r1, bmask);
  r3 = VADD(r3, VSRA(r2, BRADIX)); r2 = VAND(r2, bmask);
  r0 = VMADD(r0, 0xAA, r0, VSHUF(VSRA(r3, BRADIX), 0x4E)); r3 = VAND(r3, bmask);

  r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; 
}

// r0 = (a0 + a1)*(a0 - a1)
// r1 = 2*a0*a1
static void sqr_fp2_2x2x2w(fp2_2x2x2w r, const fp2_2x2x2w a)
{
  fp2_2x2x2w t0, t1, t2;
  fp2x2_2x2x2w t3;

  // a = A1 | A0 at Fp layer
  perm_1010_hl(t0, a);                  //      A0 |              A0
  perm_1032_hl(t1, a);                  //      A0 |              A1 
  perm_zz32_hl(t2, a);                  //       0 |              A1
  add_fp_4x2w(t0, t0, t1);              //    2*A0 |           A0+A1
  sub_fp_4x2w(t2, a, t2);               //      A1 |           A0-A1
  mul_fpx2_4x2w(t3, t0, t2);            // 2*A0*A1 | (A0+A1)*(A0-A1)
  redc_fpx2_4x2w(r, t3);                // 2*A0*A1 | (A0+A1)*(A0-A1)
}

// r0 = a0 - a1
// r1 = a0 + a1
static void mul_by_u_plus_1_fp2_2x2x2w(fp2_2x2x2w r, const fp2_2x2x2w a)
{
  fp2_2x2x2w t0;

  // a = A1 | A0 at Fp layer 
  perm_1032_hl(t0, a);                  //     A0 |    A1
  asx2_fp_4x2w(r, a, t0);               //  A0+A1 | A0-A1
}

// r0 = a0*b0 - a1*b1
// r1 = a0*b1 + a1*b0
static void mul_fp2_1x4x2w(fp2_1x4x2w r, const fp2_1x4x2w a, const fp2_1x4x2w b)
{
  fp2_1x4x2w t0, t1;
  fp2x2_1x4x2w tt0;
  const __m512i m0 = VSET(7, 6, 7, 6, 5, 4, 5, 4);
  const __m512i m1 = VSET(5, 4, 7, 6, 7, 6, 5, 4);
  const __m512i m2 = VSET(3, 2, 1, 0, 7, 6, 5, 4);

  // a = A1 | A0 | ... | ... at Fp layer
  // b = B1 | B0 | ... | ... at Fp layer
  perm_var_hl(t0, a, m0);               //          A1 |          A1 |    A0 |    A0
  perm_var_hl(t1, b, m1);               //          B0 |          B1 |    B1 |    B0
  mul_fpx2_4x2w(tt0, t0, t1);           //       A1*B0 |       A1*B1 | A0*B1 | A0*B0
  redc_fpx2_4x2w(t0, tt0);              //       A1*B0 |       A1*B1 | A0*B1 | A0*B0
  perm_var_hl(t1, t0, m2);              //       A0*B1 |       A0*B0 |   ... |   ... 
  asx2_fp_4x2w(r, t1, t0);              // A0*B1+A1*B0 | A0*B0-A1*B1 |   ... |   ...
}

// ----------------------------------------------------------------------------
// Fp2 double-length operations

static void add_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a, const fp2x2_8x1x1w b)
{
  add_fpx2_8x1w(r[0], a[0], b[0]);
  add_fpx2_8x1w(r[1], a[1], b[1]);
}

static void sub_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a, const fp2x2_8x1x1w b)
{
  sub_fpx2_8x1w(r[0], a[0], b[0]);
  sub_fpx2_8x1w(r[1], a[1], b[1]);
}

// r0 = a0 - a1
// r1 = a0 + a1
static void mul_by_u_plus_1_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2x2_8x1x1w a)
{
  sub_fpx2_8x1w(r[0], a[0], a[1]);      // a0-a1
  add_fpx2_8x1w(r[1], a[0], a[1]);      // a0+a1
}

// r0 = a0*b0 - a1*b1
// r1 = (a0 + a1)*(b0 + b1) - a0*b0 - a1*b1 = a0*b1 + a1*b0
static void mul_fp2x2_8x1x1w(fp2x2_8x1x1w r, const fp2_8x1x1w a, const fp2_8x1x1w b)
{
  fp_8x1w t0, t1;
  fpx2_8x1w tt0, tt1, tt2;

  mul_fpx2_8x1w(tt0, a[0], b[0]);       // a0*b0
  mul_fpx2_8x1w(tt1, a[1], b[1]);       // a1*b1
  add_fp_8x1w(t0, a[0], a[1]);          // a0+a1
  add_fp_8x1w(t1, b[0], b[1]);          // b0+b1
  mul_fpx2_8x1w(tt2, t0, t1);           // (a0+a1)*(b0+b1)
  sub_fpx2_8x1w(r[0], tt0, tt1);        // a0*b0-a1*b1
  sub_fpx2_8x1w(tt2, tt2, tt0);         // (a0+a1)*(b0+b1)-a0*b0
  sub_fpx2_8x1w(r[1], tt2, tt1);        // (a0+a1)*(b0+b1)-a0*b0-a1*b1
}

static void add_fp2x2_4x2x1w(fp2x2_4x2x1w r, const fp2x2_4x2x1w a, const fp2x2_4x2x1w b)
{
  add_fpx2_8x1w(r, a, b);
}

// r0 = (a0 + a1)*(a0 - a1)
// r1 = 2*a0*a1
static void sqr_fp2x2_4x2x1w(fp2x2_4x2x1w r, const fp2_4x2x1w a)
{
  fp2_4x2x1w t0, t1, t2;

  // a = A1 | A0 at Fp layer 
  shuf_00(t0, a);                       //      A0 |              A0
  shuf_01(t1, a);                       //      A0 |              A1 
  shuf_z1(t2, a);                       //       0 |              A1
  add_fp_8x1w(t0, t0, t1);              //    2*A0 |           A0+A1
  sub_fp_8x1w(t2, a, t2);               //      A1 |           A0-A1
  mul_fpx2_8x1w(r, t0, t2);             // 2*A0*A1 | (A0+A1)*(A0-A1) 
}

// r0 = a0 - a1
// r1 = a0 + a1
static void mul_by_u_plus_1_fp2x2_4x2x1w(fp2x2_4x2x1w r, const fp2x2_4x2x1w a)
{
  fp2x2_4x2x1w tt0;

  // a = A1 | A0 at Fp layer 
  shuf_01_dl(tt0, a);                   //     A0 |    A1
  asx4_fpx2_8x1w(r, a, tt0);            //  A0+A1 | A0-A1
}

// r0 = a0*b0 - a1*b1
// r1 = a0*b1 + a1*b0
static void mul_fp2x2_2x4x1w(fp2x2_2x4x1w r, const fp2_2x4x1w a, const fp2_2x4x1w b)
{
  fp2_2x4x1w t0, t1;
  fp2x2_2x4x1w tt0, tt1;

  // a = A1 | A0 | ... | ... at Fp layer
  // b = B1 | B0 | ... | ... at Fp layer
  perm_3322(t0, a);                     //          A1 |          A1 |   A0 |   A0
  perm_2332(t1, b);                     //          B0 |          B1 |   B1 |   B0
  mul_fpx2_8x1w(tt0, t0, t1);           //        A1B0 |        A1B1 | A0B1 | A0B0
  perm_10zz_dl(tt1, tt0);               //       A0*B1 |       A0*B0 |    0 |    0 
  asx4_fpx2_8x1w(r, tt1, tt0);          // A0*B1+A1*B0 | A0*B0-A1*B1 |  ... |  ...
}

static void redc_fp2x2_4x2x1w(fp2_4x2x1w r, const fp2x2_4x2x1w a)
{
  redc_fpx2_8x1w(r, a);
}

static void redc_fp2x2_2x2x2w(fp2_2x2x2w r, const fp2x2_2x2x2w a)
{
  redc_fpx2_4x2w(r, a);
}

// r0 = a0 - a1
// r1 = a0 + a1
static void mul_by_u_plus_1_fp2x2_2x2x2w(fp2x2_2x2x2w r, const fp2x2_2x2x2w a)
{
  fp2x2_2x2x2w t0;

  // a = A1 | A0 at Fp layer 
  perm_1032(t0, a);                         //     A0 |    A1
  perm_1032_hl(&t0[NWORDS], &a[NWORDS]);    //     A0 |    A1
  asx2_fpx2_4x2w(r, a, t0);                 //  A0+A1 | A0-A1
}

static void add_fp2x2_2x2x2w(fp2x2_2x2x2w r, const fp2x2_2x2x2w a, const fp2x2_2x2x2w b)
{
  add_fpx2_4x2w(r, a, b);
}

// r0 = (a0 + a1)*(a0 - a1)
// r1 = 2*a0*a1
static void sqr_fp2x2_2x2x2w(fp2x2_2x2x2w r, const fp2_2x2x2w a)
{
  fp2_2x2x2w t0, t1, t2;
  fp2x2_2x2x2w t3;

  // a = A1 | A0 at Fp layer
  perm_1010_hl(t0, a);                  //      A0 |              A0
  perm_1032_hl(t1, a);                  //      A0 |              A1 
  perm_zz32_hl(t2, a);                  //       0 |              A1
  add_fp_4x2w(t0, t0, t1);              //    2*A0 |           A0+A1
  sub_fp_4x2w(t2, a, t2);               //      A1 |           A0-A1
  mul_fpx2_4x2w(r, t0, t2);             // 2*A0*A1 | (A0+A1)*(A0-A1)
}

// r0 = a0*b0 - a1*b1
// r1 = a0*b1 + a1*b0
static void mul_fp2x2_1x4x2w(fp2x2_1x4x2w r, const fp2_1x4x2w a, const fp2_1x4x2w b)
{
  fp2_1x4x2w t0, t1;
  fp2x2_1x4x2w tt0, tt1;
  const __m512i m0 = VSET(7, 6, 7, 6, 5, 4, 5, 4);
  const __m512i m1 = VSET(5, 4, 7, 6, 7, 6, 5, 4);
  const __m512i m2 = VSET(3, 2, 1, 0, 7, 6, 5, 4);

  // a = A1 | A0 | ... | ... at Fp layer
  // b = B1 | B0 | ... | ... at Fp layer
  perm_var_hl(t0, a, m0);                       //          A1 |          A1 |    A0 |    A0
  perm_var_hl(t1, b, m1);                       //          B0 |          B1 |    B1 |    B0
  mul_fpx2_4x2w(tt0, t0, t1);                   //       A1*B0 |       A1*B1 | A0*B1 | A0*B0
  perm_var(tt1, tt0, m2);                       //       A0*B1 |       A0*B0 |   ... |   ... 
  perm_var_hl(&tt1[NWORDS], &tt0[NWORDS], m2);  //       A0*B1 |       A0*B0 |   ... |   ... 
  asx2_fpx2_4x2w(r, tt1, tt0);                  // A0*B1+A1*B0 | A0*B0-A1*B1 |   ... |   ...
}

// ----------------------------------------------------------------------------
// Fp4 operations

// r0 = a0^2 + (u+1)*a1^2
// r1 = 2*a0*a1
// double-length version
static void sqr_fp4_2x2x2x1w_v1(fp4_2x2x2x1w r, const fp4_2x2x2x1w a)
{
  fp4_2x2x2x1w t0;
  fp4x2_2x2x2x1w tt0, tt1;

  // a = A1 | A0 at Fp2 layer
  sqr_fp2x2_4x2x1w(tt0, a);                 //        A1^2 |              A0^2
  mul_by_u_plus_1_fp2x2_4x2x1w(tt1, tt0);   //  (u+1)*A1^2 |               ...
  perm_zz32_dl(tt1, tt1);                   //           0 |        (u+1)*A1^2
  add_fp2x2_4x2x1w(tt0, tt0, tt1);          //         ... | A0^2+(u+1)*(A1^2)
  perm_10zz(t0, a);                         //          A0 |                 0
  mul_fp2x2_2x4x1w(tt1, a, t0);             //       A0*A1 |               ...
  add_fp2x2_4x2x1w(tt1, tt1, tt1);          //     2*A0*A1 |               ...
  blend_0x33_dl(tt0, tt1, tt0);             //     2*A0*A1 | A0^2+(u+1)*(A1^2)
  redc_fp2x2_4x2x1w(r, tt0);                //     2*A0*A1 | A0^2+(u+1)*(A1^2)
}

// r0 = a0^2 + (u+1)*a1^2
// r1 = 2*a0*a1
// single-length version
static void sqr_fp4_2x2x2x1w_v2(fp4_2x2x2x1w r, const fp4_2x2x2x1w a)
{
  fp4_2x2x2x1w t0, t1, t2;

  // a = A1 | A0 at Fp2 layer
  sqr_fp2_4x2x1w(t0, a);                //        A1^2 |              A0^2
  mul_by_u_plus_1_fp2_4x2x1w(t1, t0);   //  (u+1)*A1^2 |               ...
  perm_zz32(t1, t1);                    //           0 |        (u+1)*A1^2
  add_fp2_4x2x1w(t0, t0, t1);           //         ... | A0^2+(u+1)*(A1^2)
  perm_10zz(t1, a);                     //          A0 |                 0
  mul_fp2_2x4x1w(t2, a, t1);            //       A0*A1 |               ...
  add_fp2_4x2x1w(t2, t2, t2);           //     2*A0*A1 |               ...
  blend_0x33(r, t2, t0);                //     2*A0*A1 | A0^2+(u+1)*(A1^2)
}

// r0 = a0^2 + (u+1)*a1^2
// r1 = 2*a0*a1
// single-length version
// TODO: double-length version
static void sqr_fp4_1x2x2x2w_v1(fp4_1x2x2x2w r, const fp4_1x2x2x2w a)
{
  fp4x2_1x2x2x2w tt0, tt1, tt2;
  fp4_1x2x2x2w t0;
  const __m512i m0 = VSET(3, 2, 1, 0, 7, 6, 5, 4);

  // a = A1 | A0 at Fp2 layer
  sqr_fp2x2_2x2x2w(tt0, a);                     //        A1^2 |              A0^2
  mul_by_u_plus_1_fp2x2_2x2x2w(tt1, tt0);       //  (u+1)*A1^2 |               ...
  perm_var(tt1, tt1, m0);                       //         ... |        (u+1)*A1^2
  perm_var_hl(&tt1[NWORDS], &tt1[NWORDS], m0);  //         ... |        (u+1)*A1^2
  add_fp2x2_2x2x2w(tt0, tt0, tt1);              //         ... | A0^2+(u+1)*(A1^2)
  perm_var_hl(t0, a, m0);                       //          A0 |                 0
  mul_fp2x2_1x4x2w(tt2, a, t0);                 //       A0*A1 |               ...
  add_fp2x2_2x2x2w(tt2, tt2, tt2);              //     2*A0*A1 |               ...  
  blend_0xF0(tt0, tt0, tt2);                    //     2*A0*A1 | A0^2+(u+1)*(A1^2)
  blend_0xF0_hl(&tt0[NWORDS], &tt0[NWORDS], &tt2[NWORDS]);
  redc_fp2x2_2x2x2w(r, tt0);                    //     2*A0*A1 | A0^2+(u+1)*(A1^2)
}

static void sqr_fp4_1x2x2x2w_v2(fp4_1x2x2x2w r, const fp4_1x2x2x2w a)
{
  fp4_1x2x2x2w t0, t1, t2;
  const __m512i m0 = VSET(3, 2, 1, 0, 7, 6, 5, 4);

  // a = A1 | A0 at Fp2 layer
  sqr_fp2_2x2x2w(t0, a);                //        A1^2 |              A0^2
  mul_by_u_plus_1_fp2_2x2x2w(t1, t0);   //  (u+1)*A1^2 |               ...
  perm_var_hl(t1, t1, m0);              //         ... |        (u+1)*A1^2
  add_fp2_2x2x2w(t0, t0, t1);           //         ... | A0^2+(u+1)*(A1^2)
  perm_var_hl(t1, a, m0);               //          A0 |                 0
  mul_fp2_1x4x2w(t2, a, t1);            //       A0*A1 |               ...
  add_fp2_2x2x2w(t2, t2, t2);           //     2*A0*A1 |               ...  
  blend_0xF0_hl(r, t0, t2);             //     2*A0*A1 | A0^2+(u+1)*(A1^2)
}

// ----------------------------------------------------------------------------
// Fp6 operations

// r0 = ((a1 + a2)*(b1 + b2) - a1*b1 - a2*b2)*(u+1) + a0*b0 = (a1*b2 + a2*b1)*(u+1) + a0*b0 
// r1 = (a0 + a1)*(b0 + b1) - a0*b0 - a1*b1 + a2*b2*(u+1) = a0*b1 + a1*b0 + a2*b2*(u+1)
// r2 = (a0 + a2)*(b0 + b2) - a0*b0 - a2*b2 + a1*b1 = a0*b2 + a2*b0 + a1*b1
// Karatsuba 
void mul_fp6x2_4x2x1x1w(fp2x2_8x1x1w r01, fp2x2_8x1x1w r2, const fp2_8x1x1w ab0, const fp2_8x1x1w ab1, const fp2_8x1x1w ab2)
{
  fp2_8x1x1w t0, t1, t2, t3, t4;
  fp2x2_8x1x1w tt0, tt1, tt2, tt3, tt4, tt5;

  // ab0 = b0 | a0 at Fp2 layer
  // ab1 = b1 | a1 at Fp2 layer
  // ab2 = b2 | a2 at Fp2 layer
  add_fp2_8x1x1w(t0, ab1, ab2);             //                                   b1+b2 |                                     a1+a2
  add_fp2_8x1x1w(t1, ab0, ab1);             //                                   b0+b1 |                                     a0+a1
  add_fp2_8x1x1w(t2, ab0, ab2);             //                                   b0+b2 |                                     a0+a2
  shuf_01_fp2_8x1x1w(t3, ab1);              //                                      a1 |                                        b1
  blend_0x55_fp2_8x1x1w(t3, t3, ab0);       //                                      a1 |                                        a0
  shuf_01_fp2_8x1x1w(t4, ab0);              //                                      a0 |                                        b0
  blend_0x55_fp2_8x1x1w(t4, ab1, t4);       //                                      b1 |                                        b0
  mul_fp2x2_8x1x1w(tt0, t3, t4);            //                                   a1*b1 |                                     a0*b0
  shuf_01_fp2_8x1x1w(t3, ab2);              //                                      a2 |                                        b2
  blend_0x55_fp2_8x1x1w(t3, t3, t0);        //                                      a2 |                                     a1+a2
  shuf_01_fp2_8x1x1w(t4, t0);               //                                   a1+a2 |                                     b1+b2
  blend_0x55_fp2_8x1x1w(t4, ab2, t4);       //                                      b2 |                                     b1+b2
  mul_fp2x2_8x1x1w(tt1, t3, t4);            //                                   a2*b2 |                           (a1+a2)*(b1+b2)
  shuf_01_fp2_8x1x1w(t3, t1);               //                                   a0+a1 |                                     b0+b1
  blend_0x55_fp2_8x1x1w(t3, t3, t2);        //                                   a0+a1 |                                     a0+a2
  shuf_01_fp2_8x1x1w(t4, t2);               //                                   a0+a2 |                                     b0+b2
  blend_0x55_fp2_8x1x1w(t4, t1, t4);        //                                   b0+b1 |                                     b0+b2
  mul_fp2x2_8x1x1w(tt2, t3, t4);            //                         (a0+a1)*(b0+b1) |                           (a0+a2)*(b0+b2)
  blend_0x55_fp2x2_8x1x1w(tt3, tt2, tt1);   //                         (a0+a1)*(b0+b1) |                           (a1+a2)*(b1+b2)
  shuf_11_fp2x2_8x1x1w(tt4, tt0);           //                                   a1*b1 |                                     a1*b1
  sub_fp2x2_8x1x1w(tt3, tt3, tt4);          //                   (a0+a1)*(b0+b1)-a1*b1 |                     (a1+a2)*(b1+b2)-a1*b1
  shuf_01_fp2x2_8x1x1w(tt2, tt2);           //                         (a0+a2)*(b0+b2) |                           (a0+a1)*(b0+b1)
  blend_0x55_fp2x2_8x1x1w(tt2, tt2, tt3);   //                         (a0+a2)*(b0+b2) |                     (a1+a2)*(b1+b2)-a1*b1
  shuf_11_fp2x2_8x1x1w(tt1, tt1);           //                                   a2*b2 |                                     a2*b2
  sub_fp2x2_8x1x1w(tt2, tt2, tt1);          //                   (a0+a2)*(b0+b2)-a2*b2 |               (a1+a2)*(b1+b2)-a1*b1-a2*b2
  shuf_11_fp2x2_8x1x1w(tt5, tt2);           //                   (a0+a2)*(b0+b2)-a2*b2 |                     (a0+a2)*(b0+b2)-a2*b2
  blend_0x55_fp2x2_8x1x1w(tt5, tt3, tt5);   //                   (a0+a1)*(b0+b1)-a1*b1 |                     (a0+a2)*(b0+b2)-a2*b2
  shuf_00_fp2x2_8x1x1w(tt0, tt0);           //                                   a0*b0 |                                     a0*b0
  sub_fp2x2_8x1x1w(tt5, tt5, tt0);          //             (a0+a1)*(b0+b1)-a1*b1-a0*b0 |               (a0+a2)*(b0+b2)-a2*b2-a0*b0
  blend_0x55_fp2x2_8x1x1w(tt2, tt1, tt2);   //                                   a2*b2 |               (a1+a2)*(b1+b2)-a1*b1-a2*b2
  mul_by_u_plus_1_fp2x2_8x1x1w(tt1, tt2);   //                             a2*b2*(u+1) |       ((a1+a2)*(b1+b2)-a1*b1-a2*b2)*(u+1)
  blend_0x55_fp2x2_8x1x1w(tt0, tt5, tt0);   //             (a0+a1)*(b0+b1)-a1*b1-a0*b0 |                                     a0*b0
  add_fp2x2_8x1x1w(r01, tt0, tt1);          // (a0+a1)*(b0+b1)-a1*b1-a0*b0+a2*b2*(u+1) | ((a1+a2)*(b1+b2)-a1*b1-a2*b2)*(u+1)+a0*b0
  add_fp2x2_8x1x1w(r2, tt5, tt4);           //                                     ... |         (a0+a2)*(b0+b2)-a2*b2-a0*b0+a1*b1
}

// ----------------------------------------------------------------------------
// Fp12 operations

// To understand the comments, see Listing 21 in "Guide to Pairing-Based Cryptography". 
void cyclotomic_sqr_fp12_vec_v1(fp4_1x2x2x2w ra, fp4_2x2x2x1w rbc, const fp4_1x2x2x2w a, const fp4_2x2x2x1w bc)
{
  fp4_1x2x2x2w ta; 
  fp4_2x2x2x1w tbc, t0;
  const __m512i m0 = VSET(3, 2, 1, 0, 5, 4, 7, 6);

  // compute A in 1x2x2x2w (some limbs are not fully shortened)
  // a = z1 | z0 at Fp2 layer
  sqr_fp4_1x2x2x2w_v1(ta, a);           //        t1 |        t0
  as_fp2_2x2x2w(ra, ta, a);             //     t1+z1 |     t0-z0
  add_fp2_2x2x2w(ra, ra, ra);           // 2*(t1+z1) | 2*(t0-z0)
  add_fp2_2x2x2w(ra, ra, ta);           // 3*t1+2*z1 | 3*t0-2*z0

  // compute B and C in 2x2x2x1w
  // bc = z5 | z4 | z3 | z2 at Fp2 layer
  sqr_fp4_2x2x2x1w_v1(tbc, bc);         //        t3 |        t2 |        t1 |              t0
  mul_by_u_plus_1_fp2_4x2x1w(t0, tbc);  //  t3*(u+1) |       ... |       ... |             ...
  blend_0xC0(tbc, tbc, t0);             //  t3*(u+1) |        t2 |        t1 |              t0
  perm_var(tbc, tbc, m0);               //        t1 |        t0 |        t2 |        t3*(u+1)
  assa_fp2_4x2x1w(rbc, tbc, bc);        //     t1+z5 |     t0-z4 |     t2-z3 |     t3*(u+1)+z2
  add_fp2_4x2x1w(rbc, rbc, rbc);        // 2*(t1+z5) | 2*(t0-z4) | 2*(t2-z3) | 2*(t3*(u+1)+z2)
  add_fp2_4x2x1w(rbc, rbc, tbc);        // 3*t1+2*z5 | 3*t0-2*z4 | 3*t2-2*z3 | 3*t3*(u+1)+2*z2
}

// To understand the comments, see Listing 21 in "Guide to Pairing-Based Cryptography". 
void cyclotomic_sqr_fp12_vec_v2(fp4_1x2x2x2w ra, fp4_2x2x2x1w rbc, const fp4_1x2x2x2w a, const fp4_2x2x2x1w bc)
{
  fp4_1x2x2x2w ta; 
  fp4_2x2x2x1w tbc, t0;
  const __m512i m0 = VSET(3, 2, 1, 0, 5, 4, 7, 6);

  // compute A in 1x2x2x2w (some limbs are not fully shortened)
  // a = z1 | z0 at Fp2 layer
  sqr_fp4_1x2x2x2w_v2(ta, a);           //        t1 |        t0
  as_fp2_2x2x2w(ra, ta, a);             //     t1+z1 |     t0-z0
  add_fp2_2x2x2w(ra, ra, ra);           // 2*(t1+z1) | 2*(t0-z0)
  add_fp2_2x2x2w(ra, ra, ta);           // 3*t1+2*z1 | 3*t0-2*z0

  // compute B and C in 2x2x2x1w
  // bc = z5 | z4 | z3 | z2 at Fp2 layer
  sqr_fp4_2x2x2x1w_v2(tbc, bc);         //        t3 |        t2 |        t1 |              t0
  mul_by_u_plus_1_fp2_4x2x1w(t0, tbc);  //  t3*(u+1) |       ... |       ... |             ...
  blend_0xC0(tbc, tbc, t0);             //  t3*(u+1) |        t2 |        t1 |              t0
  perm_var(tbc, tbc, m0);               //        t1 |        t0 |        t2 |        t3*(u+1)
  assa_fp2_4x2x1w(rbc, tbc, bc);        //     t1+z5 |     t0-z4 |     t2-z3 |     t3*(u+1)+z2
  add_fp2_4x2x1w(rbc, rbc, rbc);        // 2*(t1+z5) | 2*(t0-z4) | 2*(t2-z3) | 2*(t3*(u+1)+z2)
  add_fp2_4x2x1w(rbc, rbc, tbc);        // 3*t1+2*z5 | 3*t0-2*z4 | 3*t2-2*z3 | 3*t3*(u+1)+2*z2
}

// schoolbook
// single-length version
void mul_fp12_vec_v1(fp2_4x2x1w r01, fp2_4x2x1w r2, const fp2_8x1x1w ab0, const fp2_8x1x1w ab1, const fp2_8x1x1w ab2)
{
  fp2x2_8x1x1w tt01, tt2, tt3, tt4;
  fp2x2_4x2x1w ss0, ss1, ss2, ss3;
  fp2_4x2x1w s0, s1;
  const __m512i m0 = VSET(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i m1 = VSET(3, 2, 1, 0, 5, 4, 7, 6);

  // ab0 =  b1[0] | a1[0] | b0[0] | a1[0] | b1[0] | a0[0] | b0[0] | a0[0] at Fp2 layer
  // ab1 =  b1[1] | a1[1] | b0[1] | a1[1] | b1[1] | a0[1] | b0[1] | a0[1] at Fp2 layer
  // ab2 =  b1[2] | a1[2] | b0[2] | a1[2] | b1[2] | a0[2] | b0[2] | a0[2] at Fp2 layer

  // tt01 = a1*b1[1] | a1*b1[0] | a1*b0[1] | a1*b0[0] | a0*b1[1] | a0*b1[0] | a0*b0[1] | a0*b0[0] at Fp2 layer
  // tt2  =      ... | a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[2] at Fp2 layer
  mul_fp6x2_4x2x1x1w(tt01, tt2, ab0, ab1, ab2);
  // tt3  = a1*b1[0] | a1*b1[1] | a1*b0[0] | a1*b0[1] | a0*b1[0] | a0*b1[1] | a0*b0[0] | a0*b0[1] at Fp2 layer
  shuf_01_fp2x2_8x1x1w(tt3, tt01);
  // tt4  = a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[2] |      ... at Fp2 layer
  shuf_01_fp2x2_8x1x1w(tt4, tt2);
  // tt4  = a1*b1[2] |      ... |      ... |      ... |      ... |      ... | a0*b0[2] | a0*b0[1] at Fp2 layer
  blend_0x01_fp2x2_8x1x1w(tt4, tt4, tt3);
  // tt2  =      ... | a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[0] at Fp2 layer
  blend_0x01_fp2x2_8x1x1w(tt2, tt2, tt01);  
  // tt01 = a1*b1[1] | a1*b1[0] | a1*b0[1] | a1*b0[0] | a0*b1[1] | a0*b1[0] | a0*b0[2] | a0*b0[1] at Fp2 layer
  blend_0x03_fp2x2_8x1x1w(tt01, tt01, tt4);
  // tt3[0]  = a1*b1[0][0] | a1*b1[1][0] | a1*b0[0][0] | a1*b0[1][0] | a0*b1[0][0] | a0*b1[1][0] | a0*b0[1][0] | a0*b0[2][0] at Fp layer
  // tt3[1]  = a1*b1[0][1] | a1*b1[1][1] | a1*b0[0][1] | a1*b0[1][1] | a0*b1[0][1] | a0*b1[1][1] | a0*b0[1][1] | a0*b0[2][1] at Fp layer
  shuf_01_fp2x2_8x1x1w(tt3, tt01);
  //    ss0  =          a1*b1[0] |          a1*b0[0] |          a0*b1[0] |          a0*b0[1] at Fp2 layer
  blend_0x55_dl(ss0, tt3[1], tt01[0]);
  //    ss1  =          a1*b1[1] |          a1*b0[1] |          a0*b1[1] |          a0*b0[2] at Fp2 layer
  blend_0x55_dl(ss1, tt01[1], tt3[0]);
  //    ss2  =          a1*b0[0] |          a1*b1[0] |          a0*b0[1] |          a0*b1[0] at Fp2 layer
  perm_1032_dl(ss2, ss0);
  //    ss2  =          a1*b1[1] |          a1*b1[0] |          a0*b1[1] |          a0*b1[0] at Fp2 layer
  blend_0x33_dl(ss2, ss1, ss2);
  //    ss2  =          a0*b1[1] |          a0*b1[0] |          a1*b1[1] |          a1*b1[0] at Fp2 layer
  perm_var_dl(ss2, ss2, m0);
  //    ss3  =          a1*b0[1] |          a1*b1[1] |          a0*b0[2] |          a0*b1[1] at Fp2 layer
  perm_1032_dl(ss3, ss1);
  //    ss3  =          a1*b0[1] |          a1*b0[0] |          a0*b0[2] |          a0*b0[1] at Fp2 layer
  blend_0x33_dl(ss3, ss3, ss0);
  //    ss2  = a0*b1[1]+a1*b0[1] | a0*b1[0]+a1*b0[0] | a0*b0[2]+a1*b1[1] | a0*b0[1]+a1*b1[0] at Fp2 layer
  add_fp2x2_4x2x1w(ss2, ss2, ss3);
  //    r01  =             r1[1] |             r1[0] |             r0[2] |             r0[1] at Fp2 layer
  redc_fp2x2_4x2x1w(r01, ss2);

  // single-length sub-routines
  // tt3[0] = a1*b1[2][1] | ... | a1*b0[2][1] | ... | a0*b1[2][1] | ... | a0*b0[0][1] | ... at Fp layer
  shuf_01_dl(tt3[0], tt2[1]);
  //    ss0 =       a1*b1[2] | a1*b0[2] | a0*b1[2] |       a0*b0[0] at Fp2 layer
  blend_0x55_dl(ss0, tt3[0], tt2[0]); 
  //     s0 =       a1*b1[2] | a1*b0[2] | a0*b1[2] |       a0*b0[0] at Fp2 layer
  redc_fp2x2_4x2x1w(s0, ss0);
  //     s1 = a1*b1[2]*(u+1) |      ... |      ... |            ... at Fp2 layer
  mul_by_u_plus_1_fp2_4x2x1w(s1, s0);
  //     s1 = a1*b1[2]*(u+1) | a1*b0[2] | a0*b1[2] |       a0*b0[0] at Fp2 layer
  blend_0xC0(s1, s0, s1);
  //     s0 =            ... |      ... | a1*b0[2] | a1*b1[2]*(u+1) at Fp2 layer
  perm_var(s0, s1, m1);
  //     r2 =            ... |      ... |    r1[2] |          r0[0] at Fp2 layer
  add_fp2_4x2x1w(r2, s0, s1);
}

// schoolbook
// double-length version
void mul_fp12_vec_v2(fp2_4x2x1w r01, fp2_2x2x2w r2, const fp2_8x1x1w ab0, const fp2_8x1x1w ab1, const fp2_8x1x1w ab2)
{
  fp2x2_8x1x1w tt01, tt2, tt3, tt4;
  fp2x2_4x2x1w ss0, ss1, ss2, ss3;
  fp2_4x2x1w s0, s1;
  fp2x2_2x2x2w hh0;
  const __m512i m0 = VSET(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i m1 = VSET(3, 2, 1, 0, 5, 4, 7, 6);
  const __m512i m2 = VSET(3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i m3 = VSET(0, FMASK, 0, FMASK, 0, FMASK, 0, FMASK);
  int i;

  // ab0 =  b1[0] | a1[0] | b0[0] | a1[0] | b1[0] | a0[0] | b0[0] | a0[0] at Fp2 layer
  // ab1 =  b1[1] | a1[1] | b0[1] | a1[1] | b1[1] | a0[1] | b0[1] | a0[1] at Fp2 layer
  // ab2 =  b1[2] | a1[2] | b0[2] | a1[2] | b1[2] | a0[2] | b0[2] | a0[2] at Fp2 layer

  // tt01 = a1*b1[1] | a1*b1[0] | a1*b0[1] | a1*b0[0] | a0*b1[1] | a0*b1[0] | a0*b0[1] | a0*b0[0] at Fp2 layer
  // tt2  =      ... | a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[2] at Fp2 layer
  mul_fp6x2_4x2x1x1w(tt01, tt2, ab0, ab1, ab2);
  // tt3  = a1*b1[0] | a1*b1[1] | a1*b0[0] | a1*b0[1] | a0*b1[0] | a0*b1[1] | a0*b0[0] | a0*b0[1] at Fp2 layer
  shuf_01_fp2x2_8x1x1w(tt3, tt01);
  // tt4  = a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[2] |      ... at Fp2 layer
  shuf_01_fp2x2_8x1x1w(tt4, tt2);
  // tt4  = a1*b1[2] |      ... |      ... |      ... |      ... |      ... | a0*b0[2] | a0*b0[1] at Fp2 layer
  blend_0x01_fp2x2_8x1x1w(tt4, tt4, tt3);
  // tt2  =      ... | a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[0] at Fp2 layer
  blend_0x01_fp2x2_8x1x1w(tt2, tt2, tt01);  
  // tt01 = a1*b1[1] | a1*b1[0] | a1*b0[1] | a1*b0[0] | a0*b1[1] | a0*b1[0] | a0*b0[2] | a0*b0[1] at Fp2 layer
  blend_0x03_fp2x2_8x1x1w(tt01, tt01, tt4);
  // tt3[0]  = a1*b1[0][0] | a1*b1[1][0] | a1*b0[0][0] | a1*b0[1][0] | a0*b1[0][0] | a0*b1[1][0] | a0*b0[1][0] | a0*b0[2][0] at Fp layer
  // tt3[1]  = a1*b1[0][1] | a1*b1[1][1] | a1*b0[0][1] | a1*b0[1][1] | a0*b1[0][1] | a0*b1[1][1] | a0*b0[1][1] | a0*b0[2][1] at Fp layer
  shuf_01_fp2x2_8x1x1w(tt3, tt01);
  //    ss0  =          a1*b1[0] |          a1*b0[0] |          a0*b1[0] |          a0*b0[1] at Fp2 layer
  blend_0x55_dl(ss0, tt3[1], tt01[0]);
  //    ss1  =          a1*b1[1] |          a1*b0[1] |          a0*b1[1] |          a0*b0[2] at Fp2 layer
  blend_0x55_dl(ss1, tt01[1], tt3[0]);
  //    ss2  =          a1*b0[0] |          a1*b1[0] |          a0*b0[1] |          a0*b1[0] at Fp2 layer
  perm_1032_dl(ss2, ss0);
  //    ss2  =          a1*b1[1] |          a1*b1[0] |          a0*b1[1] |          a0*b1[0] at Fp2 layer
  blend_0x33_dl(ss2, ss1, ss2);
  //    ss2  =          a0*b1[1] |          a0*b1[0] |          a1*b1[1] |          a1*b1[0] at Fp2 layer
  perm_var_dl(ss2, ss2, m0);
  //    ss3  =          a1*b0[1] |          a1*b1[1] |          a0*b0[2] |          a0*b1[1] at Fp2 layer
  perm_1032_dl(ss3, ss1);
  //    ss3  =          a1*b0[1] |          a1*b0[0] |          a0*b0[2] |          a0*b0[1] at Fp2 layer
  blend_0x33_dl(ss3, ss3, ss0);
  //    ss2  = a0*b1[1]+a1*b0[1] | a0*b1[0]+a1*b0[0] | a0*b0[2]+a1*b1[1] | a0*b0[1]+a1*b1[0] at Fp2 layer
  add_fp2x2_4x2x1w(ss2, ss2, ss3);
  //    r01  =             r1[1] |             r1[0] |             r0[2] |             r0[1] at Fp2 layer
  redc_fp2x2_4x2x1w(r01, ss2);

  // double-length sub-routines
  // tt3[0] = a1*b1[2][1] | ... | a1*b0[2][1] | ... | a0*b1[2][1] | ... | a0*b0[0][1] | ... at Fp layer
  shuf_01_dl(tt3[0], tt2[1]);
  //    ss0 =       a1*b1[2] | a1*b0[2] |          a0*b1[2] |                a0*b0[0] at Fp2 layer
  blend_0x55_dl(ss0, tt3[0], tt2[0]); 
  //    ss1 = a1*b1[2]*(u+1) |      ... |               ... |                     ... at Fp2 layer
  mul_by_u_plus_1_fp2x2_4x2x1w(ss1, ss0);
  //    ss1 = a1*b1[2]*(u+1) | a1*b0[2] |          a0*b1[2] |                a0*b0[0] at Fp2 layer
  blend_0xC0_dl(ss1, ss0, ss1);
  //    ss0 =            ... |      ... |          a1*b0[2] |          a1*b1[2]*(u+1) at Fp2 layer
  perm_var_dl(ss0, ss1, m1);
  //    ss0 =            ... |      ... | a0*b1[2]+a1*b0[2] | a1*b1[2]*(u+1)+a0*b0[0] at Fp2 layer
  add_fp2x2_4x2x1w(ss0, ss0, ss1);
  //    hh0 =                             a0*b1[2]+a1*b0[2] | a1*b1[2]*(u+1)+a0*b0[0] at Fp2 layer
  perm_var_dl(ss0, ss0, m2);
  for (i = 0; i < VWORDS*2; i++) hh0[i] = VAND(ss0[i], m3);
  blend_0x55_hl(&hh0[VWORDS*2], &ss0[VWORDS*3], &ss0[VWORDS*2]);
  //     r2 =                                         r1[2] |                   r0[0] at Fp2 layer
  redc_fp2x2_2x2x2w(r2, hh0);
}

// schoolbook
// double-length version
void mul_fp12_vec_v3(fp2_4x2x1w r01, fp2_2x2x2w r2, const fp2_8x1x1w ab0, const fp2_8x1x1w ab1, const fp2_8x1x1w ab2)
{
  fp2x2_8x1x1w tt01, tt2, tt3, tt4;
  fp2x2_4x2x1w ss0, ss1, ss2, ss3;
  fp2_4x2x1w s0, s1;
  fp2x2_2x2x2w hh0, hh1, hh2;
  const __m512i m0 = VSET(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i m1 = VSET(3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i m2 = VSET(5, 5, 4, 4, 7, 7, 6, 6);
  const __m512i m3 = VSET(0, FMASK, 0, FMASK, 0, FMASK, 0, FMASK);
  int i;

  // ab0 =  b1[0] | a1[0] | b0[0] | a1[0] | b1[0] | a0[0] | b0[0] | a0[0] at Fp2 layer
  // ab1 =  b1[1] | a1[1] | b0[1] | a1[1] | b1[1] | a0[1] | b0[1] | a0[1] at Fp2 layer
  // ab2 =  b1[2] | a1[2] | b0[2] | a1[2] | b1[2] | a0[2] | b0[2] | a0[2] at Fp2 layer

  // tt01 = a1*b1[1] | a1*b1[0] | a1*b0[1] | a1*b0[0] | a0*b1[1] | a0*b1[0] | a0*b0[1] | a0*b0[0] at Fp2 layer
  // tt2  =      ... | a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[2] at Fp2 layer
  mul_fp6x2_4x2x1x1w(tt01, tt2, ab0, ab1, ab2);
  // tt3  = a1*b1[0] | a1*b1[1] | a1*b0[0] | a1*b0[1] | a0*b1[0] | a0*b1[1] | a0*b0[0] | a0*b0[1] at Fp2 layer
  shuf_01_fp2x2_8x1x1w(tt3, tt01);
  // tt4  = a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[2] |      ... at Fp2 layer
  shuf_01_fp2x2_8x1x1w(tt4, tt2);
  // tt4  = a1*b1[2] |      ... |      ... |      ... |      ... |      ... | a0*b0[2] | a0*b0[1] at Fp2 layer
  blend_0x01_fp2x2_8x1x1w(tt4, tt4, tt3);
  // tt2  =      ... | a1*b1[2] |      ... | a1*b0[2] |      ... | a0*b1[2] |      ... | a0*b0[0] at Fp2 layer
  blend_0x01_fp2x2_8x1x1w(tt2, tt2, tt01);  
  // tt01 = a1*b1[1] | a1*b1[0] | a1*b0[1] | a1*b0[0] | a0*b1[1] | a0*b1[0] | a0*b0[2] | a0*b0[1] at Fp2 layer
  blend_0x03_fp2x2_8x1x1w(tt01, tt01, tt4);
  // tt3[0]  = a1*b1[0][0] | a1*b1[1][0] | a1*b0[0][0] | a1*b0[1][0] | a0*b1[0][0] | a0*b1[1][0] | a0*b0[1][0] | a0*b0[2][0] at Fp layer
  // tt3[1]  = a1*b1[0][1] | a1*b1[1][1] | a1*b0[0][1] | a1*b0[1][1] | a0*b1[0][1] | a0*b1[1][1] | a0*b0[1][1] | a0*b0[2][1] at Fp layer
  shuf_01_fp2x2_8x1x1w(tt3, tt01);
  //    ss0  =          a1*b1[0] |          a1*b0[0] |          a0*b1[0] |          a0*b0[1] at Fp2 layer
  blend_0x55_dl(ss0, tt3[1], tt01[0]);
  //    ss1  =          a1*b1[1] |          a1*b0[1] |          a0*b1[1] |          a0*b0[2] at Fp2 layer
  blend_0x55_dl(ss1, tt01[1], tt3[0]);
  //    ss2  =          a1*b0[0] |          a1*b1[0] |          a0*b0[1] |          a0*b1[0] at Fp2 layer
  perm_1032_dl(ss2, ss0);
  //    ss2  =          a1*b1[1] |          a1*b1[0] |          a0*b1[1] |          a0*b1[0] at Fp2 layer
  blend_0x33_dl(ss2, ss1, ss2);
  //    ss2  =          a0*b1[1] |          a0*b1[0] |          a1*b1[1] |          a1*b1[0] at Fp2 layer
  perm_var_dl(ss2, ss2, m0);
  //    ss3  =          a1*b0[1] |          a1*b1[1] |          a0*b0[2] |          a0*b1[1] at Fp2 layer
  perm_1032_dl(ss3, ss1);
  //    ss3  =          a1*b0[1] |          a1*b0[0] |          a0*b0[2] |          a0*b0[1] at Fp2 layer
  blend_0x33_dl(ss3, ss3, ss0);
  //    ss2  = a0*b1[1]+a1*b0[1] | a0*b1[0]+a1*b0[0] | a0*b0[2]+a1*b1[1] | a0*b0[1]+a1*b1[0] at Fp2 layer
  add_fp2x2_4x2x1w(ss2, ss2, ss3);
  //    r01  =             r1[1] |             r1[0] |             r0[2] |             r0[1] at Fp2 layer
  redc_fp2x2_4x2x1w(r01, ss2);

  // double-length sub-routines
  // tt3[0] = a1*b1[2][1] | ... | a1*b0[2][1] | ... | a0*b1[2][1] | ... | a0*b0[0][1] | ... at Fp layer
  shuf_01_dl(tt3[0], tt2[1]);
  //    ss0 =  a1*b1[2] | a1*b0[2] | a0*b1[2] | a0*b0[0] at Fp2 layer
  blend_0x55_dl(ss0, tt3[0], tt2[0]); 
  //    hh0 =         a0*b1[2] |                a0*b0[0] at Fp2 layer
  perm_var_dl(ss1, ss0, m1);
  for (i = 0; i < VWORDS*2; i++) hh0[i] = VAND(ss1[i], m3);
  blend_0x55_hl(&hh0[VWORDS*2], &ss1[VWORDS*3], &ss1[VWORDS*2]);
  //   hh1 =          a1*b0[2] |                a1*b1[2] at Fp2 layer
  perm_var_dl(ss1, ss0, m2);
  for (i = 0; i < VWORDS*2; i++) hh1[i] = VAND(ss1[i], m3);
  blend_0x55_hl(&hh1[VWORDS*2], &ss1[VWORDS*3], &ss1[VWORDS*2]);
  //   hh2 =               ... |          a1*b1[2]*(u+1) at Fp2 layer
  mul_by_u_plus_1_fp2x2_2x2x2w(hh2, hh1);
  //   hh1 =          a1*b0[2] |          a1*b1[2]*(u+1) at Fp2 layer
  blend_0xF0(hh1, hh2, hh1);
  blend_0xF0_hl(&hh1[NWORDS], &hh2[NWORDS], &hh1[NWORDS]);
  //   hh0 = a0*b1[2]+a1*b0[2] | a1*b1[2]*(u+1)+a0*b0[0] at Fp2 layer
  add_fp2x2_2x2x2w(hh0, hh0, hh1);
  //     r2 =            r1[2] |                   r0[0] at Fp2 layer
  redc_fp2x2_2x2x2w(r2, hh0);
}

// ----------------------------------------------------------------------------

