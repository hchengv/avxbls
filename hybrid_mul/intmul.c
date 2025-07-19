#include <stdio.h>
#include <stdint.h>

#include "intrin.h"

// ----------------------------------------------------------------------------

#define BRADIX  48
#define NWORDS  8
#define VWORDS  4 // NWORDS / 2
#define SWORDS  6
#define BMASK   0xFFFFFFFFFFFFULL
#define FMASK   0xFFFFFFFFFFFFFFFFULL
#define BALIGN  4 // 52 (AVX-512IFMA) - 48 (BRADIX) = 4 
#define IFMAMASK  0xFFFFFFFFFFFFFULL

typedef uint64_t limb_t;

#define LIMB_T_BITS   64

#define TO_LIMB_T(limb64)     limb64

#define NLIMBS(bits)   (bits/LIMB_T_BITS)

typedef limb_t vec384[NLIMBS(384)];
typedef limb_t vec768[NLIMBS(768)];

// Fp element
typedef __m512i   fp_8x1w[NWORDS];
typedef __m512i   fpx2_8x1w[2*NWORDS];
typedef __m512i   fp_4x2w[VWORDS];
typedef __m512i   fpx2_4x2w[3*VWORDS];

// ----------------------------------------------------------------------------

#define ADCX(R, A)   __asm__ volatile (\
    "adcx %[a], %[res]"                \
    : [res] "+r" (R)                   \
    : [a] "r" (A)                      \
    : "cc"                             \
  );

#define ADOX(R, A)   __asm__ volatile (\
    "adox %[a], %[res]"                \
    : [res] "+r" (R)                   \
    : [a] "r" (A)                      \
    : "cc"                             \
  );

// ----------------------------------------------------------------------------

// various multiplications

// Karatsuba (excl. carry prop.)
// vector subroutine
void mul_fpx2_8x1w(fpx2_8x1w r, const fp_8x1w a, const fp_8x1w b)
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

// Karatsuba (excl. carry prop.)
// vector subroutine + 1*scalar subroutine
void mul_fpx2_8x1w_hybrid_v1(fpx2_8x1w r, uint64_t *s, const fp_8x1w a, const fp_8x1w b, const uint64_t *c, const uint64_t *d)
{
  // vector variables
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

  // scalar variables
  const uint64_t c0 = c[0], c1 = c[1], c2 = c[2];
  const uint64_t c3 = c[3], c4 = c[4], c5 = c[5];
  const uint64_t d0 = d[0], d1 = d[1], d2 = d[2];
  const uint64_t d3 = d[3], d4 = d[4], d5 = d[5];
  uint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, t0, t1;
  uint8_t e0, e1;
  uint64_t zero = 0;

  // compute vector zL(z0-z7) by aL(a0-a3) * bL(b0-b4)
  // compute scalar x = c * d

  z0 = VMACLO(z0, a0, b0);
  y0 = VMACHI(y0, a0, b0);
  y0 = VSHL(y0, BALIGN);

  __asm__ volatile(
    "test %%eax, %%eax" 
    :::"eax","cc"
  );

  x0 = _mulx_u64(c0, d0, (long long unsigned int *)&x1);
  t0 = _mulx_u64(c1, d0, (long long unsigned int *)&x2); ADCX(x1, t0);
  t0 = _mulx_u64(c2, d0, (long long unsigned int *)&x3); ADCX(x2, t0);
  t0 = _mulx_u64(c3, d0, (long long unsigned int *)&x4); ADCX(x3, t0);
  t0 = _mulx_u64(c4, d0, (long long unsigned int *)&x5); ADCX(x4, t0);
  t0 = _mulx_u64(c5, d0, (long long unsigned int *)&x6); ADCX(x5, t0);
                               ADCX(x6, zero);

  z1 = VMACLO(y0, a0, b1); z1 = VMACLO(z1, a1, b0);
  y1 = VMACHI(y1, a0, b1); y1 = VMACHI(y1, a1, b0);
  y1 = VSHL(y1, BALIGN);

  z2 = VMACLO(y1, a0, b2); z2 = VMACLO(z2, a1, b1); z2 = VMACLO(z2, a2, b0);
  y2 = VMACHI(y2, a0, b2); y2 = VMACHI(y2, a1, b1); y2 = VMACHI(y2, a2, b0);
  y2 = VSHL(y2, BALIGN);

  t0 = _mulx_u64(c0, d1, (long long unsigned int *)&t1); ADCX(x1, t0); ADOX(x2, t1);
  t0 = _mulx_u64(c1, d1, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c2, d1, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c3, d1, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c4, d1, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c5, d1, (long long unsigned int *)&x7); ADCX(x6, t0); ADOX(x7, zero);
                               ADCX(x7, zero);

  z3 = VMACLO(y2, a0, b3); z3 = VMACLO(z3, a1, b2); z3 = VMACLO(z3, a2, b1); 
  z3 = VMACLO(z3, a3, b0);
  y3 = VMACHI(y3, a0, b3); y3 = VMACHI(y3, a1, b2); y3 = VMACHI(y3, a2, b1); 
  y3 = VMACHI(y3, a3, b0);
  y3 = VSHL(y3, BALIGN);

  z4 = VMACLO(y3, a1, b3); z4 = VMACLO(z4, a2, b2); z4 = VMACLO(z4, a3, b1);
  y4 = VMACHI(y4, a1, b3); y4 = VMACHI(y4, a2, b2); y4 = VMACHI(y4, a3, b1);
  y4 = VSHL(y4, BALIGN);

  t0 = _mulx_u64(c0, d2, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c1, d2, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c2, d2, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c3, d2, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c4, d2, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c5, d2, (long long unsigned int *)&x8); ADCX(x7, t0); ADOX(x8, zero);
                               ADCX(x8, zero);

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

  t0 = _mulx_u64(c0, d3, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c1, d3, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c2, d3, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c3, d3, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c4, d3, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c5, d3, (long long unsigned int *)&x9); ADCX(x8, t0); ADOX(x9, zero);
                               ADCX(x9, zero);

  z9 = VMACLO(y8, a4, b5); z9 = VMACLO(z9, a5, b4);
  y9 = VMACHI(y9, a4, b5); y9 = VMACHI(y9, a5, b4);
  y9 = VSHL(y9, BALIGN);

  z10 = VMACLO(y9,  a4, b6); z10 = VMACLO(z10, a5, b5); 
  z10 = VMACLO(z10, a6, b4);
  y10 = VMACHI(y10, a4, b6); y10 = VMACHI(y10, a5, b5); 
  y10 = VMACHI(y10, a6, b4);
  y10 = VSHL(y10, BALIGN);

  t0 = _mulx_u64(c0, d4, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c1, d4, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c2, d4, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c3, d4, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c4, d4, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c5, d4, (long long unsigned int *)&x10); ADCX(x9, t0); ADOX(x10, zero);
                                ADCX(x10, zero);

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

  t0 = _mulx_u64(c0, d5, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c1, d5, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c2, d5, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c3, d5, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c4, d5, (long long unsigned int *)&t1); ADCX(x9, t0); ADOX(x10, t1);
  t0 = _mulx_u64(c5, d5, (long long unsigned int *)&x11); ADCX(x10, t0); ADOX(x11, zero);
                                ADCX(x11, zero);

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
  
  s[0 ] = x0 ; s[1 ] = x1 ; s[2 ] = x2 ; 
  s[3 ] = x3 ; s[4 ] = x4 ; s[5 ] = x5 ; 
  s[6 ] = x6 ; s[7 ] = x7 ; s[8 ] = x8 ; 
  s[9 ] = x9 ; s[10] = x10; s[11] = x11;
}

// Karatsuba (excl. carry prop.)
// vector subroutine + 2*scalar subroutines
void mul_fpx2_8x1w_hybrid_v2(fpx2_8x1w r, uint64_t *s, uint64_t *w, const fp_8x1w a, const fp_8x1w b, const uint64_t *c, const uint64_t *d, const uint64_t *u, const uint64_t *v)
{
  // vector variables
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

  // scalar variables
  uint64_t c0 = c[0], c1 = c[1], c2 = c[2];
  uint64_t c3 = c[3], c4 = c[4], c5 = c[5];
  uint64_t d0 = d[0], d1 = d[1], d2 = d[2];
  uint64_t d3 = d[3], d4 = d[4], d5 = d[5];
  uint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, t0, t1;
  uint8_t e0, e1;
  uint64_t zero = 0;

  // compute vector zL(z0-z7) by aL(a0-a3) * bL(b0-b4)
  // compute scalar x = c * d

  z0 = VMACLO(z0, a0, b0);
  y0 = VMACHI(y0, a0, b0);
  y0 = VSHL(y0, BALIGN);

  __asm__ volatile(
    "test %%eax, %%eax" 
    :::"eax","cc"
  );

  x0 = _mulx_u64(c0, d0, (long long unsigned int *)&x1);
  t0 = _mulx_u64(c1, d0, (long long unsigned int *)&x2); ADCX(x1, t0);
  t0 = _mulx_u64(c2, d0, (long long unsigned int *)&x3); ADCX(x2, t0);
  t0 = _mulx_u64(c3, d0, (long long unsigned int *)&x4); ADCX(x3, t0);
  t0 = _mulx_u64(c4, d0, (long long unsigned int *)&x5); ADCX(x4, t0);
  t0 = _mulx_u64(c5, d0, (long long unsigned int *)&x6); ADCX(x5, t0);
                               ADCX(x6, zero);

  z1 = VMACLO(y0, a0, b1); z1 = VMACLO(z1, a1, b0);
  y1 = VMACHI(y1, a0, b1); y1 = VMACHI(y1, a1, b0);
  y1 = VSHL(y1, BALIGN);

  z2 = VMACLO(y1, a0, b2); z2 = VMACLO(z2, a1, b1); z2 = VMACLO(z2, a2, b0);
  y2 = VMACHI(y2, a0, b2); y2 = VMACHI(y2, a1, b1); y2 = VMACHI(y2, a2, b0);
  y2 = VSHL(y2, BALIGN);

  t0 = _mulx_u64(c0, d1, (long long unsigned int *)&t1); ADCX(x1, t0); ADOX(x2, t1);
  t0 = _mulx_u64(c1, d1, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c2, d1, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c3, d1, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c4, d1, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c5, d1, (long long unsigned int *)&x7); ADCX(x6, t0); ADOX(x7, zero);
                               ADCX(x7, zero);

  z3 = VMACLO(y2, a0, b3); z3 = VMACLO(z3, a1, b2); z3 = VMACLO(z3, a2, b1); 
  z3 = VMACLO(z3, a3, b0);
  y3 = VMACHI(y3, a0, b3); y3 = VMACHI(y3, a1, b2); y3 = VMACHI(y3, a2, b1); 
  y3 = VMACHI(y3, a3, b0);
  y3 = VSHL(y3, BALIGN);

  z4 = VMACLO(y3, a1, b3); z4 = VMACLO(z4, a2, b2); z4 = VMACLO(z4, a3, b1);
  y4 = VMACHI(y4, a1, b3); y4 = VMACHI(y4, a2, b2); y4 = VMACHI(y4, a3, b1);
  y4 = VSHL(y4, BALIGN);

  t0 = _mulx_u64(c0, d2, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c1, d2, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c2, d2, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c3, d2, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c4, d2, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c5, d2, (long long unsigned int *)&x8); ADCX(x7, t0); ADOX(x8, zero);
                               ADCX(x8, zero);

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

  t0 = _mulx_u64(c0, d3, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c1, d3, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c2, d3, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c3, d3, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c4, d3, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c5, d3, (long long unsigned int *)&x9); ADCX(x8, t0); ADOX(x9, zero);
                               ADCX(x9, zero);

  z9 = VMACLO(y8, a4, b5); z9 = VMACLO(z9, a5, b4);
  y9 = VMACHI(y9, a4, b5); y9 = VMACHI(y9, a5, b4);
  y9 = VSHL(y9, BALIGN);

  z10 = VMACLO(y9,  a4, b6); z10 = VMACLO(z10, a5, b5); 
  z10 = VMACLO(z10, a6, b4);
  y10 = VMACHI(y10, a4, b6); y10 = VMACHI(y10, a5, b5); 
  y10 = VMACHI(y10, a6, b4);
  y10 = VSHL(y10, BALIGN);

  t0 = _mulx_u64(c0, d4, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c1, d4, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c2, d4, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c3, d4, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c4, d4, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c5, d4, (long long unsigned int *)&x10); ADCX(x9, t0); ADOX(x10, zero);
                                ADCX(x10, zero);

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

  t0 = _mulx_u64(c0, d5, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c1, d5, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c2, d5, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c3, d5, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c4, d5, (long long unsigned int *)&t1); ADCX(x9, t0); ADOX(x10, t1);
  t0 = _mulx_u64(c5, d5, (long long unsigned int *)&x11); ADCX(x10, t0); ADOX(x11, zero);
                                ADCX(x11, zero);

  s[0 ] = x0 ; s[1 ] = x1 ; s[2 ] = x2 ; 
  s[3 ] = x3 ; s[4 ] = x4 ; s[5 ] = x5 ; 
  s[6 ] = x6 ; s[7 ] = x7 ; s[8 ] = x8 ; 
  s[9 ] = x9 ; s[10] = x10; s[11] = x11;

  z13 = VMACLO(y12, a6, b7); z13 = VMACLO(z13, a7, b6);
  y13 = VMACHI(y13, a6, b7); y13 = VMACHI(y13, a7, b6);
  y13 = VSHL(y13, BALIGN);

  z14 = VMACLO(y13, a7, b7);
  y14 = VMACHI(y14, a7, b7);
  y14 = VSHL(y14, BALIGN);

  z15 = y14;

  c0 = u[0], c1 = u[1], c2 = u[2];
  c3 = u[3], c4 = u[4], c5 = u[5];
  d0 = v[0], d1 = v[1], d2 = v[2];
  d3 = v[3], d4 = v[4], d5 = v[5];

  __asm__ volatile(
    "test %%eax, %%eax" 
    :::"eax","cc"
  );

  x0 = _mulx_u64(c0, d0, (long long unsigned int *)&x1);
  t0 = _mulx_u64(c1, d0, (long long unsigned int *)&x2); ADCX(x1, t0);
  t0 = _mulx_u64(c2, d0, (long long unsigned int *)&x3); ADCX(x2, t0);
  t0 = _mulx_u64(c3, d0, (long long unsigned int *)&x4); ADCX(x3, t0);
  t0 = _mulx_u64(c4, d0, (long long unsigned int *)&x5); ADCX(x4, t0);
  t0 = _mulx_u64(c5, d0, (long long unsigned int *)&x6); ADCX(x5, t0);
                               ADCX(x6, zero);

  // ta(ta0-ta3) = aL(a0-a3) + aH(a4-a7)
  ta0 = VADD(a0, a4); ta1 = VADD(a1, a5);
  ta2 = VADD(a2, a6); ta3 = VADD(a3, a7);

  // tb(tb0-tb3) = bL(b0-b3) + bH(b4-b7)
  tb0 = VADD(b0, b4); tb1 = VADD(b1, b5); 
  tb2 = VADD(b2, b6); tb3 = VADD(b3, b7);

  t0 = _mulx_u64(c0, d1, (long long unsigned int *)&t1); ADCX(x1, t0); ADOX(x2, t1);
  t0 = _mulx_u64(c1, d1, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c2, d1, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c3, d1, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c4, d1, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c5, d1, (long long unsigned int *)&x7); ADCX(x6, t0); ADOX(x7, zero);
                               ADCX(x7, zero);

  // zM = ta * tb - zL - zH 
  
  y0 = y1 = y2 = y3 = y4 = y5 = y6 = y7 = VZERO;

  m0 = VMACLO(m0, ta0, tb0);
  y0 = VMACHI(y0, ta0, tb0);
  y0 = VSHL(y0, BALIGN);

  m1 = VMACLO(y0, ta0, tb1); m1 = VMACLO(m1, ta1, tb0);
  y1 = VMACHI(y1, ta0, tb1); y1 = VMACHI(y1, ta1, tb0);
  y1 = VSHL(y1, BALIGN);

  t0 = _mulx_u64(c0, d2, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c1, d2, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c2, d2, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c3, d2, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c4, d2, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c5, d2, (long long unsigned int *)&x8); ADCX(x7, t0); ADOX(x8, zero);
                               ADCX(x8, zero);

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

  t0 = _mulx_u64(c0, d3, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c1, d3, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c2, d3, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c3, d3, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c4, d3, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c5, d3, (long long unsigned int *)&x9); ADCX(x8, t0); ADOX(x9, zero);
                               ADCX(x9, zero);

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

  t0 = _mulx_u64(c0, d4, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c1, d4, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c2, d4, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c3, d4, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c4, d4, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c5, d4, (long long unsigned int *)&x10); ADCX(x9, t0); ADOX(x10, zero);
                                ADCX(x10, zero);

  m0 = VSUB(m0, VADD(z0, z8 )); m1 = VSUB(m1, VADD(z1, z9 ));
  m2 = VSUB(m2, VADD(z2, z10)); m3 = VSUB(m3, VADD(z3, z11));
  m4 = VSUB(m4, VADD(z4, z12)); m5 = VSUB(m5, VADD(z5, z13));
  m6 = VSUB(m6, VADD(z6, z14)); m7 = VSUB(m7, VADD(z7, z15));

  t0 = _mulx_u64(c0, d5, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c1, d5, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c2, d5, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c3, d5, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c4, d5, (long long unsigned int *)&t1); ADCX(x9, t0); ADOX(x10, t1);
  t0 = _mulx_u64(c5, d5, (long long unsigned int *)&x11); ADCX(x10, t0); ADOX(x11, zero);
                                ADCX(x11, zero);
                
  // z = z + zM
  z4  = VADD(z4 , m0); z5  = VADD(z5 , m1);
  z6  = VADD(z6 , m2); z7  = VADD(z7 , m3);
  z8  = VADD(z8 , m4); z9  = VADD(z9 , m5);
  z10 = VADD(z10, m6); z11 = VADD(z11, m7);

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;
  r[12] = z12; r[13] = z13; r[14] = z14; r[15] = z15;

  w[0 ] = x0 ; w[1 ] = x1 ; w[2 ] = x2 ; 
  w[3 ] = x3 ; w[4 ] = x4 ; w[5 ] = x5 ; 
  w[6 ] = x6 ; w[7 ] = x7 ; w[8 ] = x8 ; 
  w[9 ] = x9 ; w[10] = x10; w[11] = x11;
}

// vector subroutine
void mul_fpx2_4x2w(fpx2_4x2w r, const fp_4x2w a, const fp_4x2w b)
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

// vector subroutine + 1*scalar subroutine
void mul_fpx2_4x2w_hybrid_v1(fpx2_4x2w r, uint64_t *s, const fp_4x2w a, const fp_4x2w b, const uint64_t *c, const uint64_t *d)
{
  // vector variables
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  __m512i z0 = VZERO, z1 = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4 = VZERO, z5 = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8 = VZERO, z9 = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i y0 = VZERO, y1 = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4 = VZERO, y5 = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8 = VZERO, y9 = VZERO, y10 = VZERO, tb;

  // scalar variables
  const uint64_t c0 = c[0], c1 = c[1], c2 = c[2];
  const uint64_t c3 = c[3], c4 = c[4], c5 = c[5];
  const uint64_t d0 = d[0], d1 = d[1], d2 = d[2];
  const uint64_t d3 = d[3], d4 = d[4], d5 = d[5];
  uint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, t0, t1;
  uint8_t e0, e1;
  uint64_t zero = 0;

  tb = VSHUF(b0, 0x44);
  z0 = VMACLO(z0, tb, a0); z1 = VMACLO(z1, tb, a1);
  z2 = VMACLO(z2, tb, a2); z3 = VMACLO(z3, tb, a3);
  y0 = VMACHI(y0, tb, a0); y1 = VMACHI(y1, tb, a1);
  y2 = VMACHI(y2, tb, a2); y3 = VMACHI(y3, tb, a3);

  __asm__ volatile(
    "test %%eax, %%eax" 
    :::"eax","cc"
  );

  x0 = _mulx_u64(c0, d0, (long long unsigned int *)&x1);
  t0 = _mulx_u64(c1, d0, (long long unsigned int *)&x2); ADCX(x1, t0);
  t0 = _mulx_u64(c2, d0, (long long unsigned int *)&x3); ADCX(x2, t0);
  t0 = _mulx_u64(c3, d0, (long long unsigned int *)&x4); ADCX(x3, t0);
  t0 = _mulx_u64(c4, d0, (long long unsigned int *)&x5); ADCX(x4, t0);
  t0 = _mulx_u64(c5, d0, (long long unsigned int *)&x6); ADCX(x5, t0);
                               ADCX(x6, zero);

  tb = VSHUF(b1, 0x44);
  z1 = VMACLO(z1, tb, a0); z2 = VMACLO(z2, tb, a1);
  z3 = VMACLO(z3, tb, a2); z4 = VMACLO(z4, tb, a3);
  y1 = VMACHI(y1, tb, a0); y2 = VMACHI(y2, tb, a1);
  y3 = VMACHI(y3, tb, a2); y4 = VMACHI(y4, tb, a3);

  t0 = _mulx_u64(c0, d1, (long long unsigned int *)&t1); ADCX(x1, t0); ADOX(x2, t1);
  t0 = _mulx_u64(c1, d1, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c2, d1, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c3, d1, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c4, d1, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c5, d1, (long long unsigned int *)&x7); ADCX(x6, t0); ADOX(x7, zero);
                               ADCX(x7, zero);

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

  t0 = _mulx_u64(c0, d2, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c1, d2, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c2, d2, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c3, d2, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c4, d2, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c5, d2, (long long unsigned int *)&x8); ADCX(x7, t0); ADOX(x8, zero);
                               ADCX(x8, zero);

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

  t0 = _mulx_u64(c0, d3, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c1, d3, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c2, d3, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c3, d3, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c4, d3, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c5, d3, (long long unsigned int *)&x9); ADCX(x8, t0); ADOX(x9, zero);
                               ADCX(x9, zero);

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

  t0 = _mulx_u64(c0, d4, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c1, d4, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c2, d4, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c3, d4, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c4, d4, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c5, d4, (long long unsigned int *)&x10); ADCX(x9, t0); ADOX(x10, zero);
                                ADCX(x10, zero);


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

  t0 = _mulx_u64(c0, d5, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c1, d5, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c2, d5, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c3, d5, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c4, d5, (long long unsigned int *)&t1); ADCX(x9, t0); ADOX(x10, t1);
  t0 = _mulx_u64(c5, d5, (long long unsigned int *)&x11); ADCX(x10, t0); ADOX(x11, zero);
                                ADCX(x11, zero);

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;

  s[0 ] = x0 ; s[1 ] = x1 ; s[2 ] = x2 ; 
  s[3 ] = x3 ; s[4 ] = x4 ; s[5 ] = x5 ; 
  s[6 ] = x6 ; s[7 ] = x7 ; s[8 ] = x8 ; 
  s[9 ] = x9 ; s[10] = x10; s[11] = x11;
}

// vector subroutine + 2*scalar subroutines
void mul_fpx2_4x2w_hybrid_v2(fpx2_4x2w r, uint64_t *s, uint64_t *w, const fp_4x2w a, const fp_4x2w b, const uint64_t *c, const uint64_t *d, const uint64_t *u, const uint64_t *v)
{
  // vector variables
  const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  __m512i z0 = VZERO, z1 = VZERO, z2  = VZERO, z3  = VZERO;
  __m512i z4 = VZERO, z5 = VZERO, z6  = VZERO, z7  = VZERO;
  __m512i z8 = VZERO, z9 = VZERO, z10 = VZERO, z11 = VZERO;
  __m512i y0 = VZERO, y1 = VZERO, y2  = VZERO, y3  = VZERO;
  __m512i y4 = VZERO, y5 = VZERO, y6  = VZERO, y7  = VZERO;
  __m512i y8 = VZERO, y9 = VZERO, y10 = VZERO, tb;

  // scalar variables
  uint64_t c0 = c[0], c1 = c[1], c2 = c[2];
  uint64_t c3 = c[3], c4 = c[4], c5 = c[5];
  uint64_t d0 = d[0], d1 = d[1], d2 = d[2];
  uint64_t d3 = d[3], d4 = d[4], d5 = d[5];
  uint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, t0, t1;
  uint8_t e0, e1;
  uint64_t zero = 0;

  __asm__ volatile(
    "test %%eax, %%eax" 
    :::"eax","cc"
  );

  x0 = _mulx_u64(c0, d0, (long long unsigned int *)&x1);
  t0 = _mulx_u64(c1, d0, (long long unsigned int *)&x2); ADCX(x1, t0);
  t0 = _mulx_u64(c2, d0, (long long unsigned int *)&x3); ADCX(x2, t0);
  t0 = _mulx_u64(c3, d0, (long long unsigned int *)&x4); ADCX(x3, t0);
  t0 = _mulx_u64(c4, d0, (long long unsigned int *)&x5); ADCX(x4, t0);
  t0 = _mulx_u64(c5, d0, (long long unsigned int *)&x6); ADCX(x5, t0);
                               ADCX(x6, zero);

  tb = VSHUF(b0, 0x44);
  z0 = VMACLO(z0, tb, a0); z1 = VMACLO(z1, tb, a1);
  z2 = VMACLO(z2, tb, a2); z3 = VMACLO(z3, tb, a3);
  y0 = VMACHI(y0, tb, a0); y1 = VMACHI(y1, tb, a1);
  y2 = VMACHI(y2, tb, a2); y3 = VMACHI(y3, tb, a3);

  t0 = _mulx_u64(c0, d1, (long long unsigned int *)&t1); ADCX(x1, t0); ADOX(x2, t1);
  t0 = _mulx_u64(c1, d1, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c2, d1, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c3, d1, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c4, d1, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c5, d1, (long long unsigned int *)&x7); ADCX(x6, t0); ADOX(x7, zero);
                               ADCX(x7, zero);

  tb = VSHUF(b1, 0x44);
  z1 = VMACLO(z1, tb, a0); z2 = VMACLO(z2, tb, a1);
  z3 = VMACLO(z3, tb, a2); z4 = VMACLO(z4, tb, a3);
  y1 = VMACHI(y1, tb, a0); y2 = VMACHI(y2, tb, a1);
  y3 = VMACHI(y3, tb, a2); y4 = VMACHI(y4, tb, a3);

  t0 = _mulx_u64(c0, d2, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c1, d2, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c2, d2, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c3, d2, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c4, d2, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c5, d2, (long long unsigned int *)&x8); ADCX(x7, t0); ADOX(x8, zero);
                               ADCX(x8, zero);

  tb = VSHUF(b2, 0x44);
  z2 = VMACLO(z2, tb, a0); z3 = VMACLO(z3, tb, a1);
  z4 = VMACLO(z4, tb, a2); z5 = VMACLO(z5, tb, a3);
  y2 = VMACHI(y2, tb, a0); y3 = VMACHI(y3, tb, a1);
  y4 = VMACHI(y4, tb, a2); y5 = VMACHI(y5, tb, a3);

  t0 = _mulx_u64(c0, d3, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c1, d3, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c2, d3, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c3, d3, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c4, d3, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c5, d3, (long long unsigned int *)&x9); ADCX(x8, t0); ADOX(x9, zero);
                               ADCX(x9, zero);

  tb = VSHUF(b3, 0x44);
  z3 = VMACLO(z3, tb, a0); z4 = VMACLO(z4, tb, a1);
  z5 = VMACLO(z5, tb, a2); z6 = VMACLO(z6, tb, a3);
  y3 = VMACHI(y3, tb, a0); y4 = VMACHI(y4, tb, a1);
  y5 = VMACHI(y5, tb, a2); y6 = VMACHI(y6, tb, a3);

  t0 = _mulx_u64(c0, d4, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c1, d4, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c2, d4, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c3, d4, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c4, d4, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c5, d4, (long long unsigned int *)&x10); ADCX(x9, t0); ADOX(x10, zero);
                                ADCX(x10, zero);

  tb = VSHUF(b0, 0xEE);
  z4 = VMACLO(z4, tb, a0); z5 = VMACLO(z5, tb, a1);
  z6 = VMACLO(z6, tb, a2); z7 = VMACLO(z7, tb, a3);
  y4 = VMACHI(y4, tb, a0); y5 = VMACHI(y5, tb, a1);
  y6 = VMACHI(y6, tb, a2); y7 = VMACHI(y7, tb, a3);

  t0 = _mulx_u64(c0, d5, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c1, d5, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c2, d5, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c3, d5, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c4, d5, (long long unsigned int *)&t1); ADCX(x9, t0); ADOX(x10, t1);
  t0 = _mulx_u64(c5, d5, (long long unsigned int *)&x11); ADCX(x10, t0); ADOX(x11, zero);
                                ADCX(x11, zero);

  s[0 ] = x0 ; s[1 ] = x1 ; s[2 ] = x2 ; 
  s[3 ] = x3 ; s[4 ] = x4 ; s[5 ] = x5 ; 
  s[6 ] = x6 ; s[7 ] = x7 ; s[8 ] = x8 ; 
  s[9 ] = x9 ; s[10] = x10; s[11] = x11;

  tb = VSHUF(b1, 0xEE);
  z5 = VMACLO(z5, tb, a0); z6 = VMACLO(z6, tb, a1);
  z7 = VMACLO(z7, tb, a2); z8 = VMACLO(z8, tb, a3);
  y5 = VMACHI(y5, tb, a0); y6 = VMACHI(y6, tb, a1);
  y7 = VMACHI(y7, tb, a2); y8 = VMACHI(y8, tb, a3);

  c0 = u[0], c1 = u[1], c2 = u[2];
  c3 = u[3], c4 = u[4], c5 = u[5];
  d0 = v[0], d1 = v[1], d2 = v[2];
  d3 = v[3], d4 = v[4], d5 = v[5];

  tb = VSHUF(b2, 0xEE);
  z6 = VMACLO(z6, tb, a0); z7 = VMACLO(z7, tb, a1);
  z8 = VMACLO(z8, tb, a2); z9 = VMACLO(z9, tb, a3);
  y6 = VMACHI(y6, tb, a0); y7 = VMACHI(y7, tb, a1);
  y8 = VMACHI(y8, tb, a2); y9 = VMACHI(y9, tb, a3);

  __asm__ volatile(
    "test %%eax, %%eax" 
    :::"eax","cc"
  );

  x0 = _mulx_u64(c0, d0, (long long unsigned int *)&x1);
  t0 = _mulx_u64(c1, d0, (long long unsigned int *)&x2); ADCX(x1, t0);
  t0 = _mulx_u64(c2, d0, (long long unsigned int *)&x3); ADCX(x2, t0);
  t0 = _mulx_u64(c3, d0, (long long unsigned int *)&x4); ADCX(x3, t0);
  t0 = _mulx_u64(c4, d0, (long long unsigned int *)&x5); ADCX(x4, t0);
  t0 = _mulx_u64(c5, d0, (long long unsigned int *)&x6); ADCX(x5, t0);
                               ADCX(x6, zero);

  tb = VSHUF(b3, 0xEE);
  z7 = VMACLO(z7, tb, a0); z8 = VMACLO(z8, tb, a1);
  z9 = VMACLO(z9, tb, a2); z10 = VMACLO(z10, tb, a3);
  y7 = VMACHI(y7, tb, a0); y8 = VMACHI(y8, tb, a1);
  y9 = VMACHI(y9, tb, a2); y10 = VMACHI(y10, tb, a3);

  t0 = _mulx_u64(c0, d1, (long long unsigned int *)&t1); ADCX(x1, t0); ADOX(x2, t1);
  t0 = _mulx_u64(c1, d1, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c2, d1, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c3, d1, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c4, d1, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c5, d1, (long long unsigned int *)&x7); ADCX(x6, t0); ADOX(x7, zero);
                               ADCX(x7, zero);

  z1  = VADD(z1 , VSHL(y0 , BALIGN));
  z2  = VADD(z2 , VSHL(y1 , BALIGN));
  z3  = VADD(z3 , VSHL(y2 , BALIGN));
  z4  = VADD(z4 , VSHL(y3 , BALIGN));

  t0 = _mulx_u64(c0, d2, (long long unsigned int *)&t1); ADCX(x2, t0); ADOX(x3, t1);
  t0 = _mulx_u64(c1, d2, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c2, d2, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c3, d2, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c4, d2, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c5, d2, (long long unsigned int *)&x8); ADCX(x7, t0); ADOX(x8, zero);
                               ADCX(x8, zero);

  z5  = VADD(z5 , VSHL(y4 , BALIGN));
  z6  = VADD(z6 , VSHL(y5 , BALIGN));
  z7  = VADD(z7 , VSHL(y6 , BALIGN));
  z8  = VADD(z8 , VSHL(y7 , BALIGN));

  t0 = _mulx_u64(c0, d3, (long long unsigned int *)&t1); ADCX(x3, t0); ADOX(x4, t1);
  t0 = _mulx_u64(c1, d3, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c2, d3, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c3, d3, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c4, d3, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c5, d3, (long long unsigned int *)&x9); ADCX(x8, t0); ADOX(x9, zero);
                               ADCX(x9, zero);

  z9  = VADD(z9 , VSHL(y8 , BALIGN));
  z10 = VADD(z10, VSHL(y9 , BALIGN));
  z11 = VADD(z11, VSHL(y10, BALIGN));

  r[0 ] = z0 ; r[1 ] = z1 ; r[2 ] = z2 ; r[3 ] = z3 ;
  r[4 ] = z4 ; r[5 ] = z5 ; r[6 ] = z6 ; r[7 ] = z7 ;
  r[8 ] = z8 ; r[9 ] = z9 ; r[10] = z10; r[11] = z11;

  t0 = _mulx_u64(c0, d4, (long long unsigned int *)&t1); ADCX(x4, t0); ADOX(x5, t1);
  t0 = _mulx_u64(c1, d4, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c2, d4, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c3, d4, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c4, d4, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c5, d4, (long long unsigned int *)&x10); ADCX(x9, t0); ADOX(x10, zero);
                                ADCX(x10, zero);

  t0 = _mulx_u64(c0, d5, (long long unsigned int *)&t1); ADCX(x5, t0); ADOX(x6, t1);
  t0 = _mulx_u64(c1, d5, (long long unsigned int *)&t1); ADCX(x6, t0); ADOX(x7, t1);
  t0 = _mulx_u64(c2, d5, (long long unsigned int *)&t1); ADCX(x7, t0); ADOX(x8, t1);
  t0 = _mulx_u64(c3, d5, (long long unsigned int *)&t1); ADCX(x8, t0); ADOX(x9, t1);
  t0 = _mulx_u64(c4, d5, (long long unsigned int *)&t1); ADCX(x9, t0); ADOX(x10, t1);
  t0 = _mulx_u64(c5, d5, (long long unsigned int *)&x11); ADCX(x10, t0); ADOX(x11, zero);
                                ADCX(x11, zero);

  w[0 ] = x0 ; w[1 ] = x1 ; w[2 ] = x2 ; 
  w[3 ] = x3 ; w[4 ] = x4 ; w[5 ] = x5 ; 
  w[6 ] = x6 ; w[7 ] = x7 ; w[8 ] = x8 ; 
  w[9 ] = x9 ; w[10] = x10; w[11] = x11;
}

// ----------------------------------------------------------------------------

void carryp_dl_8x1w(fpx2_8x1w z)
{
  __m512i z0 = z[0], z1 = z[1], z2 = z[2], z3 = z[3];
  __m512i z4 = z[4], z5 = z[5], z6 = z[6], z7 = z[7];
  __m512i z8 = z[8], z9 = z[9], z10 = z[10], z11 = z[11];
  __m512i z12 = z[12], z13 = z[13], z14 = z[14], z15 = z[15];
  const __m512i bmask = VSET1(BMASK);

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

  z[0 ] = z0 ; z[1 ] = z1 ; z[2 ] = z2 ; z[3 ] = z3 ;
  z[4 ] = z4 ; z[5 ] = z5 ; z[6 ] = z6 ; z[7 ] = z7 ;
  z[8 ] = z8 ; z[9 ] = z9 ; z[10] = z10; z[11] = z11;
  z[12] = z12; z[13] = z13; z[14] = z14; z[15] = z15; 
}

// ----------------------------------------------------------------------------

// for measuring CPU cycles

extern uint64_t read_tsc();


#define LOAD_CACHE(X, ITER) for (i = 0; i < (ITER); i++) (X)

#define MEASURE_CYCLES(X, ITER)                    \
  start_cycles = read_tsc();                       \
  for (i = 0; i < (ITER); i++) (X);                \
  end_cycles = read_tsc();                         \
  diff_cycles = (end_cycles-start_cycles)/(ITER)

// ----------------------------------------------------------------------------

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

static void get_channel_dl_8x1w(uint64_t *r, const __m512i *a, const int ch) 
{
  int i;

  for(i = 0; i < 2*NWORDS; i++) {
    r[i] = ((uint64_t *)&a[i])[ch];
  }
}

static void mpi_print(const char *c, const uint64_t *a, int len)
{
  int i;

  printf("%s", c);
  for (i = len-1; i > 0; i--) printf("%016lX", a[i]);
  printf("%016lX\n", a[0]);
}

// ----------------------------------------------------------------------------
// test vectors

// a := 0x1123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF;
#define TV_A \
0x0123456789ABCDEF, 0x0123456789ABCDEF, 0x0123456789ABCDEF, \
0x0123456789ABCDEF, 0x0123456789ABCDEF, 0x1123456789ABCDEF

// b := 0x19CDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789;
#define TV_B \
0xABCDEF0123456789, 0xABCDEF0123456789, 0xABCDEF0123456789, \
0xABCDEF0123456789, 0xABCDEF0123456789, 0x19CDEF0123456789

// ----------------------------------------------------------------------------

int main()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, z64[2*SWORDS] = {0};
  uint64_t a48[NWORDS] = {0}, b48[NWORDS] = {0}, z48[2*NWORDS] = {0};
  __m512i a_8x1w[NWORDS], b_8x1w[NWORDS], z_8x1w[2*NWORDS];
  __m512i a_4x2w[VWORDS], b_4x2w[VWORDS], z_4x2w[3*VWORDS];
  vec768 s, w;
  vec384 c = {TV_A}, d = {TV_B};
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

  conv_64to48_mpi(a48, a64, NWORDS, SWORDS);
  conv_64to48_mpi(b48, b64, NWORDS, SWORDS);

  for (i = 0; i < NWORDS; i++) {
    a_8x1w[i] = VSET1(a48[i]);
    b_8x1w[i] = VSET1(b48[i]);
  }

  for (i = 0; i < VWORDS; i++) {
    a_4x2w[i] = VSET(0, 0, 0, 0, 0, 0, a48[i+VWORDS], a48[i]);
    b_4x2w[i] = VSET(0, 0, 0, 0, 0, 0, b48[i+VWORDS], b48[i]);
  }

#if 0

  mul_fpx2_8x1w(z_8x1w, a_8x1w, b_8x1w);
  carryp_dl_8x1w(z_8x1w);
  get_channel_dl_8x1w(z48, z_8x1w, 0);
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* vector result   = 0x", z64, 2*SWORDS);

  mul_fpx2_8x1w_hybrid_v1(z_8x1w, s, a_8x1w, b_8x1w, c, d);
  carryp_dl_8x1w(z_8x1w);
  get_channel_dl_8x1w(z48, z_8x1w, 0);
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* vector result   = 0x", z64, 2*SWORDS);
  mpi_print("* scalar result 1 = 0x", s, 2*SWORDS);

  mul_fpx2_8x1w_hybrid_v2(z_8x1w, s, w, a_8x1w, b_8x1w, c, d, c, d);
  carryp_dl_8x1w(z_8x1w);
  get_channel_dl_8x1w(z48, z_8x1w, 0);
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* vector result   = 0x", z64, 2*SWORDS);
  mpi_print("* scalar result 1 = 0x", s, 2*SWORDS);
  mpi_print("* scalar result 2 = 0x", w, 2*SWORDS);

  mul_fpx2_4x2w(z_4x2w, a_4x2w, b_4x2w);
  
  mul_fpx2_4x2w_hybrid_v1(z_4x2w, s, a_4x2w, b_4x2w, c, d);
  mpi_print("* scalar result 1 = 0x", s, 2*SWORDS);

  mul_fpx2_4x2w_hybrid_v2(z_4x2w, s, w, a_4x2w, b_4x2w, c, d, c, d);
  mpi_print("* scalar result 1 = 0x", s, 2*SWORDS);
  mpi_print("* scalar result 2 = 0x", w, 2*SWORDS);
#endif

  printf("- mul_fpx2_8x1w:               ");
  LOAD_CACHE(mul_fpx2_8x1w(z_8x1w, z_8x1w, z_8x1w), 10000);  
  MEASURE_CYCLES(mul_fpx2_8x1w(z_8x1w, z_8x1w, z_8x1w), 100000);
  printf("#cycle/inst = %.2f\n", (double) diff_cycles/8);

  printf("- mul_fpx2_8x1w_hybrid_v1:     ");
  LOAD_CACHE(mul_fpx2_8x1w_hybrid_v1(z_8x1w, s, a_8x1w, b_8x1w, c, d), 10000);
  MEASURE_CYCLES(mul_fpx2_8x1w_hybrid_v1(z_8x1w, s, a_8x1w, b_8x1w, c, d), 100000);
  printf("#cycle/inst = %.2f\n", (double) diff_cycles/9);

  printf("- mul_fpx2_8x1w_hybrid_v2:     ");
  LOAD_CACHE(mul_fpx2_8x1w_hybrid_v2(z_8x1w, s, w, a_8x1w, b_8x1w, c, d, c, d), 10000);
  MEASURE_CYCLES(mul_fpx2_8x1w_hybrid_v2(z_8x1w, s, w, a_8x1w, b_8x1w, c, d, c, d), 100000);
  printf("#cycle/inst = %.2f\n", (double) diff_cycles/10);

  printf("- mul_fpx2_4x2w:               ");
  LOAD_CACHE(mul_fpx2_4x2w(z_4x2w, z_4x2w, z_4x2w), 10000);
  MEASURE_CYCLES(mul_fpx2_4x2w(z_4x2w, z_4x2w, z_4x2w), 100000);
  printf("#cycle/inst = %.2f\n", (double) diff_cycles/4);

  printf("- mul_fpx2_4x2w_hybrid_v1:     ");
  LOAD_CACHE(mul_fpx2_4x2w_hybrid_v1(z_4x2w, s, a_4x2w, b_4x2w, c, d), 10000);
  MEASURE_CYCLES(mul_fpx2_4x2w_hybrid_v1(z_4x2w, s, a_4x2w, b_4x2w, c, d), 100000);
  printf("#cycle/inst = %.2f\n", (double) diff_cycles/5);

  printf("- mul_fpx2_4x2w_hybrid_v2:     ");
  LOAD_CACHE(mul_fpx2_4x2w_hybrid_v2(z_4x2w, s, w, a_4x2w, b_4x2w, c, d, c, d), 10000);
  MEASURE_CYCLES(mul_fpx2_4x2w_hybrid_v2(z_4x2w, s, w, a_4x2w, b_4x2w, c, d, c, d), 100000);
  printf("#cycle/inst = %.2f\n", (double) diff_cycles/6);

  return 0;
}
