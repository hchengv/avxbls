#include "vect.h"
#include "fields.h"
#include "consts.h"

#define INTRINSICS 0

#define MULX(hi, lo, a, b)                    \
{                                             \
  __uint128_t t = (__uint128_t)a * b;         \
  *hi = t >> 64;                              \
  lo  = (uint64_t)t;                          \
}

#define ADCX(c_out, out, c_in, a, b)          \
{                                             \
  __uint128_t t = (__uint128_t)a + b + c_in;  \
  *out  = (uint64_t)t;                        \
  c_out = (t >> 64) & 1;                      \
}

#define SUBB(c_out, out, c_in, a, b)          \
{                                             \
  __uint128_t t = (__uint128_t)a - b - c_in;  \
  *out  = (uint64_t)t;                        \
  c_out = (t >> 64) & 1;                      \
}

// fp arithmetic 

// ret = a - b mod p 
void sub_mod_384(vec384 ret, const vec384 a, const vec384 b, const vec384 p)
{
  uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
  uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5];
  uint64_t p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3], p4 = p[4], p5 = p[5];
  uint64_t r0, r1, r2, r3, r4, r5;
  uint64_t m;
  uint8_t c;

  // r = a - b
  SUBB(c, &r0, 0, a0, b0);
  SUBB(c, &r1, c, a1, b1);
  SUBB(c, &r2, c, a2, b2);
  SUBB(c, &r3, c, a3, b3);
  SUBB(c, &r4, c, a4, b4);
  SUBB(c, &r5, c, a5, b5);

  // m = 0 - c
  m = 0 - (uint64_t)c;

  // p = p & m
  p0 &= m;
  p1 &= m;
  p2 &= m;
  p3 &= m;
  p4 &= m;
  p5 &= m;

  // r = r + p
  ADCX(c, &r0, 0, r0, p0);
  ADCX(c, &r1, c, r1, p1);
  ADCX(c, &r2, c, r2, p2);
  ADCX(c, &r3, c, r3, p3);
  ADCX(c, &r4, c, r4, p4);
  ADCX(c, &r5, c, r5, p5);

  ret[0] = r0; ret[1] = r1; ret[2] = r2;
  ret[3] = r3; ret[4] = r4; ret[5] = r5;
}

// if flag == 1: ret = p - a
// if flag == 0: ret = a 
void cneg_mod_384(vec384 ret, const vec384 a, bool_t flag, const vec384 p)
{
  uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
  uint64_t p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3], p4 = p[4], p5 = p[5];
  uint64_t r0, r1, r2, r3, r4, r5;
  const uint64_t m = 0 - (uint64_t)(flag & 1);
  uint8_t c;

  // r = p - a
  SUBB(c, &r0, 0, p0, a0);
  SUBB(c, &r1, c, p1, a1);
  SUBB(c, &r2, c, p2, a2);
  SUBB(c, &r3, c, p3, a3);
  SUBB(c, &r4, c, p4, a4);
  SUBB(c, &r5, c, p5, a5);

  // r = r ^ a
  r0 ^= a0;
  r1 ^= a1;
  r2 ^= a2;
  r3 ^= a3;
  r4 ^= a4;
  r5 ^= a5;

  // r = r & m
  r0 &= m;
  r1 &= m;
  r2 &= m;
  r3 &= m;
  r4 &= m;
  r5 &= m;

  // r = r ^ a
  r0 ^= a0;
  r1 ^= a1;
  r2 ^= a2;
  r3 ^= a3;
  r4 ^= a4;
  r5 ^= a5;

  ret[0] = r0; ret[1] = r1; ret[2] = r2;
  ret[3] = r3; ret[4] = r4; ret[5] = r5;
}

// ret = (a << count) mod p
void lshift_mod_384(vec384 ret, const vec384 a, size_t count, const vec384 p)
{
  vec_copy(ret, a, sizeof(vec384));

  while (count--) {
    add_mod_384(ret, ret, ret, p);
  }
}

// ret = a * 8 mod p
void mul_by_8_mod_384(vec384 ret, const vec384 a, const vec384 p)
{
  vec384 t;

  add_mod_384(t, a, a, p);
  add_mod_384(t, t, t, p);
  add_mod_384(ret, t, t, p);
}

// ret = a * 3 mod p
void mul_by_3_mod_384(vec384 ret, const vec384 a, const vec384 p)
{
  vec384 t;

  add_mod_384(t, a, a, p);
  add_mod_384(ret, t, a, p);
}

// ret = a * b * 2^-384 mod p
// coarsely integrated operand-scanning (CIOS)
void mul_mont_384(vec384 ret, const vec384 a, const vec384 b, const vec384 p, limb_t n0)
{
  uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
  uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5];
  uint64_t p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3], p4 = p[4], p5 = p[5];
  uint64_t z0, z1, z2, z3, z4, z5, z6;
  uint64_t t0, t1, u;
  uint8_t c0, c1;

  // z = a0 * b
  MULX(&z1, z0, a0, b0);
  MULX(&z2, t0, a0, b1); ADCX(c0, &z1,  0, z1, t0);
  MULX(&z3, t0, a0, b2); ADCX(c0, &z2, c0, z2, t0);
  MULX(&z4, t0, a0, b3); ADCX(c0, &z3, c0, z3, t0);
  MULX(&z5, t0, a0, b4); ADCX(c0, &z4, c0, z4, t0);
  MULX(&z6, t0, a0, b5); ADCX(c0, &z5, c0, z5, t0);
                         z6 = z6 + c0;
  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; z6 = 0;

  // z = a1 * b + z
  MULX(&t1, t0, a1, b0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0, a1, b1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0, a1, b2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0, a1, b3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0, a1, b4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0, a1, b5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z 
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; z6 = 0;

  // z = a2 * b + z
  MULX(&t1, t0, a2, b0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0, a2, b1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0, a2, b2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0, a2, b3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0, a2, b4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0, a2, b5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z 
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; z6 = 0;

  // z = a3 * b + z
  MULX(&t1, t0, a3, b0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0, a3, b1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0, a3, b2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0, a3, b3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0, a3, b4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0, a3, b5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z 
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; z6 = 0;

  // z = a4 * b + z
  MULX(&t1, t0, a4, b0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0, a4, b1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0, a4, b2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0, a4, b3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0, a4, b4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0, a4, b5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z 
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; z6 = 0;

  // z = a5 * b + z
  MULX(&t1, t0, a5, b0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0, a5, b1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0, a5, b2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0, a5, b3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0, a5, b4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0, a5, b5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z 
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         z6 = z6 + c0;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; z6 = 0;

  // final correction
  SUBB(c0, &z0,  0, z0, p0);
  SUBB(c0, &z1, c0, z1, p1);
  SUBB(c0, &z2, c0, z2, p2);
  SUBB(c0, &z3, c0, z3, p3);
  SUBB(c0, &z4, c0, z4, p4);
  SUBB(c0, &z5, c0, z5, p5);
  t0 = 0 - (uint64_t)c0;
  p0 &= t0; 
  p1 &= t0; 
  p2 &= t0; 
  p3 &= t0;
  p4 &= t0;
  p5 &= t0;
  ADCX(c0, &z0,  0, z0, p0);
  ADCX(c0, &z1, c0, z1, p1);
  ADCX(c0, &z2, c0, z2, p2);
  ADCX(c0, &z3, c0, z3, p3);
  ADCX(c0, &z4, c0, z4, p4);
  ADCX(c0, &z5, c0, z5, p5);

  ret[0] = z0; ret[1] = z1; ret[2] = z2;
  ret[3] = z3; ret[4] = z4; ret[5] = z5; 
}

// ret = a^2 * 2^-384 mod p
void sqr_mont_384(vec384 ret, const vec384 a, const vec384 p, limb_t n0)
{
  mul_mont_384(ret, a, a, p, n0);
}

// ret = a * 2^-384 mod p
void redc_mont_384(vec384 ret, const vec768 a, const vec384 p, limb_t n0)
{
  uint64_t z0 = a[0], z1 = a[1], z2 = a[2], z3 = a[3], z4  = a[4],  z5  = a[5];
  uint64_t z6 = a[6], z7 = a[7], z8 = a[8], z9 = a[9], z10 = a[10], z11 = a[11];
  uint64_t p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3], p4  = p[4],  p5  = p[5];
  uint64_t t0, t1, u;
  uint8_t c0, c1, c2, c3;

  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c2, &z5, c0, z5, t0); ADCX(c3, &z6, c1, z6, t1);
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4;  z4  = z5;  z5 = z6; 
  z6 = z7; z7 = z8; z8 = z9; z9 = z10; z10 = z11; 

  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         ADCX(c2, &z5, c2, z5,  0); ADCX(c3, &z6, c3, z6,  0);
                         c2 = c0 | c2;              c3 = c1 | c3;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4;  z4  = z5; z5 = z6; 
  z6 = z7; z7 = z8; z8 = z9; z9 = z10; 

  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         ADCX(c2, &z5, c2, z5,  0); ADCX(c3, &z6, c3, z6,  0);
                         c2 = c0 | c2;              c3 = c1 | c3;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; 
  z6 = z7; z7 = z8; z8 = z9; 

  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         ADCX(c2, &z5, c2, z5,  0); ADCX(c3, &z6, c3, z6,  0);
                         c2 = c0 | c2;              c3 = c1 | c3;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; 
  z6 = z7; z7 = z8; 

  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         ADCX(c2, &z5, c2, z5,  0); ADCX(c3, &z6, c3, z6,  0);
                         c2 = c0 | c2;              c3 = c1 | c3;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; 
  z6 = z7;

  // u = (z0 * n0) mod 2^64
  MULX(&t1,  u, z0, n0);
  // z = u * p + z
  MULX(&t1, t0,  u, p0); ADCX(c0, &z0,  0, z0, t0); ADCX(c1, &z1,  0, z1, t1);
  MULX(&t1, t0,  u, p1); ADCX(c0, &z1, c0, z1, t0); ADCX(c1, &z2, c1, z2, t1);
  MULX(&t1, t0,  u, p2); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0,  u, p3); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0,  u, p4); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0,  u, p5); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
                         ADCX(c2, &z5, c2, z5,  0); ADCX(c3, &z6, c3, z6,  0);
                         c2 = c2 | c0;
                         z6 = z6 + c2;
  // z = z / 2^64
  z0 = z1; z1 = z2; z2 = z3; z3 = z4; z4 = z5; z5 = z6; 

  // final correction
  SUBB(c0, &z0,  0, z0, p0);
  SUBB(c0, &z1, c0, z1, p1);
  SUBB(c0, &z2, c0, z2, p2);
  SUBB(c0, &z3, c0, z3, p3);
  SUBB(c0, &z4, c0, z4, p4);
  SUBB(c0, &z5, c0, z5, p5);
  t0 = 0 - (uint64_t)c0;
  p0 &= t0;
  p1 &= t0; 
  p2 &= t0; 
  p3 &= t0;
  p4 &= t0;
  p5 &= t0;
  ADCX(c0, &z0,  0, z0, p0);
  ADCX(c0, &z1, c0, z1, p1);
  ADCX(c0, &z2, c0, z2, p2);
  ADCX(c0, &z3, c0, z3, p3);
  ADCX(c0, &z4, c0, z4, p4);
  ADCX(c0, &z5, c0, z5, p5);

  ret[0] = z0; ret[1] = z1; ret[2] = z2;
  ret[3] = z3; ret[4] = z4; ret[5] = z5; 
}

void mul_384(vec768 ret, const vec384 a, const vec384 b)
{
  uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
  uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5];
  uint64_t z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11;
  uint64_t t0, t1;
  uint8_t c0, c1;

  // z = a0 * b
  MULX(&z1, z0, a0, b0);
  MULX(&z2, t0, a0, b1); ADCX(c0, &z1,  0, z1, t0);
  MULX(&z3, t0, a0, b2); ADCX(c0, &z2, c0, z2, t0);
  MULX(&z4, t0, a0, b3); ADCX(c0, &z3, c0, z3, t0);
  MULX(&z5, t0, a0, b4); ADCX(c0, &z4, c0, z4, t0);
  MULX(&z6, t0, a0, b5); ADCX(c0, &z5, c0, z5, t0);
                         z6 = z6 + c0;

  // z = a1 * b + z
  MULX(&t1, t0, a1, b0); ADCX(c0, &z1,  0, z1, t0); ADCX(c1, &z2,  0, z2, t1);
  MULX(&t1, t0, a1, b1); ADCX(c0, &z2, c0, z2, t0); ADCX(c1, &z3, c1, z3, t1);
  MULX(&t1, t0, a1, b2); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0, a1, b3); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0, a1, b4); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
  MULX(&t1, t0, a1, b5); ADCX(c0, &z6, c0, z6, t0); ADCX(c1, &z7, c1, z7, t1);
                         z7 = z7 + c0;
  
  // z = a2 * b + z
  MULX(&t1, t0, a2, b0); ADCX(c0, &z2,  0, z2, t0); ADCX(c1, &z3,  0, z3, t1);
  MULX(&t1, t0, a2, b1); ADCX(c0, &z3, c0, z3, t0); ADCX(c1, &z4, c1, z4, t1);
  MULX(&t1, t0, a2, b2); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0, a2, b3); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
  MULX(&t1, t0, a2, b4); ADCX(c0, &z6, c0, z6, t0); ADCX(c1, &z7, c1, z7, t1);
  MULX(&t1, t0, a2, b5); ADCX(c0, &z7, c0, z7, t0); ADCX(c1, &z8, c1, z8, t1);
                         z8 = z8 + c0;
  
  // z = a3 * b + z
  MULX(&t1, t0, a3, b0); ADCX(c0, &z3,  0, z3, t0); ADCX(c1, &z4,  0, z4, t1);
  MULX(&t1, t0, a3, b1); ADCX(c0, &z4, c0, z4, t0); ADCX(c1, &z5, c1, z5, t1);
  MULX(&t1, t0, a3, b2); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6, c1, z6, t1);
  MULX(&t1, t0, a3, b3); ADCX(c0, &z6, c0, z6, t0); ADCX(c1, &z7, c1, z7, t1);
  MULX(&t1, t0, a3, b4); ADCX(c0, &z7, c0, z7, t0); ADCX(c1, &z8, c1, z8, t1);
  MULX(&t1, t0, a3, b5); ADCX(c0, &z8, c0, z8, t0); ADCX(c1, &z9, c1, z9, t1);
                         z9 = z9 + c0;

  // z = a4 * b + z
  MULX(&t1, t0, a4, b0); ADCX(c0, &z4,  0, z4, t0); ADCX(c1, &z5,   0, z5,  t1);
  MULX(&t1, t0, a4, b1); ADCX(c0, &z5, c0, z5, t0); ADCX(c1, &z6,  c1, z6,  t1);
  MULX(&t1, t0, a4, b2); ADCX(c0, &z6, c0, z6, t0); ADCX(c1, &z7,  c1, z7,  t1);
  MULX(&t1, t0, a4, b3); ADCX(c0, &z7, c0, z7, t0); ADCX(c1, &z8,  c1, z8,  t1);
  MULX(&t1, t0, a4, b4); ADCX(c0, &z8, c0, z8, t0); ADCX(c1, &z9,  c1, z9,  t1);
  MULX(&t1, t0, a4, b5); ADCX(c0, &z9, c0, z9, t0); ADCX(c1, &z10, c1, z10, t1);
                         z10 = z10 + c0;

  // z = a5 * b + z
  MULX(&t1, t0, a5, b0); ADCX(c0, &z5,   0, z5,  t0); ADCX(c1, &z6,   0, z6,  t1);
  MULX(&t1, t0, a5, b1); ADCX(c0, &z6,  c0, z6,  t0); ADCX(c1, &z7,  c1, z7,  t1);
  MULX(&t1, t0, a5, b2); ADCX(c0, &z7,  c0, z7,  t0); ADCX(c1, &z8,  c1, z8,  t1);
  MULX(&t1, t0, a5, b3); ADCX(c0, &z8,  c0, z8,  t0); ADCX(c1, &z9,  c1, z9,  t1);
  MULX(&t1, t0, a5, b4); ADCX(c0, &z9,  c0, z9,  t0); ADCX(c1, &z10, c1, z10, t1);
  MULX(&t1, t0, a5, b5); ADCX(c0, &z10, c0, z10, t0); ADCX(c1, &z11, c1, z11, t1);
                         z11 = z11 + c0;

  ret[0] = z0; ret[1]  = z1;  ret[2]  = z2;
  ret[3] = z3; ret[4]  = z4;  ret[5]  = z5;
  ret[6] = z6; ret[7]  = z7;  ret[8]  = z8;
  ret[9] = z9; ret[10] = z10; ret[11] = z11;
} 

void flt_inverse_mont_384(vec384 ret, const vec384 inp, const vec384 p, limb_t n0)
{
  vec384 r = { ONE_MONT_P }, a, p_minus_2, two = { 2 };
  uint64_t t;
  int i, j;

  vec_copy(a, inp, sizeof(vec384));
  sub_mod_384(p_minus_2, p, two, p);

  for (i = 0; i < NLIMBS(384); i++) {
    t = p_minus_2[i];
    for (j = 0; j < LIMB_T_BITS; j++, t >>= 1) {
      if (t & 1) mul_mont_384(r, r, a, p, n0);
      sqr_mont_384(a, a, p, n0);
    }
  }

  vec_copy(ret, r, sizeof(vec384));
}

// fp2 arithmetic 

// Karatsuba
void mul_mont_384x(vec384x ret, const vec384x a, const vec384x b, const vec384 p, limb_t n0)
{
  vec768x t;

  mul_fp2x2(t, a, b);
  redc_mont_384(ret[0], t[0], p, n0);
  redc_mont_384(ret[1], t[1], p, n0);
}

// Karatsuba
void sqr_mont_384x(vec384x ret, const vec384x a, const vec384 p, limb_t n0)
{
  vec768x t;
  
  sqr_fp2x2(t, a);
  redc_mont_384(ret[0], t[0], p, n0);
  redc_mont_384(ret[1], t[1], p, n0);
}

void add_mod_384x(vec384x ret, const vec384x a, const vec384x b, const vec384 p)
{
  add_mod_384(ret[0], a[0], b[0], p);
  add_mod_384(ret[1], a[1], b[1], p);
}

void sub_mod_384x(vec384x ret, const vec384x a, const vec384x b, const vec384 p)
{  
  sub_mod_384(ret[0], a[0], b[0], p);
  sub_mod_384(ret[1], a[1], b[1], p);
}

void mul_by_8_mod_384x(vec384x ret, const vec384x a, const vec384 p)
{
  mul_by_8_mod_384(ret[0], a[0], p);
  mul_by_8_mod_384(ret[1], a[1], p);
}

void mul_by_3_mod_384x(vec384x ret, const vec384x a, const vec384 p)
{
  mul_by_3_mod_384(ret[0], a[0], p);
  mul_by_3_mod_384(ret[1], a[1], p);
}

void mul_by_1_plus_i_mod_384x(vec384x ret, const vec384x a, const vec384 p)
{
  vec384 r0;

  sub_mod_384(r0, a[0], a[1], p);
  add_mod_384(ret[1], a[0], a[1], p);

  vec_copy(ret[0], r0, sizeof(vec384));
}

// double-length ret = a + b mod (p * 2^384)
void add_mod_384x384(vec768 ret, const vec768 a, const vec768 b, const vec384 p)
{
  uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4  = a[4],  a5  = a[5];
  uint64_t a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11];
  uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4  = b[4],  b5  = b[5];
  uint64_t b6 = b[6], b7 = b[7], b8 = b[8], b9 = b[9], b10 = b[10], b11 = b[11];
  uint64_t p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3], p4  = p[4],  p5  = p[5];
  uint64_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
  uint64_t m;
  uint8_t c;

  // r = a + b
  ADCX(c, &r0,  0, a0,  b0);
  ADCX(c, &r1,  c, a1,  b1);
  ADCX(c, &r2,  c, a2,  b2);
  ADCX(c, &r3,  c, a3,  b3);
  ADCX(c, &r4,  c, a4,  b4);
  ADCX(c, &r5,  c, a5,  b5);
  ADCX(c, &r6,  c, a6,  b6);
  ADCX(c, &r7,  c, a7,  b7);
  ADCX(c, &r8,  c, a8,  b8);
  ADCX(c, &r9,  c, a9,  b9);
  ADCX(c, &r10, c, a10, b10);
  ADCX(c, &r11, c, a11, b11);

  // r = r - p * 2^384
  SUBB(c, &r6,  0, r6,  p0);
  SUBB(c, &r7,  c, r7,  p1);
  SUBB(c, &r8,  c, r8,  p2);
  SUBB(c, &r9,  c, r9,  p3);
  SUBB(c, &r10, c, r10, p4);
  SUBB(c, &r11, c, r11, p5);

  // m = 0 - c
  m = 0 - (uint64_t)c;

  // p = p & m
  p0 &= m;
  p1 &= m;
  p2 &= m;
  p3 &= m;
  p4 &= m;
  p5 &= m;

  // r = r + p * 2^384
  ADCX(c, &r6,  0, r6,  p0);
  ADCX(c, &r7,  c, r7,  p1);
  ADCX(c, &r8,  c, r8,  p2);
  ADCX(c, &r9,  c, r9,  p3);
  ADCX(c, &r10, c, r10, p4);
  ADCX(c, &r11, c, r11, p5);

  ret[0] = r0; ret[1]  = r1;  ret[2]  = r2;
  ret[3] = r3; ret[4]  = r4;  ret[5]  = r5;
  ret[6] = r6; ret[7]  = r7;  ret[8]  = r8;
  ret[9] = r9; ret[10] = r10; ret[11] = r11;
}

// double-length ret = a - b mod (p * 2^384)
void sub_mod_384x384(vec768 ret, const vec768 a, const vec768 b, const vec384 p)
{
  uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4  = a[4],  a5  = a[5];
  uint64_t a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11];
  uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4  = b[4],  b5  = b[5];
  uint64_t b6 = b[6], b7 = b[7], b8 = b[8], b9 = b[9], b10 = b[10], b11 = b[11];
  uint64_t p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3], p4  = p[4],  p5  = p[5];
  uint64_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
  uint64_t m;
  uint8_t c;

  // r = a - b
  SUBB(c, &r0,  0, a0,  b0);
  SUBB(c, &r1,  c, a1,  b1);
  SUBB(c, &r2,  c, a2,  b2);
  SUBB(c, &r3,  c, a3,  b3);
  SUBB(c, &r4,  c, a4,  b4);
  SUBB(c, &r5,  c, a5,  b5);
  SUBB(c, &r6,  c, a6,  b6);
  SUBB(c, &r7,  c, a7,  b7);
  SUBB(c, &r8,  c, a8,  b8);
  SUBB(c, &r9,  c, a9,  b9);
  SUBB(c, &r10, c, a10, b10);
  SUBB(c, &r11, c, a11, b11);

  // m = 0 - c
  m = 0 - (uint64_t)c;

  // p = p & m
  p0 &= m;
  p1 &= m;
  p2 &= m;
  p3 &= m;
  p4 &= m;
  p5 &= m;

  // r = r + p * 2^384
  ADCX(c, &r6,  0, r6,  p0);
  ADCX(c, &r7,  c, r7,  p1);
  ADCX(c, &r8,  c, r8,  p2);
  ADCX(c, &r9,  c, r9,  p3);
  ADCX(c, &r10, c, r10, p4);
  ADCX(c, &r11, c, r11, p5);

  ret[0] = r0; ret[1]  = r1;  ret[2]  = r2;
  ret[3] = r3; ret[4]  = r4;  ret[5]  = r5;
  ret[6] = r6; ret[7]  = r7;  ret[8]  = r8;
  ret[9] = r9; ret[10] = r10; ret[11] = r11;
}
