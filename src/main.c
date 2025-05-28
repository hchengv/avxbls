#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "pairing.h"
#include "point.h"
#include "ec_ops.h"
#include "fp12_avx.h"

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
// for printing mpi

static void mpi_print(const char *c, const uint64_t *a, int len)
{
  int i;

  printf("%s", c);
  for (i = len-1; i > 0; i--) printf("%016lX", a[i]);
  printf("%016lX\n", a[0]);
}

// ----------------------------------------------------------------------------

// Only for generating a random point used for the correctness test,
// meaning the constant time is not required.

static void POINTonE1_scalarmul(POINTonE1 *R, const POINTonE1 *P, uint64_t *k)
{
  POINTonE1 _R[1], _T[1];
  int i, b, j = 0;

  vec_copy(_T, P, sizeof(POINTonE1));

  for (i = 0; i < 256; i++) {
    b = k[i>>6] >> (i & 63);
    if (b & 1) {
      if (j) POINTonE1_add(_R, _R, _T);  
      else { vec_copy(_R, _T, sizeof(POINTonE1)); j = 1;}
    }
    POINTonE1_double(_T, _T);
  }

  vec_copy(R, _R, sizeof(POINTonE1));
}

static void POINTonE2_scalarmul(POINTonE2 *R, const POINTonE2 *Q, uint64_t *k)
{
  POINTonE2 _R[1], _T[1];
  int i, b, j = 0;

  vec_copy(_T, Q, sizeof(POINTonE2));

  for (i = 0; i < 256; i++) {
    b = k[i>>6] >> (i & 63);
    if (b & 1) {
      if (j) POINTonE2_add(_R, _R, _T);  
      else { vec_copy(_R, _T, sizeof(POINTonE2)); j = 1;}
    }
    POINTonE2_double(_T, _T);
  }

  vec_copy(R, _R, sizeof(POINTonE2));
}

// ----------------------------------------------------------------------------

void test_pairing()
{
  POINTonE2_affine Q[1];
  POINTonE1_affine P[1];
  POINTonE2 _Q[1];
  POINTonE1 _P[1];
  vec384fp12 e1, e2;
  int seed;

  seed = (int) time(NULL);
	srand(seed);

  // The scalar k can be modified to be any non-0 value, e.g., 
  // uint64_t k[4] = { rand(), rand(), rand(), rand(), };
  uint64_t k[4] = {
    0x0123456789ABCDEF, 0x89ABCDEF01234567, 
    0x0123456789ABCDEF, 0x89ABCDEF01234567, };

  puts("\n=============================================================\n");
  puts("TEST - PAIRING\n");

  printf("- e(P, [k]Q) == e([k]P, Q) \n");
  // compute [k]Q
  POINTonE2_scalarmul(_Q, &BLS12_381_G2, k);
  // compute e1 = e(P, [k]Q)
  POINTonE1_to_affine(P, &BLS12_381_G1);
  POINTonE2_to_affine(Q, _Q);
  optimal_ate_pairing(e1, Q, P, 1);
  // compute [k]P
  POINTonE1_scalarmul(_P, &BLS12_381_G1, k);
  // compute e2 = e([k]P, Q)
  POINTonE1_to_affine(P, _P);
  POINTonE2_to_affine(Q, &BLS12_381_G2);
  optimal_ate_pairing(e2, Q, P, 1);
  // test whether e1 == e2: i.e., whether e(P, [k]Q) == e([k]P, Q)
  if (memcmp(e1, e2, sizeof(vec384fp12))) 
    printf("\x1b[31m  e1 != e2 \x1b[0m \n");
  else 
    printf("\x1b[32m  e1 == e2 \x1b[0m \n");

#ifdef TEST
  // print out the result
  mpi_print("e1[0][0][0] = ",  e1[0][0][0], 6);
  mpi_print("e1[0][0][1] = ",  e1[0][0][1], 6);
  mpi_print("e1[0][1][0] = ",  e1[0][1][0], 6);
  mpi_print("e1[0][1][1] = ",  e1[0][1][1], 6);
  mpi_print("e1[0][2][0] = ",  e1[0][2][0], 6);
  mpi_print("e1[0][2][1] = ",  e1[0][2][1], 6);
  mpi_print("e1[1][0][0] = ",  e1[1][0][0], 6);
  mpi_print("e1[1][0][1] = ",  e1[1][0][1], 6);
  mpi_print("e1[1][1][0] = ",  e1[1][1][0], 6);
  mpi_print("e1[1][1][1] = ",  e1[1][1][1], 6);
  mpi_print("e1[1][2][0] = ",  e1[1][2][0], 6);
  mpi_print("e1[1][2][1] = ",  e1[1][2][1], 6);
  puts("");
#endif 

  printf("- (e(P, [k]Q))^2 == e([2k]P, Q) \n");
  // compute [2k]P 
  POINTonE1_double(_P, _P);
  // compute e2 = e([2k]P, Q)
  POINTonE1_to_affine(P, _P);
  POINTonE2_to_affine(Q, &BLS12_381_G2);
  optimal_ate_pairing(e2, Q, P, 1);
  // compute e1 = e1^2 
  sqr_fp12(e1, e1);
  // test whether e1 == e2: i.e., whether (e(P, [k]Q)])^2 == e([2k]P, Q)
  if (memcmp(e1, e2, sizeof(vec384fp12))) 
    printf("\x1b[31m  e1 != e2 \x1b[0m \n");
  else 
    printf("\x1b[32m  e1 == e2 \x1b[0m \n");

#ifdef TEST
  // print out the result
  mpi_print("e1[0][0][0] = ",  e1[0][0][0], 6);
  mpi_print("e1[0][0][1] = ",  e1[0][0][1], 6);
  mpi_print("e1[0][1][0] = ",  e1[0][1][0], 6);
  mpi_print("e1[0][1][1] = ",  e1[0][1][1], 6);
  mpi_print("e1[0][2][0] = ",  e1[0][2][0], 6);
  mpi_print("e1[0][2][1] = ",  e1[0][2][1], 6);
  mpi_print("e1[1][0][0] = ",  e1[1][0][0], 6);
  mpi_print("e1[1][0][1] = ",  e1[1][0][1], 6);
  mpi_print("e1[1][1][0] = ",  e1[1][1][0], 6);
  mpi_print("e1[1][1][1] = ",  e1[1][1][1], 6);
  mpi_print("e1[1][2][0] = ",  e1[1][2][0], 6);
  mpi_print("e1[1][2][1] = ",  e1[1][2][1], 6);
  puts("");
#endif 

  printf("- (e(P, [k]Q))^4 == e([2k]P, [2]Q) \n");
  // compute [2]Q 
  POINTonE2_double(_Q, &BLS12_381_G2);
  // compute e2 = e([2k]P, [2]Q)
  POINTonE1_to_affine(P, _P);
  POINTonE2_to_affine(Q, _Q);
  optimal_ate_pairing(e2, Q, P, 1);
  // compute e1 = e1^2 
  sqr_fp12(e1, e1);
  // test whether e1 == e2: i.e., whether (e(P, [k]Q))^4 == e([2k]P, [2]Q)
  if (memcmp(e1, e2, sizeof(vec384fp12))) 
    printf("\x1b[31m  e1 != e2 \x1b[0m \n");
  else 
    printf("\x1b[32m  e1 == e2 \x1b[0m \n");

#ifdef TEST
  // print out the result
  mpi_print("e1[0][0][0] = ",  e1[0][0][0], 6);
  mpi_print("e1[0][0][1] = ",  e1[0][0][1], 6);
  mpi_print("e1[0][1][0] = ",  e1[0][1][0], 6);
  mpi_print("e1[0][1][1] = ",  e1[0][1][1], 6);
  mpi_print("e1[0][2][0] = ",  e1[0][2][0], 6);
  mpi_print("e1[0][2][1] = ",  e1[0][2][1], 6);
  mpi_print("e1[1][0][0] = ",  e1[1][0][0], 6);
  mpi_print("e1[1][0][1] = ",  e1[1][0][1], 6);
  mpi_print("e1[1][1][0] = ",  e1[1][1][0], 6);
  mpi_print("e1[1][1][1] = ",  e1[1][1][1], 6);
  mpi_print("e1[1][2][0] = ",  e1[1][2][0], 6);
  mpi_print("e1[1][2][1] = ",  e1[1][2][1], 6);
  puts("");
#endif 
}

void timing_pairing()
{
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

  puts("\n=============================================================\n");
  puts("TIMING - PAIRING\n");

  POINTonE2_affine Q[1];
  POINTonE1_affine P[1];
  POINTonE2 _Q[1];
  POINTonE1 _P[1];
  vec384fp12 e1, f;

  POINTonE2_double(_Q, &BLS12_381_G2);
  POINTonE1_to_affine(P, &BLS12_381_G1);
  POINTonE2_to_affine(Q, _Q);

  printf("- miller_loop:        ");
  LOAD_CACHE(miller_loop_n(f, Q, P, 1), 1000);
  MEASURE_CYCLES(miller_loop_n(f, Q, P, 1), 10000);
  printf("  #cycle = %ld\n", diff_cycles);

  printf("- final_exp:          ");
  LOAD_CACHE(final_exp(e1, f), 1000);
  MEASURE_CYCLES(final_exp(e1, f), 10000);
  printf("  #cycle = %ld\n", diff_cycles);

  printf("- optimal_ate_pairing:");
  LOAD_CACHE(optimal_ate_pairing(e1, Q, P, 1), 1000);
  MEASURE_CYCLES(optimal_ate_pairing(e1, Q, P, 1), 10000);
  printf("  #cycle = %ld\n", diff_cycles);

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

// f := 0x19FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
#define TV_F \
0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, \
0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x19FFFFFFFFFFFFFF,

// ----------------------------------------------------------------------------

void test_timing_fp()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS], z64[2*SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS], z48[2*NWORDS];
  __m512i a_8x1w[NWORDS], b_8x1w[NWORDS], r_8x1w[NWORDS], z_8x1w[2*NWORDS];
  __m512i a_4x2w[VWORDS], b_4x2w[VWORDS], r_4x2w[VWORDS], z_4x2w[3*VWORDS];
  vec768 s;
  vec384 c, d;
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

#if 0
  puts("\n=============================================================\n");
  puts("TEST - FP\n");

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

  add_fp_8x1w(r_8x1w, a_8x1w, b_8x1w);
  get_channel_8x1w(r48, r_8x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* add_fp_8x1w r0 = 0x", r64, SWORDS);

  sub_fp_8x1w(r_8x1w, a_8x1w, b_8x1w);
  get_channel_8x1w(r48, r_8x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sub_fp_8x1w r0 = 0x", r64, SWORDS);

  mul_fpx2_8x1w_v1(z_8x1w, a_8x1w, b_8x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_8x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fpx2_8x1w_v1 r0 = 0x", z64, 2*SWORDS);

  mul_fpx2_8x1w_v3(z_8x1w, a_8x1w, b_8x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_8x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fpx2_8x1w_v3 r0 = 0x", z64, 2*SWORDS);

  redc_fpx2_8x1w(r_8x1w, z_8x1w);
  get_channel_8x1w(r48, r_8x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* redc_fpx2_8x1w r0 = 0x", r64, SWORDS);

  add_fp_4x2w(r_4x2w, a_4x2w, b_4x2w);
  get_channel_4x2w(r48, r_4x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* add_fp_4x2w r0 = 0x", r64, SWORDS);

  mul_fpx2_4x2w(z_4x2w, a_4x2w, b_4x2w);
  redc_fpx2_4x2w(r_4x2w, z_4x2w);
  get_channel_4x2w(r48, r_4x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp_4x2w r0 = 0x", r64, SWORDS);
#endif

#ifdef BENCHMARK
  puts("\n=============================================================\n");
  puts("TIMING - FP\n");

  printf("- mul_fpx2_8x1w_v1:     ");
  LOAD_CACHE(mul_fpx2_8x1w_v1(z_8x1w, a_8x1w, b_8x1w), 10000);
  MEASURE_CYCLES(mul_fpx2_8x1w_v1(z_8x1w, a_8x1w, b_8x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fpx2_8x1w_v2:     ");
  LOAD_CACHE(mul_fpx2_8x1w_v2(z_8x1w, a_8x1w, b_8x1w), 10000);
  MEASURE_CYCLES(mul_fpx2_8x1w_v2(z_8x1w, a_8x1w, b_8x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fpx2_8x1w_hybrid_v0: ");
  LOAD_CACHE(mul_fpx2_8x1w_hybrid_v0(z_8x1w, a_8x1w, b_8x1w), 10000);
  MEASURE_CYCLES(mul_fpx2_8x1w_hybrid_v0(z_8x1w, a_8x1w, b_8x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fpx2_8x1w_hybrid_v1: ");
  LOAD_CACHE(mul_fpx2_8x1w_hybrid_v1(z_8x1w, s, a_8x1w, b_8x1w, c, d), 10000);
  MEASURE_CYCLES(mul_fpx2_8x1w_hybrid_v1(z_8x1w, s, a_8x1w, b_8x1w, c, d), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fpx2_8x1w_v3:     ");
  LOAD_CACHE(mul_fpx2_8x1w_v3(z_8x1w, a_8x1w, b_8x1w), 10000);
  MEASURE_CYCLES(mul_fpx2_8x1w_v3(z_8x1w, a_8x1w, b_8x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fpx2_8x1w_v4:     ");
  LOAD_CACHE(mul_fpx2_8x1w_v4(z_8x1w, a_8x1w, b_8x1w), 10000);
  MEASURE_CYCLES(mul_fpx2_8x1w_v4(z_8x1w, a_8x1w, b_8x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fpx2_4x2w_v1:     ");
  LOAD_CACHE(mul_fpx2_4x2w_v1(z_4x2w, a_4x2w, b_4x2w), 10000);
  MEASURE_CYCLES(mul_fpx2_4x2w_v1(z_4x2w, a_4x2w, b_4x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- redc_fpx2_8x1w:       ");
  LOAD_CACHE(redc_fpx2_8x1w(r_8x1w, z_8x1w), 10000);
  MEASURE_CYCLES(redc_fpx2_8x1w(r_8x1w, z_8x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- redc_fpx2_4x2w:       ");
  LOAD_CACHE(redc_fpx2_4x2w(r_4x2w, z_4x2w), 10000);
  MEASURE_CYCLES(redc_fpx2_4x2w(r_4x2w, z_4x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);
#endif
}

// ----------------------------------------------------------------------------

void test_timing_fp2()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS], z64[2*SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS], z48[2*NWORDS];
  __m512i a_4x2x1w[NWORDS], b_4x2x1w[NWORDS], r_4x2x1w[NWORDS], z_4x2x1w[2*NWORDS];
  __m512i a_2x2x2w[VWORDS], b_2x2x2w[VWORDS], r_2x2x2w[VWORDS], z_2x2x2w[3*VWORDS];
  __m512i a_2x4x1w[NWORDS], b_2x4x1w[NWORDS], z_2x4x1w[2*NWORDS];
  __m512i a_1x4x2w[VWORDS], b_1x4x2w[VWORDS], z_1x4x2w[3*VWORDS];
  fp2x2_2x2x2w aa_2x2x2w, rr_2x2x2w;
  fp2x2_8x1x1w z_8x1x1w;
  fp2_8x1x1w a_8x1x1w, b_8x1x1w;
  vec384x r_scalar, a_scalar, b_scalar;
  vec768x z_scalar;
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

#if 0
  puts("\n=============================================================\n");
  puts("\nTEST - FP2\n");

  conv_64to48_mpi(a48, a64, NWORDS, SWORDS);
  conv_64to48_mpi(b48, b64, NWORDS, SWORDS);

  for (i = 0; i < NWORDS; i++) {
    a_4x2x1w[i] = VSET1(a48[i]);
    b_4x2x1w[i] = VSET1(b48[i]);
    a_2x4x1w[i] = VSET1(a48[i]);
    b_2x4x1w[i] = VSET1(b48[i]);
  }

  for (i = 0; i < VWORDS; i++) {
    a_2x2x2w[i] = VSET(0, 0, a48[i+VWORDS], a48[i], 0, 0, a48[i+VWORDS], a48[i]);
    b_2x2x2w[i] = VSET(0, 0, b48[i+VWORDS], b48[i], 0, 0, b48[i+VWORDS], b48[i]);
  }

  for (i = 0; i < VWORDS*2; i++) {
    aa_2x2x2w[i] = VSET(0, 0, 0, 0, 0, b48[i], 0, a48[i]);
  }
  for (i = VWORDS*2; i < VWORDS*3-1; i++) {
    aa_2x2x2w[i] = VSET(0 ,0, 0, 0, b48[i+VWORDS-NWORDS], b48[i-NWORDS], a48[i+VWORDS-NWORDS], a48[i-NWORDS]);
  }
  aa_2x2x2w[VWORDS*3-1] = VSET(0, 0, 0, 0, 0, b48[VWORDS-1], 0, a48[VWORDS-1]);

  mul_by_u_plus_1_fp2x2_2x2x2w(rr_2x2x2w, aa_2x2x2w);
  for(i = 0; i < 2*VWORDS; i++) z48[i] = ((uint64_t *)&rr_2x2x2w[i])[0];
  get_channel_4x2w(&z48[2*VWORDS], &rr_2x2x2w[2*VWORDS], 0);
  carryp_mpi48(&z48[2*VWORDS]);
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_u_plus_1_fp2x2_2x2x2w r0 = 0x", z64, 2*SWORDS);
  for(i = 0; i < 2*VWORDS; i++) z48[i] = ((uint64_t *)&rr_2x2x2w[i])[2];
  get_channel_4x2w(&z48[2*VWORDS], &rr_2x2x2w[2*VWORDS], 2);
  carryp_mpi48(&z48[2*VWORDS]);
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_u_plus_1_fp2x2_2x2x2w r2 = 0x", z64, 2*SWORDS);

  assa_fp2_4x2x1w(r_4x2x1w, a_4x2x1w, b_4x2x1w);
  get_channel_8x1w(r48, r_4x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* assa_fp2_4x2x1w r0 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_4x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* assa_fp2_4x2x1w r2 = 0x", r64, SWORDS);

  as_fp2_2x2x2w(r_2x2x2w, a_2x2x2w, b_2x2x2w);
  get_channel_4x2w(r48, r_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* as_fp2_2x2x2w r0 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r_2x2x2w, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* as_fp2_2x2x2w r4 = 0x", r64, SWORDS);

  for (i = 0; i < NWORDS; i++) 
    a_4x2x1w[i] = VSET(0, 0, 0, 0, 0, 0, b48[i], a48[i]);
  sqr_fp2x2_4x2x1w(z_4x2x1w, a_4x2x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_4x2x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* sqr_fp2x2_4x2x1w r0 = 0x", z64, 2*SWORDS);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_4x2x1w[i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* sqr_fp2x2_4x2x1w r1 = 0x", z64, 2*SWORDS);

  for (i = 0; i < VWORDS; i++) 
    a_2x2x2w[i] = VSET(0, 0, 0, 0, b48[i+VWORDS], b48[i], a48[i+VWORDS], a48[i]);
  sqr_fp2_2x2x2w(r_2x2x2w, a_2x2x2w);
  get_channel_4x2w(r48, r_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp2_2x2x2w r0 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp2_2x2x2w r2 = 0x", r64, SWORDS);

  mul_by_u_plus_1_fp2x2_4x2x1w(z_4x2x1w, z_4x2x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_4x2x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_u_plus_1_fp2x2_4x2x1w r0 = 0x", z64, 2*SWORDS);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_4x2x1w[i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_u_plus_1_fp2x2_4x2x1w r1 = 0x", z64, 2*SWORDS);

  mul_by_u_plus_1_fp2_2x2x2w(r_2x2x2w, r_2x2x2w);
  get_channel_4x2w(r48, r_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_u_plus_1_fp2_2x2x2w r0 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_u_plus_1_fp2_2x2x2w r2 = 0x", r64, SWORDS);

  mul_fp2x2_2x4x1w(z_2x4x1w, a_2x4x1w, b_2x4x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_2x4x1w[i])[2];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp2x2_2x4x1w r2 = 0x", z64, 2*SWORDS);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_2x4x1w[i])[3];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp2x2_2x4x1w r3 = 0x", z64, 2*SWORDS);
#endif

#ifdef BENCHMARK
  puts("\n=============================================================\n");
  puts("TIMING - FP2\n");

  printf("- mul_fp2_scalar:       ");
  LOAD_CACHE(mul_fp2(r_scalar, a_scalar, b_scalar), 10000);
  MEASURE_CYCLES(mul_fp2(r_scalar, a_scalar, b_scalar), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp2_4x2x1w:       ");
  LOAD_CACHE(mul_fp2_4x2x1w(r_4x2x1w, a_4x2x1w, b_4x2x1w), 10000);
  MEASURE_CYCLES(mul_fp2_4x2x1w(r_4x2x1w, a_4x2x1w, b_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp2_2x2x2w:       ");
  LOAD_CACHE(mul_fp2_2x2x2w(r_2x2x2w, a_2x2x2w, b_2x2x2w), 10000);
  MEASURE_CYCLES(mul_fp2_2x2x2w(r_2x2x2w, a_2x2x2w, b_2x2x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp2x2_scalar:     ");
  LOAD_CACHE(mul_fp2x2(z_scalar, a_scalar, b_scalar), 10000);
  MEASURE_CYCLES(mul_fp2x2(z_scalar, a_scalar, b_scalar), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp2x2_8x1x1w:     ");
  LOAD_CACHE(mul_fp2x2_8x1x1w(z_8x1x1w, a_8x1x1w, b_8x1x1w), 10000);
  MEASURE_CYCLES(mul_fp2x2_8x1x1w(z_8x1x1w, a_8x1x1w, b_8x1x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp2x2_4x2x1w:     ");
  LOAD_CACHE(mul_fp2x2_4x2x1w(z_4x2x1w, a_4x2x1w, b_4x2x1w), 10000);
  MEASURE_CYCLES(mul_fp2x2_4x2x1w(z_4x2x1w, a_4x2x1w, b_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp2x2_2x4x1w:     ");
  LOAD_CACHE(mul_fp2x2_2x4x1w(z_2x4x1w, a_2x4x1w, b_2x4x1w), 10000);
  MEASURE_CYCLES(mul_fp2x2_2x4x1w(z_2x4x1w, a_2x4x1w, b_2x4x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp2x2_2x2x2w:     ");
  LOAD_CACHE(mul_fp2x2_2x2x2w(rr_2x2x2w, a_2x2x2w, b_2x2x2w), 10000);
  MEASURE_CYCLES(mul_fp2x2_2x2x2w(rr_2x2x2w, a_2x2x2w, b_2x2x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp2x2_1x4x2w:     ");
  LOAD_CACHE(mul_fp2x2_1x4x2w(z_1x4x2w, a_1x4x2w, b_1x4x2w), 10000);
  MEASURE_CYCLES(mul_fp2x2_1x4x2w(z_1x4x2w, a_1x4x2w, b_1x4x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp2_scalar:       ");
  LOAD_CACHE(sqr_fp2(r_scalar, a_scalar), 10000);
  MEASURE_CYCLES(sqr_fp2(r_scalar, a_scalar), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp2_4x2x1w:       ");
  LOAD_CACHE(sqr_fp2_4x2x1w(r_4x2x1w, a_4x2x1w), 10000);
  MEASURE_CYCLES(sqr_fp2_4x2x1w(r_4x2x1w, a_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp2_2x2x2w:       ");
  LOAD_CACHE(sqr_fp2_2x2x2w(r_2x2x2w, a_2x2x2w), 10000);
  MEASURE_CYCLES(sqr_fp2_2x2x2w(r_2x2x2w, a_2x2x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp2x2_scalar:     ");
  LOAD_CACHE(sqr_fp2x2(z_scalar, a_scalar), 10000);
  MEASURE_CYCLES(sqr_fp2x2(z_scalar, a_scalar), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp2x2_4x2x1w:     ");
  LOAD_CACHE(sqr_fp2x2_4x2x1w(z_4x2x1w, a_4x2x1w), 10000);
  MEASURE_CYCLES(sqr_fp2x2_4x2x1w(z_4x2x1w, a_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp2x2_2x2x2w:     ");
  LOAD_CACHE(sqr_fp2x2_2x2x2w(rr_2x2x2w, a_2x2x2w), 10000);
  MEASURE_CYCLES(sqr_fp2x2_2x2x2w(rr_2x2x2w, a_2x2x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);
#endif
}

// ----------------------------------------------------------------------------

void test_timing_fp4()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS], z64[2*SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS], z48[2*NWORDS];
  __m512i a_2x2x2x1w[NWORDS], b_2x2x2x1w[NWORDS], r_2x2x2x1w[NWORDS], z[2*NWORDS];
  __m512i a_1x2x2x2w[VWORDS], r_1x2x2x2w[VWORDS];
  vec384fp4 r_scalar;
  vec384x a0_scalar, a1_scalar;
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

#if 0
  puts("\n=============================================================\n");
  puts("\nTEST - FP4\n");

  conv_64to48_mpi(a48, a64, NWORDS, SWORDS);
  conv_64to48_mpi(b48, b64, NWORDS, SWORDS);

  for (i = 0; i < NWORDS; i++) {
    a_2x2x2x1w[i] = VSET(0, 0, 0, 0, b48[i], a48[i], b48[i], a48[i]);
  }

  sqr_fp4_2x2x2x1w_v1(r_2x2x2x1w, a_2x2x2x1w);
  get_channel_8x1w(r48, r_2x2x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w_v1 r00 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 1);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w_v1 r01 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w_v1 r10 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w_v1 r11 = 0x", r64, SWORDS);

  sqr_fp4_2x2x2x1w_v2(r_2x2x2x1w, a_2x2x2x1w);
  get_channel_8x1w(r48, r_2x2x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w_v2 r00 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 1);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w_v2 r01 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w_v2 r10 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w_v2 r11 = 0x", r64, SWORDS);
#endif

#ifdef BENCHMARK
  puts("\n=============================================================\n");
  puts("TIMING - FP4\n");

  printf("- sqr_fp4_scalar:       ");
  LOAD_CACHE(sqr_fp4(r_scalar, a0_scalar, a1_scalar), 10000);
  MEASURE_CYCLES(sqr_fp4(r_scalar, a0_scalar, a1_scalar), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp4_2x2x2x1w_v1:  ");
  LOAD_CACHE(sqr_fp4_2x2x2x1w_v1(r_2x2x2x1w, a_2x2x2x1w), 10000);
  MEASURE_CYCLES(sqr_fp4_2x2x2x1w_v1(r_2x2x2x1w, a_2x2x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp4_1x2x2x2w_v1:  ");
  LOAD_CACHE(sqr_fp4_1x2x2x2w_v1(r_1x2x2x2w, a_1x2x2x2w), 10000);
  MEASURE_CYCLES(sqr_fp4_1x2x2x2w_v1(r_1x2x2x2w, a_1x2x2x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);
#endif
}

// ----------------------------------------------------------------------------

void test_timing_fp6()
{
  uint64_t r64[NWORDS], z64[2*SWORDS];
  uint64_t r48[NWORDS], z48[2*NWORDS];
  fp2x2_8x1x1w r01_8x1x1w, r23_8x1x1w, r45_8x1x1w; 
  fp2_8x1x1w a0_8x1x1w, a1_8x1x1w, a2_8x1x1w, a_8x1x1w, b_8x1x1w;
  fp2x2_8x1x1w z01_8x1x1w, z2_8x1x1w;
  fp2_4x2x1w a0_4x2x1w, a1_4x2x1w, a2_4x2x1w, a01_4x2x1w, r01_4x2x1w, r2_4x2x1w;
  fp2x2_4x2x1w z01_4x2x1w, z2_4x2x1w, z3_4x2x1w, z45_4x2x1w;
  fp2_2x2x2w a0_2x2x2w, a1_2x2x2w, a2_2x2x2w, r01_2x2x2w, r2_2x2x2w;
  fp2x2_2x2x2w z01_2x2x2w, z2_2x2x2w;
  vec768fp6 r; 
  vec384fp6 a = {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}}, b;
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

#if 0

  puts("\n=============================================================\n");
  puts("\nFP6 TEST\n");

  a01_4x2x1w[0] = VSET(0, 0, 0, 0, 4, 3, 2, 1);
  a2_4x2x1w[0] = VSET(0, 0, 0, 0, 6, 5, 6, 5);

  for (i = 1; i < NWORDS; i++) {
    a01_4x2x1w[i] = VZERO;
    a2_4x2x1w[i] = VZERO; 
  }

  a_8x1x1w[0][0] = VSET(0, 0, 0, 0, 5, 5, 3, 1);
  a_8x1x1w[1][0] = VSET(0, 0, 0, 0, 6, 6, 4, 2);
  b_8x1x1w[0][0] = VSET(0, 0, 0, 0, 1, 3, 1, 5);
  b_8x1x1w[1][0] = VSET(0, 0, 0, 0, 2, 4, 2, 6);

  for (i = 1; i < NWORDS; i++) {
    a_8x1x1w[0][i] = a_8x1x1w[1][i] = VZERO;
    b_8x1x1w[0][i] = b_8x1x1w[1][i] = VZERO;
  }

  mul_by_xy0_fp6x2(r, a, a);
  mpi_print("* mul_by_xy0_fp6x2 r00 = 0x", r[0][0], 2*SWORDS);
  mpi_print("* mul_by_xy0_fp6x2 r01 = 0x", r[0][1], 2*SWORDS);
  mpi_print("* mul_by_xy0_fp6x2 r10 = 0x", r[1][0], 2*SWORDS);
  mpi_print("* mul_by_xy0_fp6x2 r11 = 0x", r[1][1], 2*SWORDS);
  mpi_print("* mul_by_xy0_fp6x2 r20 = 0x", r[2][0], 2*SWORDS);
  mpi_print("* mul_by_xy0_fp6x2 r21 = 0x", r[2][1], 2*SWORDS);

  mul_by_0y0_fp6x2(r, a, a[2]);
  mpi_print("* mul_by_0y0_fp6x2 r00 = 0x", r[0][0], 2*SWORDS);
  mpi_print("* mul_by_0y0_fp6x2 r01 = 0x", r[0][1], 2*SWORDS);
  mpi_print("* mul_by_0y0_fp6x2 r10 = 0x", r[1][0], 2*SWORDS);
  mpi_print("* mul_by_0y0_fp6x2 r11 = 0x", r[1][1], 2*SWORDS);
  mpi_print("* mul_by_0y0_fp6x2 r20 = 0x", r[2][0], 2*SWORDS);
  mpi_print("* mul_by_0y0_fp6x2 r21 = 0x", r[2][1], 2*SWORDS);

  mul_by_xy00z0_fp6x2_2x4x1x1w(r01_8x1x1w, r23_8x1x1w, r45_8x1x1w, a_8x1x1w, b_8x1x1w);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r01_8x1x1w[0][i])[3];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z00 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r01_8x1x1w[1][i])[3];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z01 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r01_8x1x1w[0][i])[2];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z10 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r01_8x1x1w[1][i])[2];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z11 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r23_8x1x1w[0][i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z20 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r23_8x1x1w[1][i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z21 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r23_8x1x1w[0][i])[2];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z30 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r23_8x1x1w[1][i])[2];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z31 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r45_8x1x1w[0][i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z40 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r45_8x1x1w[1][i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z41 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r45_8x1x1w[0][i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z50 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&r45_8x1x1w[1][i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_xy00z0_fp6x2_2x4x2x1w z51 = 0x", z64, 2*SWORDS);

  // mul_by_xy00z0_fp6x2_2x2x2x1w(z01_4x2x1w, z2_4x2x1w, z3_4x2x1w, z45_4x2x1w,
  //                              a01_4x2x1w, a2_4x2x1w, a01_4x2x1w, a2_4x2x1w);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z00 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[1];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z01 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[2];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z10 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[3];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z11 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_4x2x1w[i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z20 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_4x2x1w[i])[1];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z21 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z3_4x2x1w[i])[2];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z30 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z3_4x2x1w[i])[3];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z31 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z45_4x2x1w[i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z40 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z45_4x2x1w[i])[1];
  // for (i = 0; i < 2*NWORDS-1; i++) {
  //   z48[i+1] += z48[i]>>BRADIX; z48[i] &= BMASK;
  // }
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z41 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z45_4x2x1w[i])[2];
  // for (i = 0; i < 2*NWORDS-1; i++) {
  //   z48[i+1] += z48[i]>>BRADIX; z48[i] &= BMASK;
  // }
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z50 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z45_4x2x1w[i])[3];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_by_xy00z0_fp6x2_2x2x2x1w z51 = 0x", z64, 2*SWORDS);

  // a0_8x1x1w[0][0] = VSET1(1);
  // a0_8x1x1w[1][0] = VSET1(2);
  // a1_8x1x1w[0][0] = VSET1(3);
  // a1_8x1x1w[1][0] = VSET1(4);
  // a2_8x1x1w[0][0] = VSET1(5);
  // a2_8x1x1w[1][0] = VSET1(6);

  // for(i = 1; i < NWORDS; i++) {
  //   a0_8x1x1w[0][i] = VZERO;
  //   a0_8x1x1w[1][i] = VZERO;
  //   a1_8x1x1w[0][i] = VZERO;
  //   a1_8x1x1w[1][i] = VZERO;
  //   a2_8x1x1w[0][i] = VZERO;
  //   a2_8x1x1w[1][i] = VZERO;
  // }

  // a0_4x2x1w[0] = VSET(0, 0, 0, 0, 2, 1, 2, 1);
  // a1_4x2x1w[0] = VSET(0, 0, 0, 0, 4, 3, 4, 3);
  // a2_4x2x1w[0] = VSET(0, 0, 0, 0, 6, 5, 6, 5);

  // for (i = 1; i < NWORDS; i++) {
  //   a0_4x2x1w[i] = VZERO;
  //   a1_4x2x1w[i] = VZERO;
  //   a2_4x2x1w[i] = VZERO;
  // }

  // a0_2x2x2w[0] = VSET(0, 2, 0, 1, 0, 2, 0, 1);
  // a1_2x2x2w[0] = VSET(0, 4, 0, 3, 0, 4, 0, 3);
  // a2_2x2x2w[0] = VSET(0, 6, 0, 5, 0, 6, 0, 5);

  // for (i = 1; i < VWORDS; i++) {
  //   a0_2x2x2w[i] = VZERO;
  //   a1_2x2x2w[i] = VZERO;
  //   a2_2x2x2w[i] = VZERO;
  // }

  // mul_fp6x2(r, a, a);
  // mpi_print("* mul_fp6x2 r00 = 0x", r[0][0], 2*SWORDS);
  // mpi_print("* mul_fp6x2 r01 = 0x", r[0][1], 2*SWORDS);
  // mpi_print("* mul_fp6x2 r10 = 0x", r[1][0], 2*SWORDS);
  // mpi_print("* mul_fp6x2 r11 = 0x", r[1][1], 2*SWORDS);
  // mpi_print("* mul_fp6x2 r20 = 0x", r[2][0], 2*SWORDS);
  // mpi_print("* mul_fp6x2 r21 = 0x", r[2][1], 2*SWORDS);

  // mul_fp6x2_4x2x1x1w(z01_8x1x1w, z2_8x1x1w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_8x1x1w[0][i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_4x2x1x1w z00 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_8x1x1w[1][i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_4x2x1x1w z01 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_8x1x1w[0][i])[1];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_4x2x1x1w z10 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_8x1x1w[1][i])[1];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_4x2x1x1w z11 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_8x1x1w[0][i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_4x2x1x1w z20 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_8x1x1w[1][i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_4x2x1x1w z21 = 0x", z64, 2*SWORDS);

  // mul_fp6x2_2x2x2x1w(z01_4x2x1w, z2_4x2x1w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w z00 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[1];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w z01 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[2];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w z10 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[3];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w z11 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_4x2x1w[i])[0];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w z20 = 0x", z64, 2*SWORDS);
  // for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_4x2x1w[i])[1];
  // conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w z21 = 0x", z64, 2*SWORDS);
  // redc_fpx2_8x1w(r01_4x2x1w, z01_4x2x1w);
  // redc_fpx2_8x1w(r2_4x2x1w, z2_4x2x1w);
  // for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r01_4x2x1w[i])[0];
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w r00 = 0x", r64, SWORDS);
  // for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r01_4x2x1w[i])[1];
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w r01 = 0x", r64, SWORDS);
  // for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r01_4x2x1w[i])[2];
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w r10 = 0x", r64, SWORDS);
  // for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r01_4x2x1w[i])[3];
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w r11 = 0x", r64, SWORDS);
  // for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r2_4x2x1w[i])[0];
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w r20 = 0x", r64, SWORDS);
  // for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r2_4x2x1w[i])[1];
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_2x2x2x1w r21 = 0x", r64, SWORDS);

  // mul_fp6x2_1x2x2x2w(z01_2x2x2w, z2_2x2x2w, a0_2x2x2w, a1_2x2x2w, a2_2x2x2w);
  // redc_fpx2_4x2w(r01_2x2x2w, z01_2x2x2w);
  // redc_fpx2_4x2w(r2_2x2x2w, z2_2x2x2w);
  // get_channel_4x2w(r48, r01_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_1x2x2x2w r00 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r01_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_1x2x2x2w r01 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r01_2x2x2w, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_1x2x2x2w r10 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r01_2x2x2w, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_1x2x2x2w r11 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_1x2x2x2w r20 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp6x2_1x2x2x2w r21 = 0x", r64, SWORDS);
#endif

#ifdef BENCHMARK
  puts("\n=============================================================\n");
  puts("TIMING - FP6\n");

  printf("- mul_fp6x2_scalar:     ");
  LOAD_CACHE(mul_fp6x2(r, a, b), 10000);
  MEASURE_CYCLES(mul_fp6x2(r, a, b), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp6x2_2x2x2x1w:   ");
  LOAD_CACHE(mul_fp6x2_2x2x2x1w(z01_4x2x1w, z2_4x2x1w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w), 10000);
  MEASURE_CYCLES(mul_fp6x2_2x2x2x1w(z01_4x2x1w, z2_4x2x1w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp6x2_1x2x2x2w:   ");
  LOAD_CACHE(mul_fp6x2_1x2x2x2w(z01_2x2x2w, z2_2x2x2w, a0_2x2x2w, a1_2x2x2w, a2_2x2x2w), 10000);
  MEASURE_CYCLES(mul_fp6x2_1x2x2x2w(z01_2x2x2w, z2_2x2x2w, a0_2x2x2w, a1_2x2x2w, a2_2x2x2w), 100000);
  printf("#cycle = %ld\n", diff_cycles);
#endif
}

// ----------------------------------------------------------------------------

void test_timing_fp12()
{
  vec384fp12 r;
  vec384fp12 a = {
    {{{1}, {2}}, {{3}, {4 }}, {{5 }, {6 }}},
    {{{7}, {8}}, {{9}, {10}}, {{11}, {12}}}};
  vec384fp6 b = {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}};
  uint64_t r64[NWORDS], z64[2*SWORDS];
  uint64_t r48[NWORDS], z48[2*NWORDS];
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

  fp4_1x2x2x2w ra_1x2x2x2w, a_1x2x2x2w;
  fp4_2x2x2x1w rbc_2x2x2x1w, bc_2x2x2x1w;
  fp2_8x1x1w a0_8x1x1w, a1_8x1x1w, a2_8x1x1w, a_8x1x1w, b_8x1x1w;
  fp2_4x2x1w r01_4x2x1w, r2_4x2x1w;
  fp2_4x2x1w a0_4x2x1w, a1_4x2x1w, a2_4x2x1w;
  fp2_2x2x2w r001_2x2x2w, r101_2x2x2w, r2_2x2x2w, r12_2x2x2w;
  fp2_4x2x1w r0_4x2x1w, r1_4x2x1w, a01_4x2x1w, b01_4x2x1w, b4_4x2x1w;
  fp2_2x2x2w r1_2x2x2w; 

#if 0
  puts("\nTEST - FP12\n");

  // a0_4x2x1w[0] = VSET(11, 11, 6  , 5  , 4 , 3, 2, 1);
  // a1_4x2x1w[0] = VSET(12, 12, 12 , 11 , 10, 9, 8, 7);

  // for (i = 1; i < NWORDS; i++) {
  //   a0_4x2x1w[i] = VZERO;
  //   a1_4x2x1w[i] = VZERO;
  // }

  // sqr_fp12_scalar(r, a);
  // mpi_print("* sqr_fp12_scalar r000 = 0x", r[0][0][0], SWORDS);
  // mpi_print("* sqr_fp12_scalar r001 = 0x", r[0][0][1], SWORDS);
  // mpi_print("* sqr_fp12_scalar r010 = 0x", r[0][1][0], SWORDS);
  // mpi_print("* sqr_fp12_scalar r011 = 0x", r[0][1][1], SWORDS);
  // mpi_print("* sqr_fp12_scalar r020 = 0x", r[0][2][0], SWORDS);
  // mpi_print("* sqr_fp12_scalar r021 = 0x", r[0][2][1], SWORDS);
  // mpi_print("* sqr_fp12_scalar r100 = 0x", r[1][0][0], SWORDS);
  // mpi_print("* sqr_fp12_scalar r101 = 0x", r[1][0][1], SWORDS);
  // mpi_print("* sqr_fp12_scalar r110 = 0x", r[1][1][0], SWORDS);
  // mpi_print("* sqr_fp12_scalar r111 = 0x", r[1][1][1], SWORDS);
  // mpi_print("* sqr_fp12_scalar r120 = 0x", r[1][2][0], SWORDS);
  // mpi_print("* sqr_fp12_scalar r121 = 0x", r[1][2][1], SWORDS);

  // sqr_fp12_vec_v1(r0_4x2x1w, r1_4x2x1w, a0_4x2x1w, a1_4x2x1w);
  // get_channel_8x1w(r48, r0_4x2x1w, 0);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r000 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 1);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r001 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 2);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r010 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 3);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r011 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 4);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r020 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 5);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r021 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r1_4x2x1w, 0);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r100 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r1_4x2x1w, 1);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r101 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r1_4x2x1w, 2);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r110 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r1_4x2x1w, 3);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r111 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r1_4x2x1w, 4);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r120 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r1_4x2x1w, 5);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* sqr_fp12_vec_v1 r121 = 0x", r64, SWORDS);

  mul_by_xy00z0_fp12_scalar(r, a, b);
  mpi_print("* mul_by_xy00z0_fp12_scalar r000 = 0x", r[0][0][0], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r001 = 0x", r[0][0][1], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r010 = 0x", r[0][1][0], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r011 = 0x", r[0][1][1], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r020 = 0x", r[0][2][0], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r021 = 0x", r[0][2][1], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r100 = 0x", r[1][0][0], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r101 = 0x", r[1][0][1], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r110 = 0x", r[1][1][0], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r111 = 0x", r[1][1][1], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r120 = 0x", r[1][2][0], SWORDS);
  mpi_print("* mul_by_xy00z0_fp12_scalar r121 = 0x", r[1][2][1], SWORDS);

  a_8x1x1w[0][0] = VSET(11, 11, 9, 7, 5, 5, 3, 1);
  a_8x1x1w[1][0] = VSET(12, 12, 10, 8, 6, 6, 4, 2);
  b_8x1x1w[0][0] = VSET(1, 3, 1, 5, 1, 3, 1, 5);
  b_8x1x1w[1][0] = VSET(2, 4, 2, 6, 2, 4, 2, 6);

  for (i = 1; i < NWORDS; i++) {
    a_8x1x1w[0][i] = a_8x1x1w[1][i] = VZERO;
    b_8x1x1w[0][i] = b_8x1x1w[1][i] = VZERO;    
  }

  mul_by_xy00z0_fp12_vec_v2(r0_4x2x1w, r1_2x2x2w, a_8x1x1w, b_8x1x1w);
  get_channel_4x2w(r48, r1_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r000 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r1_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r001 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r0_4x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r010 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r0_4x2x1w, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r011 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r0_4x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r020 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r0_4x2x1w, 1);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r021 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r0_4x2x1w, 4);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r100 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r0_4x2x1w, 5);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r101 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r0_4x2x1w, 6);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r110 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r0_4x2x1w, 7);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r111 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r1_2x2x2w, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r120 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r1_2x2x2w, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_by_xy00z0_fp12_vec_v2 r121 = 0x", r64, SWORDS);

  // a01_4x2x1w[0] = VSET(10, 9 , 8 , 7 , 4, 3, 2, 1);
  // a2_4x2x1w [0] = VSET(12, 11, 12, 11, 6, 5, 6, 5);
  // b01_4x2x1w[0] = VSET(4 , 3 , 2 , 1 , 4, 3, 2, 1);
  // b4_4x2x1w [0] = VSET(6 , 5 , 6 , 5 , 6, 5, 6, 5);

  // for (i = 1; i < NWORDS; i++) {
  //   a01_4x2x1w[i] = VZERO;
  //   a2_4x2x1w [i] = VZERO;
  //   b01_4x2x1w[i] = VZERO;
  //   b4_4x2x1w [i] = VZERO;
  // }

  // mul_by_xy00z0_fp12_vec_v1(r0_4x2x1w, r1_2x2x2w, a01_4x2x1w, a2_4x2x1w, b01_4x2x1w, b4_4x2x1w);
  // get_channel_4x2w(r48, r1_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r000 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r1_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r001 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 2);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r010 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 3);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r011 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 0);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r020 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 1);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r021 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 4);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r100 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 5);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r101 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 6);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r110 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 7);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r111 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r1_2x2x2w, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r120 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r1_2x2x2w, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_by_xy00z0_fp12_vec_v1 r121 = 0x", r64, SWORDS);

  // a0_8x1x1w[0][0] = VSET(7 , 7 , 1, 7 , 7 , 1, 1, 1);
  // a0_8x1x1w[1][0] = VSET(8 , 8 , 2, 8 , 8 , 2, 2, 2);
  // a1_8x1x1w[0][0] = VSET(9 , 9 , 3, 9 , 9 , 3, 3, 3);
  // a1_8x1x1w[1][0] = VSET(10, 10, 4, 10, 10, 4, 4, 4);
  // a2_8x1x1w[0][0] = VSET(11, 11, 5, 11, 11, 5, 5, 5);
  // a2_8x1x1w[1][0] = VSET(12, 12, 6, 12, 12, 6, 6, 6);

  // for (i = 1; i < NWORDS; i++) {
  //   a0_8x1x1w[0][i] = VZERO;
  //   a0_8x1x1w[1][i] = VZERO;
  //   a1_8x1x1w[0][i] = VZERO;
  //   a1_8x1x1w[1][i] = VZERO;
  //   a2_8x1x1w[0][i] = VZERO;
  //   a2_8x1x1w[1][i] = VZERO;
  // }

  // a0_4x2x1w[0] = VSET(8 , 7 , 8 , 7 , 2, 1, 2, 1);
  // a1_4x2x1w[0] = VSET(10, 9 , 10, 9 , 4, 3, 4, 3);
  // a2_4x2x1w[0] = VSET(12, 11, 12, 11, 6, 5, 6, 5);

  // for (i = 1; i < NWORDS; i++) {
  //   a0_4x2x1w[i] = VZERO;
  //   a1_4x2x1w[i] = VZERO;
  //   a2_4x2x1w[i] = VZERO;
  // }

  // mul_fp12_scalar(r, a, a);
  // mpi_print("* mul_fp12 r000 = 0x", r[0][0][0], SWORDS);
  // mpi_print("* mul_fp12 r001 = 0x", r[0][0][1], SWORDS);
  // mpi_print("* mul_fp12 r010 = 0x", r[0][1][0], SWORDS);
  // mpi_print("* mul_fp12 r011 = 0x", r[0][1][1], SWORDS);
  // mpi_print("* mul_fp12 r020 = 0x", r[0][2][0], SWORDS);
  // mpi_print("* mul_fp12 r021 = 0x", r[0][2][1], SWORDS);
  // mpi_print("* mul_fp12 r100 = 0x", r[1][0][0], SWORDS);
  // mpi_print("* mul_fp12 r101 = 0x", r[1][0][1], SWORDS);
  // mpi_print("* mul_fp12 r110 = 0x", r[1][1][0], SWORDS);
  // mpi_print("* mul_fp12 r111 = 0x", r[1][1][1], SWORDS);
  // mpi_print("* mul_fp12 r120 = 0x", r[1][2][0], SWORDS);
  // mpi_print("* mul_fp12 r121 = 0x", r[1][2][1], SWORDS);

  // mul_fp12_vec_v1(r01_4x2x1w, r2_4x2x1w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w);
  // get_channel_8x1w(r48, r2_4x2x1w, 0);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r000 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r2_4x2x1w, 1);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r001 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 0);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r010 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 1);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r011 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 2);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r020 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 3);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r021 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 4);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r100 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 5);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r101 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 6);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r110 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 7);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r111 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r2_4x2x1w, 2);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r120 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r2_4x2x1w, 3);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v1 r121 = 0x", r64, SWORDS);

  // mul_fp12_vec_v2(r01_4x2x1w, r2_2x2x2w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w);
  // get_channel_4x2w(r48, r2_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r000 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r001 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 0);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r010 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 1);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r011 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 2);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r020 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 3);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r021 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 4);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r100 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 5);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r101 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 6);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r110 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r01_4x2x1w, 7);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r111 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r120 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v2 r121 = 0x", r64, SWORDS);

  // mul_fp12_vec_v3(r001_2x2x2w, r101_2x2x2w, r2_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w);
  // get_channel_4x2w(r48, r001_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r000 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r001_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r001 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r001_2x2x2w, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r010 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r001_2x2x2w, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r011 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r020 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r021 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r101_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r100 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r101_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r101 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r101_2x2x2w, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r110 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r101_2x2x2w, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r111 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r120 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r2_2x2x2w, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v3 r121 = 0x", r64, SWORDS);

  // mul_fp12_vec_v4(r0_4x2x1w, r101_2x2x2w, r12_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w); 
  // get_channel_8x1w(r48, r0_4x2x1w, 0);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r000 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 1);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r001 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 2);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r010 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 3);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r011 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 4);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r020 = 0x", r64, SWORDS);
  // get_channel_8x1w(r48, r0_4x2x1w, 5);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r021 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r101_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r100 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r101_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r101 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r101_2x2x2w, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r110 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r101_2x2x2w, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r111 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r12_2x2x2w, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r120 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, r12_2x2x2w, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* mul_fp12_vec_v4 r121 = 0x", r64, SWORDS);
#endif 

  puts("\n=============================================================\n");
  puts("TIMING - FP12\n");

  printf("- cyclotomic_sqr_fp12_scalar:            ");
  LOAD_CACHE(cyclotomic_sqr_fp12_scalar(r, a), 10000);
  MEASURE_CYCLES(cyclotomic_sqr_fp12_scalar(r, a), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- cyclotomic_sqr_fp12_vec_v1:            ");
  LOAD_CACHE(cyclotomic_sqr_fp12_vec_v1(ra_1x2x2x2w, rbc_2x2x2x1w, a_1x2x2x2w, bc_2x2x2x1w), 10000);
  MEASURE_CYCLES(cyclotomic_sqr_fp12_vec_v1(ra_1x2x2x2w, rbc_2x2x2x1w, a_1x2x2x2w, bc_2x2x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- cyclotomic_sqr_fp12_vec_v2:            ");
  LOAD_CACHE(cyclotomic_sqr_fp12_vec_v2(ra_1x2x2x2w, rbc_2x2x2x1w, a_1x2x2x2w, bc_2x2x2x1w), 10000);
  MEASURE_CYCLES(cyclotomic_sqr_fp12_vec_v2(ra_1x2x2x2w, rbc_2x2x2x1w, a_1x2x2x2w, bc_2x2x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

#if COMPRESSED_CYCLOTOMIC_SQR
  printf("- compressed_cyclotomic_sqr_fp12_scalar: ");
  LOAD_CACHE(compressed_cyclotomic_sqr_fp12_scalar(r, a), 10000);
  MEASURE_CYCLES(compressed_cyclotomic_sqr_fp12_scalar(r, a), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- compressed_cyclotomic_sqr_fp12_vec_v1: ");
  LOAD_CACHE(compressed_cyclotomic_sqr_fp12_vec_v1(rbc_2x2x2x1w, bc_2x2x2x1w), 10000);
  MEASURE_CYCLES(compressed_cyclotomic_sqr_fp12_vec_v1(rbc_2x2x2x1w, bc_2x2x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);
#endif

  printf("- mul_fp12_scalar:                       ");
  LOAD_CACHE(mul_fp12_scalar(r, a, a), 10000);
  MEASURE_CYCLES(mul_fp12_scalar(r, a, a), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp12_vec_v1:                       ");
  LOAD_CACHE(mul_fp12_vec_v1(r01_4x2x1w, r2_4x2x1w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 10000);
  MEASURE_CYCLES(mul_fp12_vec_v1(r01_4x2x1w, r2_4x2x1w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);  

  printf("- mul_fp12_vec_v2:                       ");
  LOAD_CACHE(mul_fp12_vec_v2(r01_4x2x1w, r2_2x2x2w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 10000);
  MEASURE_CYCLES(mul_fp12_vec_v2(r01_4x2x1w, r2_2x2x2w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);  

  printf("- mul_fp12_vec_v3:                       ");
  LOAD_CACHE(mul_fp12_vec_v3(r001_2x2x2w, r101_2x2x2w, r2_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w), 10000);
  MEASURE_CYCLES(mul_fp12_vec_v3(r001_2x2x2w, r101_2x2x2w, r2_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);  

  printf("- mul_fp12_vec_v4:                       ");
  LOAD_CACHE(mul_fp12_vec_v4(r0_4x2x1w, r101_2x2x2w, r12_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w), 10000);
  MEASURE_CYCLES(mul_fp12_vec_v4(r0_4x2x1w, r101_2x2x2w, r12_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);  

  printf("- mul_by_xy00z0_fp12_scalar:             ");
  LOAD_CACHE(mul_by_xy00z0_fp12_scalar(r, a, b), 10000);
  MEASURE_CYCLES(mul_by_xy00z0_fp12_scalar(r, a, b), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_by_xy00z0_fp12_vec_v1:             ");
  LOAD_CACHE(mul_by_xy00z0_fp12_vec_v1(r0_4x2x1w, r1_2x2x2w, a01_4x2x1w, a2_4x2x1w, b01_4x2x1w, b4_4x2x1w), 10000);
  MEASURE_CYCLES(mul_by_xy00z0_fp12_vec_v1(r0_4x2x1w, r1_2x2x2w, a01_4x2x1w, a2_4x2x1w, b01_4x2x1w, b4_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_by_xy00z0_fp12_vec_v2:             ");
  LOAD_CACHE(mul_by_xy00z0_fp12_vec_v2(r0_4x2x1w, r1_2x2x2w, a_8x1x1w, b_8x1x1w), 10000);
  MEASURE_CYCLES(mul_by_xy00z0_fp12_vec_v2(r0_4x2x1w, r1_2x2x2w, a_8x1x1w, b_8x1x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp12_scalar:                       ");
  LOAD_CACHE(sqr_fp12_scalar(r, a), 10000);
  MEASURE_CYCLES(sqr_fp12_scalar(r, a), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- sqr_fp12_vec_v1:                       ");
  LOAD_CACHE(sqr_fp12_vec_v1(r0_4x2x1w, r1_4x2x1w, a0_4x2x1w, a1_4x2x1w), 10000);
  MEASURE_CYCLES(sqr_fp12_vec_v1(r0_4x2x1w, r1_4x2x1w, a0_4x2x1w, a1_4x2x1w), 100000);
  printf("#cycle = %ld\n", diff_cycles);
}

// ----------------------------------------------------------------------------

void test_timing_line()
{
  vec384fp6 line;
  POINTonE2 T[1], Q[1];
  POINTonE2_affine R[1];
  fp2_2x2x2w X1Y1, Z1, l0Y3, l1, l2, X3, Z3;
  fp2_2x2x2w Z1Y2, X2;
  fp2_4x2x1w X1Y1Z1, l0, l12, _X3, _Y3, _Z3;
  uint64_t r64[NWORDS];
  uint64_t r48[NWORDS];
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

#if 0
  puts("\nTEST - LINE\n");

  Q[0].X[0][0] = 1; Q[0].X[1][0] = 2;
  Q[0].Y[0][0] = 3; Q[0].Y[1][0] = 4;
  Q[0].Z[0][0] = 5; Q[0].Z[1][0] = 6;

  R[0].X[0][0] = 7; R[0].X[1][0] = 8 ;
  R[0].Y[0][0] = 9; R[0].Y[1][0] = 10;

  for (i = 1 ; i < SWORDS; i++) {
    Q[0].X[0][i] = Q[0].X[1][i] = 0;
    Q[0].Y[0][i] = Q[0].Y[1][i] = 0;
    Q[0].Z[0][i] = Q[0].Z[1][i] = 0;
    R[0].X[0][i] = R[0].X[1][i] = 0;
    R[0].Y[0][i] = R[0].Y[1][i] = 0;
  }

  X1Y1   [0] = VSET(0, 4, 0, 3, 0, 2  , 0, 1);
  Z1     [0] = VSET(0, 6, 0, 5, 0, 6  , 0, 5);
  Z1Y2   [0] = VSET(0, 6, 0, 5, 0, 10 , 0, 9);
  X2     [0] = VSET(0, 8, 0, 7, 0, 8  , 0, 7);
  X1Y1Z1 [0] = VSET(4, 3, 2, 1, 6, 5  , 4, 3);

  for (i = 1; i < VWORDS; i++) {
    X1Y1[i] = VZERO;
    Z1  [i] = VZERO;
    Z1Y2[i] = VZERO;
    X2  [i] = VZERO;
  }

  for (i = 1; i < NWORDS; i++) X1Y1Z1[i] = VZERO;

  line_dbl_scalar(line, T, Q);
  mpi_print("* line_dbl_scalar line00 = 0x", line[0][0], SWORDS);
  mpi_print("* line_dbl_scalar line01 = 0x", line[0][1], SWORDS);
  mpi_print("* line_dbl_scalar line10 = 0x", line[1][0], SWORDS);
  mpi_print("* line_dbl_scalar line11 = 0x", line[1][1], SWORDS);
  mpi_print("* line_dbl_scalar line20 = 0x", line[2][0], SWORDS);
  mpi_print("* line_dbl_scalar line21 = 0x", line[2][1], SWORDS);
  mpi_print("* line_dbl_scalar X30    = 0x", T[0].X[0] , SWORDS);
  mpi_print("* line_dbl_scalar X31    = 0x", T[0].X[1] , SWORDS);
  mpi_print("* line_dbl_scalar Y30    = 0x", T[0].Y[0] , SWORDS);
  mpi_print("* line_dbl_scalar Y31    = 0x", T[0].Y[1] , SWORDS);
  mpi_print("* line_dbl_scalar Z30    = 0x", T[0].Z[0] , SWORDS);
  mpi_print("* line_dbl_scalar Z31    = 0x", T[0].Z[1] , SWORDS);

  line_dbl_vec_v1(l0Y3, l1, l2, X3, Z3, X1Y1, Z1);
  get_channel_4x2w(r48, l0Y3, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 line00 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, l0Y3, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 line01 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, l1, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 line10 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, l1, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 line11 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, l2, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 line20 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, l2, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 line21 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, X3, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 X30    = 0x", r64, SWORDS);
  get_channel_4x2w(r48, X3, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 X31    = 0x", r64, SWORDS);
  get_channel_4x2w(r48, l0Y3, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 Y30    = 0x", r64, SWORDS);
  get_channel_4x2w(r48, l0Y3, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 Y31    = 0x", r64, SWORDS);
  get_channel_4x2w(r48, Z3, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 Z30    = 0x", r64, SWORDS);
  get_channel_4x2w(r48, Z3, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v1 Z31    = 0x", r64, SWORDS);

  line_dbl_vec_v2(l0, l12, _X3, _Y3, _Z3, X1Y1Z1);
  get_channel_8x1w(r48, l0, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 line00 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, l0, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 line01 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, l12, 4);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 line10 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, l12, 5);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 line11 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, l12, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 line20 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, l12, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 line21 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, _X3, 6);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 X30    = 0x", r64, SWORDS);
  get_channel_8x1w(r48, _X3, 7);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 X31    = 0x", r64, SWORDS);
  get_channel_8x1w(r48, _Y3, 6);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 Y30    = 0x", r64, SWORDS);
  get_channel_8x1w(r48, _Y3, 7);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 Y31    = 0x", r64, SWORDS);
  get_channel_8x1w(r48, _Z3, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 Z30    = 0x", r64, SWORDS);
  get_channel_8x1w(r48, _Z3, 1);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* line_dbl_vec_v2 Z31    = 0x", r64, SWORDS);

  // X1Y1[0] = VSET(0, 2 , 0, 1, 0, 4, 0, 3);

  // line_add_scalar(line, T, Q, R);
  // mpi_print("* line_add_scalar line00 = 0x", line[0][0], SWORDS);
  // mpi_print("* line_add_scalar line01 = 0x", line[0][1], SWORDS);
  // mpi_print("* line_add_scalar line10 = 0x", line[1][0], SWORDS);
  // mpi_print("* line_add_scalar line11 = 0x", line[1][1], SWORDS);
  // mpi_print("* line_add_scalar line20 = 0x", line[2][0], SWORDS);
  // mpi_print("* line_add_scalar line21 = 0x", line[2][1], SWORDS);
  // mpi_print("* line_add_scalar X30    = 0x", T[0].X[0] , SWORDS);
  // mpi_print("* line_add_scalar X31    = 0x", T[0].X[1] , SWORDS);
  // mpi_print("* line_add_scalar Y30    = 0x", T[0].Y[0] , SWORDS);
  // mpi_print("* line_add_scalar Y31    = 0x", T[0].Y[1] , SWORDS);
  // mpi_print("* line_add_scalar Z30    = 0x", T[0].Z[0] , SWORDS);
  // mpi_print("* line_add_scalar Z31    = 0x", T[0].Z[1] , SWORDS);

  // line_add_vec_v1(l0Y3, l1, X3, Z3, X1Y1, Z1Y2, X2);
  // get_channel_4x2w(r48, l0Y3, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 line00 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, l0Y3, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 line01 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, l1, 0);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 line10 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, l1, 2);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 line11 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, Z3, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 line20 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, Z3, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 line21 = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, X3, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 X30    = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, X3, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 X31    = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, l0Y3, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 Y30    = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, l0Y3, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 Y31    = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, Z3, 4);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 Z30    = 0x", r64, SWORDS);
  // get_channel_4x2w(r48, Z3, 6);
  // carryp_mpi48(r48);
  // conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  // mpi_print("* line_add_vec_v1 Z31    = 0x", r64, SWORDS);

#endif

  puts("\n=============================================================\n");
  puts("TIMING - LINE\n");

  printf("- line_dbl_scalar: ");
  LOAD_CACHE(line_dbl_scalar(line, T, Q), 10000);
  MEASURE_CYCLES(line_dbl_scalar(line, T, Q), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- line_dbl_vec_v1: ");
  LOAD_CACHE(line_dbl_vec_v1(l0Y3, l1, l2, X3, Z3, X1Y1, Z1), 10000);
  MEASURE_CYCLES(line_dbl_vec_v1(l0Y3, l1, l2, X3, Z3, X1Y1, Z1), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- line_dbl_vec_v2: ");
  LOAD_CACHE(line_dbl_vec_v2(l0, l12, _X3, _Y3, _Z3, X1Y1Z1), 10000);
  MEASURE_CYCLES(line_dbl_vec_v2(l0, l12, _X3, _Y3, _Z3, X1Y1Z1), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- line_add_scalar: ");
  LOAD_CACHE(line_add_scalar(line, T, Q, R), 10000);
  MEASURE_CYCLES(line_add_scalar(line, T, Q, R), 100000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- line_add_vec_v1: ");
  LOAD_CACHE(line_add_vec_v1(l0Y3, l1, X3, Z3, X1Y1, Z1Y2, X2), 10000);
  MEASURE_CYCLES(line_add_vec_v1(l0Y3, l1, X3, Z3, X1Y1, Z1Y2, X2), 100000);
  printf("#cycle = %ld\n", diff_cycles);
}

// ----------------------------------------------------------------------------

int main()
{
  test_pairing();
  timing_pairing();

#ifdef BENCHMARK
  test_timing_fp();
  test_timing_fp2();
  test_timing_fp4();
  test_timing_fp6();
#endif

  test_timing_fp12();
  test_timing_line();

  return 0;
}
