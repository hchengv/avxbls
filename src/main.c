#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "pairing.h"
#include "point.h"
#include "ec_ops.h"
#include "fp12_avx.h"

// ----------------------------------------------------------------------------
// for profiling

#ifdef PROFILING
uint64_t sqr_fp12_cycles;
uint64_t mul_fp12_cycles;
uint64_t cyclotomic_sqr_fp12_cycles;
uint64_t mul_by_xy00z0_fp12_cycles;
uint64_t inverse_fp12_cycles;
uint64_t frobenius_map_fp12_cycles;

uint64_t line_add_cycles;
uint64_t line_dbl_cycles;
uint64_t line_by_Px2_cycles;
#endif

#ifdef PROFILING
static void profiling_reset() {
  sqr_fp12_cycles            = 0;
  mul_fp12_cycles            = 0;
  cyclotomic_sqr_fp12_cycles = 0;
  mul_by_xy00z0_fp12_cycles  = 0;
  inverse_fp12_cycles        = 0;
  frobenius_map_fp12_cycles  = 0;

  line_add_cycles            = 0;
  line_dbl_cycles            = 0;
  line_by_Px2_cycles         = 0;
}

static void profiling_dump(uint64_t total_cycles, int iter){
  puts("");
  printf("sqr_fp12_cycles                = %lu (%.2f%%)\n", sqr_fp12_cycles              / iter, (double)(sqr_fp12_cycles            / iter) / total_cycles*100.0f);
  printf("mul_fp12_cycles                = %lu (%.2f%%)\n", mul_fp12_cycles              / iter, (double)(mul_fp12_cycles            / iter) / total_cycles*100.0f);
  printf("cyclotomic_sqr_fp12_cycles     = %lu (%.2f%%)\n", cyclotomic_sqr_fp12_cycles   / iter, (double)(cyclotomic_sqr_fp12_cycles / iter) / total_cycles*100.0f);
  printf("mul_by_xy00z0_fp12_cycles      = %lu (%.2f%%)\n", mul_by_xy00z0_fp12_cycles    / iter, (double)(mul_by_xy00z0_fp12_cycles  / iter) / total_cycles*100.0f);
  printf("inverse_fp12_cycles            = %lu (%.2f%%)\n", inverse_fp12_cycles          / iter, (double)(inverse_fp12_cycles        / iter) / total_cycles*100.0f);
  printf("frobenius_map_fp12_cycles      = %lu (%.2f%%)\n", frobenius_map_fp12_cycles    / iter, (double)(frobenius_map_fp12_cycles  / iter) / total_cycles*100.0f);

  printf("line_add_cycles                = %lu (%.2f%%)\n", line_add_cycles              / iter, (double)(line_add_cycles            / iter) / total_cycles*100.0f);
  printf("line_dbl_cycles                = %lu (%.2f%%)\n", line_dbl_cycles              / iter, (double)(line_dbl_cycles            / iter) / total_cycles*100.0f);
  printf("line_by_Px2_cycles             = %lu (%.2f%%)\n", line_by_Px2_cycles           / iter, (double)(line_by_Px2_cycles         / iter) / total_cycles*100.0f);

  puts("");
}
#endif

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

// only for generating a random point used for the correctness test,
// which means the constant time is not required

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

  // scalar k can be modified to be any non-0 value
  // uint64_t k[4] = { rand(), rand(), rand(), rand(), };
  // uint64_t k[4] = { 3 };
  uint64_t k[4] = {
    0x0123456789ABCDEF, 0x89ABCDEF01234567, 
    0x0123456789ABCDEF, 0x89ABCDEF01234567, };

  // currently use _P = BLS12_381_G1 and _Q = BLS12_381_G2 and k = 2 
  // to conduct a very simple test
  printf("\n=============================================================\n");
  printf("bilinear test:\n");
  printf("=============================================================\n");
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

  // print out parts of the result
  printf("e1[0][0][0] = ");
  for (int i = 6; i > 0; i-- ) printf("%016lX", e1[0][0][0][i]);
  printf("\n");

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

  printf("=============================================================\n");
}

void timing_pairing()
{
  uint64_t start_cycles, end_cycles, diff_cycles;
  int i;

  printf("timing measurement:\n");
  printf("=============================================================\n");

  POINTonE2_affine Q[1];
  POINTonE1_affine P[1];
  POINTonE2 _Q[1];
  POINTonE1 _P[1];
  vec384fp12 e1, f;

  POINTonE2_double(_Q, &BLS12_381_G2);
  POINTonE1_to_affine(P, &BLS12_381_G1);
  POINTonE2_to_affine(Q, _Q);

  printf("- miller_loop:        ");
  LOAD_CACHE(miller_loop_n(f, Q, P, 1), 10);
  #ifdef PROFILING
  profiling_reset();
  #endif
  MEASURE_CYCLES(miller_loop_n(f, Q, P, 1), 100);
  printf("  #cycle = %ld\n", diff_cycles);
  #ifdef PROFILING
  profiling_dump(diff_cycles, 100);
  #endif

  printf("- final_exp:          ");
  LOAD_CACHE(final_exp(e1, f), 10);
  #ifdef PROFILING
  profiling_reset();
  #endif
  MEASURE_CYCLES(final_exp(e1, f), 100);
  printf("  #cycle = %ld\n", diff_cycles);
  #ifdef PROFILING
  profiling_dump(diff_cycles, 100);
  #endif

  printf("- optimal_ate_pairing:");
  LOAD_CACHE(optimal_ate_pairing(e1, Q, P, 1), 10);
  MEASURE_CYCLES(optimal_ate_pairing(e1, Q, P, 1), 100);
  printf("  #cycle = %ld\n", diff_cycles);

  printf("=============================================================\n");
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

void test_fp()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS], z64[2*SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS], z48[2*NWORDS];
  __m512i a_8x1w[NWORDS], b_8x1w[NWORDS], r_8x1w[NWORDS], z_8x1w[2*NWORDS];
  __m512i a_4x2w[VWORDS], b_4x2w[VWORDS], r_4x2w[VWORDS], z_4x2w[3*VWORDS];
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

  puts("\nFP TEST\n");

#if 0
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
}

// ----------------------------------------------------------------------------

void test_fp2()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS], z64[2*SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS], z48[2*NWORDS];
  __m512i a_4x2x1w[NWORDS], b_4x2x1w[NWORDS], r_4x2x1w[NWORDS], z_4x2x1w[2*NWORDS];
  __m512i a_2x2x2w[VWORDS], b_2x2x2w[VWORDS], r_2x2x2w[VWORDS];
  __m512i a_2x4x1w[NWORDS], b_2x4x1w[NWORDS], z_2x4x1w[2*NWORDS];
  fp2x2_2x2x2w aa_2x2x2w, rr_2x2x2w;
  int i;

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

  puts("\nFP2 TEST\n");

#if 0

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
}

// ----------------------------------------------------------------------------

void test_fp4()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS], z64[2*SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS], z48[2*NWORDS];
  __m512i a_2x2x2x1w[NWORDS], b_2x2x2x1w[NWORDS], r_2x2x2x1w[NWORDS], z[2*NWORDS];
  int i;
  uint64_t start_cycles, end_cycles, diff_cycles;


  conv_64to48_mpi(a48, a64, NWORDS, SWORDS);
  conv_64to48_mpi(b48, b64, NWORDS, SWORDS);

  for (i = 0; i < NWORDS; i++) {
    a_2x2x2x1w[i] = VSET(0, 0, 0, 0, b48[i], a48[i], b48[i], a48[i]);
  }

  puts("\nFP4 TEST\n");

#if 0
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
}

void test_fp6()
{
  uint64_t r64[NWORDS], z64[2*SWORDS];
  uint64_t r48[NWORDS], z48[2*NWORDS];
  fp2_8x1x1w a0_8x1x1w, a1_8x1x1w, a2_8x1x1w;
  fp2x2_8x1x1w z01_8x1x1w, z2_8x1x1w;
  fp2_4x2x1w a0_4x2x1w, a1_4x2x1w, a2_4x2x1w, r01_4x2x1w, r2_4x2x1w;
  fp2x2_4x2x1w z01_4x2x1w, z2_4x2x1w;
  fp2_2x2x2w a0_2x2x2w, a1_2x2x2w, a2_2x2x2w, r01_2x2x2w, r2_2x2x2w;
  fp2x2_2x2x2w z01_2x2x2w, z2_2x2x2w;
  vec768fp6 r; 
  vec384fp6 a = {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}};
  int i;

  a0_8x1x1w[0][0] = VSET1(1);
  a0_8x1x1w[1][0] = VSET1(2);
  a1_8x1x1w[0][0] = VSET1(3);
  a1_8x1x1w[1][0] = VSET1(4);
  a2_8x1x1w[0][0] = VSET1(5);
  a2_8x1x1w[1][0] = VSET1(6);

  for(i = 1; i < NWORDS; i++) {
    a0_8x1x1w[0][i] = VZERO;
    a0_8x1x1w[1][i] = VZERO;
    a1_8x1x1w[0][i] = VZERO;
    a1_8x1x1w[1][i] = VZERO;
    a2_8x1x1w[0][i] = VZERO;
    a2_8x1x1w[1][i] = VZERO;
  }

  a0_4x2x1w[0] = VSET(0, 0, 0, 0, 2, 1, 2, 1);
  a1_4x2x1w[0] = VSET(0, 0, 0, 0, 4, 3, 4, 3);
  a2_4x2x1w[0] = VSET(0, 0, 0, 0, 6, 5, 6, 5);

  for (i = 1; i < NWORDS; i++) {
    a0_4x2x1w[i] = VZERO;
    a1_4x2x1w[i] = VZERO;
    a2_4x2x1w[i] = VZERO;
  }

  a0_2x2x2w[0] = VSET(0, 2, 0, 1, 0, 2, 0, 1);
  a1_2x2x2w[0] = VSET(0, 4, 0, 3, 0, 4, 0, 3);
  a2_2x2x2w[0] = VSET(0, 6, 0, 5, 0, 6, 0, 5);

  for (i = 1; i < VWORDS; i++) {
    a0_2x2x2w[i] = VZERO;
    a1_2x2x2w[i] = VZERO;
    a2_2x2x2w[i] = VZERO;
  }

  puts("\nFP6 TEST\n");

#if 0

  mul_fp6x2(r, a, a);
  mpi_print("* mul_fp6x2 r00 = 0x", r[0][0], 2*SWORDS);
  mpi_print("* mul_fp6x2 r01 = 0x", r[0][1], 2*SWORDS);
  mpi_print("* mul_fp6x2 r10 = 0x", r[1][0], 2*SWORDS);
  mpi_print("* mul_fp6x2 r11 = 0x", r[1][1], 2*SWORDS);
  mpi_print("* mul_fp6x2 r20 = 0x", r[2][0], 2*SWORDS);
  mpi_print("* mul_fp6x2 r21 = 0x", r[2][1], 2*SWORDS);

  mul_fp6x2_4x2x1x1w(z01_8x1x1w, z2_8x1x1w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_8x1x1w[0][i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_4x2x1x1w z00 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_8x1x1w[1][i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_4x2x1x1w z01 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_8x1x1w[0][i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_4x2x1x1w z10 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_8x1x1w[1][i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_4x2x1x1w z11 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_8x1x1w[0][i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_4x2x1x1w z20 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_8x1x1w[1][i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_4x2x1x1w z21 = 0x", z64, 2*SWORDS);

  mul_fp6x2_2x2x2x1w(z01_4x2x1w, z2_4x2x1w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w z00 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w z01 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[2];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w z10 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z01_4x2x1w[i])[3];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w z11 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_4x2x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w z20 = 0x", z64, 2*SWORDS);
  for (i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z2_4x2x1w[i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w z21 = 0x", z64, 2*SWORDS);
  redc_fpx2_8x1w(r01_4x2x1w, z01_4x2x1w);
  redc_fpx2_8x1w(r2_4x2x1w, z2_4x2x1w);
  for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r01_4x2x1w[i])[0];
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w r00 = 0x", r64, SWORDS);
  for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r01_4x2x1w[i])[1];
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w r01 = 0x", r64, SWORDS);
  for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r01_4x2x1w[i])[2];
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w r10 = 0x", r64, SWORDS);
  for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r01_4x2x1w[i])[3];
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w r11 = 0x", r64, SWORDS);
  for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r2_4x2x1w[i])[0];
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w r20 = 0x", r64, SWORDS);
  for (i = 0; i < NWORDS; i++) r48[i] = ((uint64_t *)&r2_4x2x1w[i])[1];
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_2x2x2x1w r21 = 0x", r64, SWORDS);

  mul_fp6x2_1x2x2x2w(z01_2x2x2w, z2_2x2x2w, a0_2x2x2w, a1_2x2x2w, a2_2x2x2w);
  redc_fpx2_4x2w(r01_2x2x2w, z01_2x2x2w);
  redc_fpx2_4x2w(r2_2x2x2w, z2_2x2x2w);
  get_channel_4x2w(r48, r01_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_1x2x2x2w r00 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r01_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_1x2x2x2w r01 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r01_2x2x2w, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_1x2x2x2w r10 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r01_2x2x2w, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_1x2x2x2w r11 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r2_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_1x2x2x2w r20 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r2_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp6x2_1x2x2x2w r21 = 0x", r64, SWORDS);
#endif
}

// ----------------------------------------------------------------------------

void test_fp12()
{
  vec384fp12 r;
  vec384fp12 a = {
    {{{1}, {2}}, {{3}, {4 }}, {{5 }, {6 }}},
    {{{7}, {8}}, {{9}, {10}}, {{11}, {12}}}};
  __m512i ra_1x2x2x2w[VWORDS], a_1x2x2x2w[VWORDS];
  __m512i rbc_2x2x2x1w[NWORDS], bc_2x2x2x1w[NWORDS];

  uint64_t r64[NWORDS], z64[2*SWORDS];
  uint64_t r48[NWORDS], z48[2*NWORDS];
  fp2_8x1x1w a0_8x1x1w, a1_8x1x1w, a2_8x1x1w;
  fp2_4x2x1w r01_4x2x1w, r2_4x2x1w;
  fp2_2x2x2w r2_2x2x2w;
  fp2_4x2x1w a0_4x2x1w, a1_4x2x1w, a2_4x2x1w;
  fp2_2x2x2w r001_2x2x2w, r02_2x2x2w, r101_2x2x2w, r12_2x2x2w;
  int i;

  uint64_t start_cycles, end_cycles, diff_cycles;

#if 1
  puts("\nFP12 TEST\n");

  a0_8x1x1w[0][0] = VSET(7 , 7 , 1, 7 , 7 , 1, 1, 1);
  a0_8x1x1w[1][0] = VSET(8 , 8 , 2, 8 , 8 , 2, 2, 2);
  a1_8x1x1w[0][0] = VSET(9 , 9 , 3, 9 , 9 , 3, 3, 3);
  a1_8x1x1w[1][0] = VSET(10, 10, 4, 10, 10, 4, 4, 4);
  a2_8x1x1w[0][0] = VSET(11, 11, 5, 11, 11, 5, 5, 5);
  a2_8x1x1w[1][0] = VSET(12, 12, 6, 12, 12, 6, 6, 6);

  for (i = 1; i < NWORDS; i++) {
    a0_8x1x1w[0][i] = VZERO;
    a0_8x1x1w[1][i] = VZERO;
    a1_8x1x1w[0][i] = VZERO;
    a1_8x1x1w[1][i] = VZERO;
    a2_8x1x1w[0][i] = VZERO;
    a2_8x1x1w[1][i] = VZERO;
  }

  a0_4x2x1w[0] = VSET(8 , 7 , 8 , 7 , 2, 1, 2, 1);
  a1_4x2x1w[0] = VSET(10, 9 , 10, 9 , 4, 3, 4, 3);
  a2_4x2x1w[0] = VSET(12, 11, 12, 11, 6, 5, 6, 5);

  for (i = 1; i < NWORDS; i++) {
    a0_4x2x1w[i] = VZERO;
    a1_4x2x1w[i] = VZERO;
    a2_4x2x1w[i] = VZERO;
  }

  mul_fp12_scalar(r, a, a);
  mpi_print("* mul_fp12 r000 = 0x", r[0][0][0], SWORDS);
  mpi_print("* mul_fp12 r001 = 0x", r[0][0][1], SWORDS);
  mpi_print("* mul_fp12 r010 = 0x", r[0][1][0], SWORDS);
  mpi_print("* mul_fp12 r011 = 0x", r[0][1][1], SWORDS);
  mpi_print("* mul_fp12 r020 = 0x", r[0][2][0], SWORDS);
  mpi_print("* mul_fp12 r021 = 0x", r[0][2][1], SWORDS);
  mpi_print("* mul_fp12 r100 = 0x", r[1][0][0], SWORDS);
  mpi_print("* mul_fp12 r101 = 0x", r[1][0][1], SWORDS);
  mpi_print("* mul_fp12 r110 = 0x", r[1][1][0], SWORDS);
  mpi_print("* mul_fp12 r111 = 0x", r[1][1][1], SWORDS);
  mpi_print("* mul_fp12 r120 = 0x", r[1][2][0], SWORDS);
  mpi_print("* mul_fp12 r121 = 0x", r[1][2][1], SWORDS);

  mul_fp12_vec_v1(r01_4x2x1w, r2_4x2x1w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w);
  get_channel_8x1w(r48, r2_4x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r000 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r2_4x2x1w, 1);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r001 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r010 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 1);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r011 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r020 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r021 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 4);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r100 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 5);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r101 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 6);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r110 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 7);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r111 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r2_4x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r120 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r2_4x2x1w, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v1 r121 = 0x", r64, SWORDS);

  mul_fp12_vec_v3(r01_4x2x1w, r2_2x2x2w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w);
  get_channel_4x2w(r48, r2_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r000 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r2_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r001 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r010 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 1);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r011 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r020 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r021 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 4);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r100 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 5);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r101 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 6);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r110 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r01_4x2x1w, 7);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r111 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r2_2x2x2w, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r120 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r2_2x2x2w, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v3 r121 = 0x", r64, SWORDS);

  mul_fp12_vec_v4(r001_2x2x2w, r02_2x2x2w, r101_2x2x2w, r12_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w);
  get_channel_4x2w(r48, r001_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r000 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r001_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r001 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r001_2x2x2w, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r010 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r001_2x2x2w, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r011 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r02_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r020 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r02_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r021 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r101_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r100 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r101_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r101 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r101_2x2x2w, 4);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r110 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r101_2x2x2w, 6);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r111 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r12_2x2x2w, 0);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r120 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r12_2x2x2w, 2);
  carryp_mpi48(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp12_vec_v4 r121 = 0x", r64, SWORDS);
#endif 

  puts("\nFP12 TIMING\n");

  printf("- cyclotomic_sqr_fp12_scalar: ");
  LOAD_CACHE(cyclotomic_sqr_fp12_scalar(r, a), 1000);
  MEASURE_CYCLES(cyclotomic_sqr_fp12_scalar(r, a), 10000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- cyclotomic_sqr_fp12_vector: ");
  LOAD_CACHE(cyclotomic_sqr_fp12_vector(r, a), 1000);
  MEASURE_CYCLES(cyclotomic_sqr_fp12_vector(r, a), 10000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- cyclotomic_sqr_fp12_vec_v1: ");
  LOAD_CACHE(cyclotomic_sqr_fp12_vec_v1(ra_1x2x2x2w, rbc_2x2x2x1w, a_1x2x2x2w, bc_2x2x2x1w), 1000);
  MEASURE_CYCLES(cyclotomic_sqr_fp12_vec_v1(ra_1x2x2x2w, rbc_2x2x2x1w, a_1x2x2x2w, bc_2x2x2x1w), 10000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- cyclotomic_sqr_fp12_vec_v2: ");
  LOAD_CACHE(cyclotomic_sqr_fp12_vec_v2(ra_1x2x2x2w, rbc_2x2x2x1w, a_1x2x2x2w, bc_2x2x2x1w), 1000);
  MEASURE_CYCLES(cyclotomic_sqr_fp12_vec_v2(ra_1x2x2x2w, rbc_2x2x2x1w, a_1x2x2x2w, bc_2x2x2x1w), 10000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp12_scalar: ");
  LOAD_CACHE(mul_fp12_scalar(r, a, a), 1000);
  MEASURE_CYCLES(mul_fp12_scalar(r, a, a), 10000);
  printf("#cycle = %ld\n", diff_cycles);

  printf("- mul_fp12_vec_v1: ");
  LOAD_CACHE(mul_fp12_vec_v1(r01_4x2x1w, r2_4x2x1w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 1000);
  MEASURE_CYCLES(mul_fp12_vec_v1(r01_4x2x1w, r2_4x2x1w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 10000);
  printf("#cycle = %ld\n", diff_cycles);  

  printf("- mul_fp12_vec_v2: ");
  LOAD_CACHE(mul_fp12_vec_v2(r01_4x2x1w, r2_2x2x2w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 1000);
  MEASURE_CYCLES(mul_fp12_vec_v2(r01_4x2x1w, r2_2x2x2w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 10000);
  printf("#cycle = %ld\n", diff_cycles); 

  printf("- mul_fp12_vec_v3: ");
  LOAD_CACHE(mul_fp12_vec_v3(r01_4x2x1w, r2_2x2x2w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 1000);
  MEASURE_CYCLES(mul_fp12_vec_v3(r01_4x2x1w, r2_2x2x2w, a0_8x1x1w, a1_8x1x1w, a2_8x1x1w), 10000);
  printf("#cycle = %ld\n", diff_cycles);  

  printf("- mul_fp12_vec_v4: ");
  LOAD_CACHE(mul_fp12_vec_v4(r001_2x2x2w, r02_2x2x2w, r101_2x2x2w, r12_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w), 1000);
  MEASURE_CYCLES(mul_fp12_vec_v4(r001_2x2x2w, r02_2x2x2w, r101_2x2x2w, r12_2x2x2w, a0_4x2x1w, a1_4x2x1w, a2_4x2x1w), 10000);
  printf("#cycle = %ld\n", diff_cycles);  
}

// ----------------------------------------------------------------------------

int main()
{
  test_pairing();
  timing_pairing();

  // test_fp();
  // test_fp2();
  // test_fp4();
  // test_fp6();
  test_fp12();

  return 0;
}
