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

  add_fp_8x1w(r_8x1w, a_8x1w, b_8x1w);
  get_channel_8x1w(r48, r_8x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* add_fp_8x1w r0 = 0x", r64, SWORDS);

  sub_fp_8x1w(r_8x1w, a_8x1w, b_8x1w);
  get_channel_8x1w(r48, r_8x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sub_fp_8x1w r0 = 0x", r64, SWORDS);

  mul_mp_8x1w_v1(z_8x1w, a_8x1w, b_8x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_8x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_mp_8x1w_v1 r0 = 0x", z64, 2*SWORDS);

  mul_mp_8x1w_v3(z_8x1w, a_8x1w, b_8x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_8x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_mp_8x1w_v3 r0 = 0x", z64, 2*SWORDS);

  redc_fpx2_8x1w(r_8x1w, z_8x1w);
  get_channel_8x1w(r48, r_8x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* redc_fpx2_8x1w r0 = 0x", r64, SWORDS);

  add_fp_4x2w(r_4x2w, a_4x2w, b_4x2w);
  get_channel_4x2w(r48, r_4x2w, 0);
  mpi48_carryp(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* add_fp_4x2w r0 = 0x", r64, SWORDS);

  mul_mp_4x2w(z_4x2w, a_4x2w, b_4x2w);
  redc_fpx2_4x2w(r_4x2w, z_4x2w);
  get_channel_4x2w(r48, r_4x2w, 0);
  mpi48_carryp(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* mul_fp_4x2w r0 = 0x", r64, SWORDS);
}

// ----------------------------------------------------------------------------

void test_fp2()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS], z64[2*SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS], z48[2*NWORDS];
  __m512i a_4x2x1w[NWORDS], b_4x2x1w[NWORDS], r_4x2x1w[NWORDS], z_4x2x1w[2*NWORDS];
  __m512i a_2x2x2w[VWORDS], b_2x2x2w[VWORDS], r_2x2x2w[VWORDS];
  __m512i a_2x4x1w[NWORDS], b_2x4x1w[NWORDS], z_2x4x1w[2*NWORDS];
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

  puts("\nFP2 TEST\n");

  assa_fp2_4x2x1w(r_4x2x1w, a_4x2x1w, b_4x2x1w);
  get_channel_8x1w(r48, r_4x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* assa_fp2_4x2x1w r0 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_4x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* assa_fp2_4x2x1w r2 = 0x", r64, SWORDS);

  as_fp2_2x2x2w(r_2x2x2w, a_2x2x2w, b_2x2x2w);
  get_channel_4x2w(r48, r_2x2x2w, 0);
  mpi48_carryp(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* as_fp2_2x2x2w r0 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r_2x2x2w, 4);
  mpi48_carryp(r48);
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
  mpi48_carryp(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp2_2x2x2w r0 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r_2x2x2w, 2);
  mpi48_carryp(r48);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp2_2x2x2w r2 = 0x", r64, SWORDS);

  mul_by_u_plus_1_fp2x2_4x2x1w(z_4x2x1w, z_4x2x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_4x2x1w[i])[0];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_u_plus_1_fp2x2_4x2x1w r0 = 0x", z64, 2*SWORDS);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_4x2x1w[i])[1];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_by_u_plus_1_fp2x2_4x2x1w r1 = 0x", z64, 2*SWORDS);

  mul_fp2x2_2x4x1w(z_2x4x1w, a_2x4x1w, b_2x4x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_2x4x1w[i])[2];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp2x2_2x4x1w r2 = 0x", z64, 2*SWORDS);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_2x4x1w[i])[3];
  conv_48to64_mpi(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_fp2x2_2x4x1w r3 = 0x", z64, 2*SWORDS);
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

  sqr_fp4_2x2x2x1w(r_2x2x2x1w, a_2x2x2x1w);
  get_channel_8x1w(r48, r_2x2x2x1w, 0);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w r00 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 1);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w r01 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 2);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w r10 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_2x2x2x1w, 3);
  conv_48to64_mpi(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp4_2x2x2x1w r11 = 0x", r64, SWORDS);
}

// ----------------------------------------------------------------------------

int main()
{
  // test_pairing();
  // timing_pairing();

  test_fp();
  // test_fp2();
  // test_fp4();

  return 0;
}
