#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "pairing.h"
#include "point.h"
#include "ec_ops.h"

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
// for printing mpi

static void mpi_print(const char *c, const unsigned long long *a, int len)
{
  int i;

  printf("%s", c);
  for (i = len-1; i > 0; i--) printf("%016llX", a[i]);
  printf("%016llX\n", a[0]);
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

  // scalar k can be modified to be any non-0 value
  // uint64_t k[4] = { rand(), rand(), rand(), rand(), };
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

#if 1
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

#if 1
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

#if 1
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
  LOAD_CACHE(miller_loop_n(f, Q, P, 1), 1000);
  #ifdef PROFILING
  profiling_reset();
  #endif
  MEASURE_CYCLES(miller_loop_n(f, Q, P, 1), 10000);
  printf("  #cycle = %ld\n", diff_cycles);
  #ifdef PROFILING
  profiling_dump(diff_cycles, 10000);
  #endif

  printf("- final_exp:          ");
  LOAD_CACHE(final_exp(e1, f), 1000);
  #ifdef PROFILING
  profiling_reset();
  #endif
  MEASURE_CYCLES(final_exp(e1, f), 10000);
  printf("  #cycle = %ld\n", diff_cycles);
  #ifdef PROFILING
  profiling_dump(diff_cycles, 10000);
  #endif

  printf("- optimal_ate_pairing:");
  LOAD_CACHE(optimal_ate_pairing(e1, Q, P, 1), 1000);
  MEASURE_CYCLES(optimal_ate_pairing(e1, Q, P, 1), 10000);
  printf("  #cycle = %ld\n", diff_cycles);

  printf("=============================================================\n");
}

// ----------------------------------------------------------------------------

int main()
{
  test_pairing();
  timing_pairing();

  return 0;
}
