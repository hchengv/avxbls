#include <stdio.h>
#include <string.h>

#include "pairing.h"
#include "point.h"
#include "ec_ops.h"


// for measuring CPU cycles

static uint64_t rdtsc() {
  uint64_t tsc = 0;

  __asm__ __volatile__( "rdcycle %0" : "=r" (tsc) );

  return tsc;
}

#define LOAD_CACHE(X, ITER) for (i = 0; i < (ITER); i++) (X)

#define MEASURE_CYCLES(X, ITER)                    \
  start_cycles = rdtsc();                          \
  for (i = 0; i < (ITER); i++) (X);                \
  end_cycles = rdtsc();                            \
  diff_cycles = (end_cycles-start_cycles)/(ITER)


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

  // scalar k can be modified to be any non-0 value
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

int main()
{
  test_pairing();

  return 0;
}
