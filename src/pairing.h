#ifndef __BLS12_381_ASM_PAIRING_H__
#define __BLS12_381_ASM_PAIRING_H__

#include "fields.h"
#include "point.h"
#include "consts.h"
#include "fp12_avx.h"

#define line_dbl line_dbl_vector
void line_dbl_scalar(vec384fp6 line, POINTonE2 *T, const POINTonE2 *Q);
void line_dbl_vector(vec384fp6 line, POINTonE2 *T, const POINTonE2 *Q);

#define line_add line_add_vector
void line_add_scalar(vec384fp6 line, POINTonE2 *T, const POINTonE2 *R,
                                                   const POINTonE2_affine *Q);
void line_add_vector(vec384fp6 line, POINTonE2 *T, const POINTonE2 *R,
                                                   const POINTonE2_affine *Q);
                                                   

void miller_loop_n(vec384fp12 ret, const POINTonE2_affine Q[],
                                   const POINTonE1_affine P[], size_t n);
                                   
void final_exp(vec384fp12 ret, const vec384fp12 f);

void optimal_ate_pairing(vec384fp12 ret, POINTonE2_affine Q[], 
                                         POINTonE1_affine P[], size_t n);

#endif 
