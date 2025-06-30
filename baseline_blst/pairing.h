#ifndef __BLS12_381_ASM_PAIRING_H__
#define __BLS12_381_ASM_PAIRING_H__

#include "fields.h"
#include "point.h"
#include "consts.h"


// use Fp12 compressed cyclotomic squaring or not  
#ifdef PROFILING
#define COMPRESSED_CYCLOTOMIC_SQR 0
#else
#define COMPRESSED_CYCLOTOMIC_SQR 0
#endif

void miller_loop_n(vec384fp12 ret, const POINTonE2_affine Q[],
                                   const POINTonE1_affine P[], size_t n);
                                   
void final_exp(vec384fp12 ret, const vec384fp12 f);

void optimal_ate_pairing(vec384fp12 ret, POINTonE2_affine Q[], 
                                         POINTonE1_affine P[], size_t n);

#endif 
