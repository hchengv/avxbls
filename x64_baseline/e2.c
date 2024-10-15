/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "point.h"
#include "fields.h"
#include "ec_ops.h"

const POINTonE2 BLS12_381_G2 = {    /* generator point [in Montgomery] */
{ /* (0x024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02
        b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8 << 384) % P */
  { TO_LIMB_T(0xf5f28fa202940a10), TO_LIMB_T(0xb3f5fb2687b4961a),
    TO_LIMB_T(0xa1a893b53e2ae580), TO_LIMB_T(0x9894999d1a3caee9),
    TO_LIMB_T(0x6f67b7631863366b), TO_LIMB_T(0x058191924350bcd7) },
  /* (0x13e02b6052719f607dacd3a088274f65596bd0d09920b61a
        b5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e << 384) % P */
  { TO_LIMB_T(0xa5a9c0759e23f606), TO_LIMB_T(0xaaa0c59dbccd60c3),
    TO_LIMB_T(0x3bb17e18e2867806), TO_LIMB_T(0x1b1ab6cc8541b367),
    TO_LIMB_T(0xc2b6ed0ef2158547), TO_LIMB_T(0x11922a097360edf3) }
},
{ /* (0x0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a7
        6d429a695160d12c923ac9cc3baca289e193548608b82801 << 384) % P */
  { TO_LIMB_T(0x4c730af860494c4a), TO_LIMB_T(0x597cfa1f5e369c5a),
    TO_LIMB_T(0xe7e6856caa0a635a), TO_LIMB_T(0xbbefb5e96e0d495f),
    TO_LIMB_T(0x07d3a975f0ef25a2), TO_LIMB_T(0x0083fd8e7e80dae5) },
  /* (0x0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af
        267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be << 384) % P */
  { TO_LIMB_T(0xadc0fc92df64b05d), TO_LIMB_T(0x18aa270a2b1461dc),
    TO_LIMB_T(0x86adac6a3be4eba0), TO_LIMB_T(0x79495c4ec93da33a),
    TO_LIMB_T(0xe7175850a43ccaed), TO_LIMB_T(0x0b2bc2a163de1bf2) },
},
{ { ONE_MONT_P }, { 0 } }
};

static void POINTonE2_from_Jacobian(POINTonE2 *out, const POINTonE2 *in)
{
    vec384x Z, ZZ;
    limb_t inf = vec_is_zero(in->Z, sizeof(in->Z));

    reciprocal_fp2(Z, in->Z);                           /* 1/Z */

    sqr_fp2(ZZ, Z);
    mul_fp2(out->X, in->X, ZZ);                         /* X = X/Z^2 */

    mul_fp2(ZZ, ZZ, Z);
    mul_fp2(out->Y, in->Y, ZZ);                         /* Y = Y/Z^3 */

    vec_select(out->Z, in->Z, BLS12_381_G2.Z,
                       sizeof(BLS12_381_G2.Z), inf);    /* Z = inf ? 0 : 1 */
}

void POINTonE2_to_affine(POINTonE2_affine *out, const POINTonE2 *in)
{
    POINTonE2 p;

    if (!vec_is_equal(in->Z, BLS12_381_Rx.p2, sizeof(in->Z))) {
        POINTonE2_from_Jacobian(&p, in);
        in = &p;
    }
    vec_copy(out, in, sizeof(*out));
}
