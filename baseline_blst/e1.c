/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "point.h"
#include "fields.h"

/*
 * y^2 = x^3 + B
 */
const POINTonE1 BLS12_381_G1 = {    /* generator point [in Montgomery] */
  /* (0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905
   *    a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb << 384) % P */
  { TO_LIMB_T(0x5cb38790fd530c16), TO_LIMB_T(0x7817fc679976fff5),
    TO_LIMB_T(0x154f95c7143ba1c1), TO_LIMB_T(0xf0ae6acdf3d0e747),
    TO_LIMB_T(0xedce6ecc21dbf440), TO_LIMB_T(0x120177419e0bfb75) },
  /* (0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af6
   *    00db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1 << 384) % P */
  { TO_LIMB_T(0xbaac93d50ce72271), TO_LIMB_T(0x8c22631a7918fd8e),
    TO_LIMB_T(0xdd595f13570725ce), TO_LIMB_T(0x51ac582950405194),
    TO_LIMB_T(0x0e1c8c3fad0059c0), TO_LIMB_T(0x0bbc3efc5008a26a) },
  { ONE_MONT_P }
};

static void POINTonE1_from_Jacobian(POINTonE1 *out, const POINTonE1 *in)
{
    vec384 Z, ZZ;
    limb_t inf = vec_is_zero(in->Z, sizeof(in->Z));

    reciprocal_fp(Z, in->Z);                            /* 1/Z   */

    sqr_fp(ZZ, Z);
    mul_fp(out->X, in->X, ZZ);                          /* X = X/Z^2 */

    mul_fp(ZZ, ZZ, Z);
    mul_fp(out->Y, in->Y, ZZ);                          /* Y = Y/Z^3 */

    vec_select(out->Z, in->Z, BLS12_381_G1.Z,
                       sizeof(BLS12_381_G1.Z), inf);    /* Z = inf ? 0 : 1 */
}

void POINTonE1_to_affine(POINTonE1_affine *out, const POINTonE1 *in)
{
    POINTonE1 p;

    if (!vec_is_equal(in->Z, BLS12_381_Rx.p, sizeof(in->Z))) {
        POINTonE1_from_Jacobian(&p, in);
        in = &p;
    }
    vec_copy(out, in, sizeof(*out));
}
