#include "fp12_avx.h"
#include <string.h>


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

void fp_test()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS], z64[2*SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS], z48[2*NWORDS];
  __m512i a_8x1w[NWORDS], b_8x1w[NWORDS], r_8x1w[NWORDS], z_8x1w[2*NWORDS];
  __m512i a_4x2w[VWORDS], b_4x2w[VWORDS], r_4x2w[VWORDS];
  int i;

  mpi_conv_64to48(a48, a64, NWORDS, SWORDS);
  mpi_conv_64to48(b48, b64, NWORDS, SWORDS);

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
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* add_fp_8x1w r0 = 0x", r64, SWORDS);

  sub_fp_8x1w(r_8x1w, a_8x1w, b_8x1w);
  get_channel_8x1w(r48, r_8x1w, 0);
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* sub_fp_8x1w r0 = 0x", r64, SWORDS);

  mul_mp_8x1w_v1(z_8x1w, a_8x1w, b_8x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_8x1w[i])[0];
  mpi_conv_48to64(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_mp_8x1w_v1 r0 = 0x", z64, 2*SWORDS);

  mul_mp_8x1w_v2(z_8x1w, a_8x1w, b_8x1w);
  for(i = 0; i < 2*NWORDS; i++) z48[i] = ((uint64_t *)&z_8x1w[i])[0];
  mpi_conv_48to64(z64, z48, 2*SWORDS, 2*NWORDS);
  mpi_print("* mul_mp_8x1w_v2 r0 = 0x", z64, 2*SWORDS);

  add_fp_4x2w(r_4x2w, a_4x2w, b_4x2w);
  get_channel_4x2w(r48, r_4x2w, 0);
  mpi48_carryp(r48);
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* add_fp_4x2w r0 = 0x", r64, SWORDS);

}

// ----------------------------------------------------------------------------

void fp2_test()
{
  uint64_t a64[SWORDS] = { TV_A }, b64[SWORDS] = { TV_B }, r64[SWORDS];
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS];
  __m512i a_4x2x1w[NWORDS], b_4x2x1w[NWORDS], r_4x2x1w[NWORDS];
  __m512i a_2x2x2w[NWORDS], b_2x2x2w[NWORDS], r_2x2x2w[NWORDS];
  int i;

  mpi_conv_64to48(a48, a64, NWORDS, SWORDS);
  mpi_conv_64to48(b48, b64, NWORDS, SWORDS);

  for (i = 0; i < NWORDS; i++) {
    a_4x2x1w[i] = VSET1(a48[i]);
    b_4x2x1w[i] = VSET1(b48[i]);
  }

  for (i = 0; i < VWORDS; i++) {
    a_2x2x2w[i] = VSET(0, 0, a48[i+VWORDS], a48[i], 0, 0, a48[i+VWORDS], a48[i]);
    b_2x2x2w[i] = VSET(0, 0, b48[i+VWORDS], b48[i], 0, 0, b48[i+VWORDS], b48[i]);
  }

  assa_fp2_4x2x1w(r_4x2x1w, a_4x2x1w, b_4x2x1w);
  get_channel_8x1w(r48, r_4x2x1w, 0);
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* assa_fp2_4x2x1w r0 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_4x2x1w, 2);
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* assa_fp2_4x2x1w r2 = 0x", r64, SWORDS);

  as_fp2_2x2x2w(r_2x2x2w, a_2x2x2w, b_2x2x2w);
  get_channel_4x2w(r48, r_2x2x2w, 0);
  mpi48_carryp(r48);
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* as_fp2_2x2x2w r0 = 0x", r64, SWORDS);
  get_channel_4x2w(r48, r_2x2x2w, 4);
  mpi48_carryp(r48);
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* as_fp2_2x2x2w r4 = 0x", r64, SWORDS);


  for (i = 0; i < NWORDS; i++) 
    a_4x2x1w[i] = VSET(0, 0, 0, 0, 0, 0, b48[i], a48[i]);
  sqr_fp2x2_4x2x1w(r_4x2x1w, a_4x2x1w);
  get_channel_8x1w(r48, r_4x2x1w, 0);
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp2x2_4x2x1w r0 = 0x", r64, SWORDS);
  get_channel_8x1w(r48, r_4x2x1w, 1);
  mpi_conv_48to64(r64, r48, SWORDS, NWORDS);
  mpi_print("* sqr_fp2x2_4x2x1w r1 = 0x", r64, SWORDS);
}

// ----------------------------------------------------------------------------

int main()
{
  fp_test();
  fp2_test();

  return 0;
}