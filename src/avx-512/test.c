#include "fp12_avx.h"
#include <string.h>


// ----------------------------------------------------------------------------
// test vectors

// a := 0x1123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF
#define TV_A \
0x0123456789ABCDEF, 0x0123456789ABCDEF, 0x0123456789ABCDEF, \
0x0123456789ABCDEF, 0x0123456789ABCDEF, 0x1123456789ABCDEF

// b:= 0x19CDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789
#define TV_B \
0xABCDEF0123456789, 0xABCDEF0123456789, 0xABCDEF0123456789, \
0xABCDEF0123456789, 0xABCDEF0123456789, 0x19CDEF0123456789

// ----------------------------------------------------------------------------

void fp_test()
{
  uint64_t a64[6] = { TV_A }, b64[6] = { TV_B }, r64[6] = { 0 };
  uint64_t a48[NWORDS], b48[NWORDS], r48[NWORDS];
  __m512i a_8x1w[NWORDS], b_8x1w[NWORDS], r_8x1w[NWORDS];
  int i;


  add_fp_8x1w(r_8x1w, a_8x1w, b_8x1w);


}

// ----------------------------------------------------------------------------

int main()
{
  fp_test();

  return 0;
}