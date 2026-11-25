// Loom's half-precision conversions, which the compiled models use for the
// operands and results of the mesh: FP32 to FP16 (rounded to nearest even)
// and back, values and their expected bits by hand.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint16_t narrow(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  uint64_t half;
  __asm__ volatile(".option push\n.option arch, +zfh\n"
                   "fcvt.h.s ft0, %1, rne\nfmv.x.h %0, ft0\n"
                   ".option pop"
                   : "=r"(half)
                   : "f"(value)
                   : "ft0");
  return (uint16_t)half;
}

static uint32_t widen(uint16_t bits) {
  uint64_t half = bits;
  float value;
  __asm__ volatile(".option push\n.option arch, +zfh\n"
                   "fmv.h.x ft0, %1\nfcvt.s.h %0, ft0\n"
                   ".option pop"
                   : "=f"(value)
                   : "r"(half)
                   : "ft0");
  uint32_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

// FP32 bits and the FP16 bits they round to.
static const uint32_t singles[] = {
    0x00000000,
    0x80000000,
    0x3f800000,
    0xc0000000, // 0, -0, 1, -2
    0x3f801000,
    0x3f803000,
    0x3f801001, // ties to even: down, up; above a tie
    0x477fe000,
    0x477ff000, // the largest FP16; half an ulp above: infinity
    0x33800000,
    0x33000000,
    0x33000001, // the smallest subnormal; half of it:
                // 0; above: it
    0x7f800000,
    0x7fc00000, // infinity, NaN
};
static const uint16_t halves[] = {
    0x0000, 0x8000, 0x3c00, 0xc000, 0x3c00, 0x3c02, 0x3c01,
    0x7bff, 0x7c00, 0x0001, 0x0000, 0x0001, 0x7c00, 0x7e00,
};
// FP16 bits and the FP32 bits they are.
static const uint16_t narrows[] = {0x3c00, 0xc000, 0x7bff, 0x0001,
                                   0x0400, 0x7c00, 0x3555};
static const uint32_t wides[] = {0x3f800000, 0xc0000000, 0x477fe000, 0x33800000,
                                 0x38800000, 0x7f800000, 0x3eaaa000};

int main(void) {
  unsigned errors = 0;
  for (unsigned i = 0; i < sizeof(halves) / sizeof(halves[0]); ++i) {
    uint16_t got = narrow(singles[i]);
    if (got != halves[i]) {
      printf("Half precision: %08x narrows to %04x, expected %04x\n",
             (unsigned)singles[i], got, halves[i]);
      ++errors;
    }
  }
  for (unsigned i = 0; i < sizeof(narrows) / sizeof(narrows[0]); ++i) {
    uint32_t got = widen(narrows[i]);
    if (got != wides[i]) {
      printf("Half precision: %04x widens to %08x, expected %08x\n", narrows[i],
             (unsigned)got, (unsigned)wides[i]);
      ++errors;
    }
  }
  printf("Half precision: 21 conversions, %u errors\n", errors);
  return errors != 0;
}
