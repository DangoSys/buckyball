// layernorm and rmsnorm on the action thread alone: 2 FP32 rows of 16 from
// bank row 1 to another bank, the parameters in a third. Row 0 is +-2
// (mean 0, variance 4), row 1 is 3 +- 8 (mean 3, variance 64), with eps 0:
// the normalized values are +-1 exactly, and the outputs are
// +-gamma + beta (layernorm) and +-gamma for row 0 (rmsnorm), integers.
#include <bbhw/isa/isa.h>
#include <isa/layernorm.h>
#include <stdint.h>

enum { R = 2, W = 16, ROWS = LAYERNORM_ROWS(W) };
// Parameters: gamma, beta, eps (layernorm) from row 0; gamma, eps (rmsnorm)
// from row 2 * ROWS + 1.
enum { RMS = 2 * ROWS + 1, PARAMS = RMS + ROWS + 1 };
#define X ((volatile uint32_t *)HOST_TSS_BASE)
#define P ((volatile uint32_t *)(HOST_TSS_BASE + (1 + R * ROWS) * 16))
#define Y ((volatile uint32_t *)(HOST_TSS_BASE + (1 + R * ROWS + PARAMS) * 16))
#define Z                                                                      \
  ((volatile uint32_t *)(HOST_TSS_BASE +                                       \
                         (1 + R * ROWS + PARAMS + R * ROWS) * 16))

static int sign(int i) { return i % 2 ? -1 : 1; }
static int gamma(int i) { return i + 1; }
static int beta(int i) { return 7 - i; }

// The FP32 bits of an integer, |value| < 2^24.
static uint32_t fp32(int32_t value) {
  if (!value)
    return 0;
  uint32_t s = value < 0 ? 0x80000000u : 0;
  uint32_t magnitude = value < 0 ? -value : value;
  int top = 31;
  while (!(magnitude >> top))
    --top;
  return s | (uint32_t)(127 + top) << 23 |
         ((magnitude << (23 - top)) & 0x7fffff);
}

uint64_t entry(void) {
  for (int i = 0; i < W; ++i) {
    X[4 + i] = fp32(2 * sign(i));
    X[4 + ROWS * 4 + i] = fp32(3 + 8 * sign(i));
    P[i] = fp32(gamma(i));
    P[ROWS * 4 + i] = fp32(beta(i));
    P[RMS * 4 + i] = fp32(gamma(i));
  }
  P[2 * ROWS * 4] = 0;     // eps of layernorm
  P[(RMS + ROWS) * 4] = 0; // eps of rmsnorm
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mem_alloc(4, 1, 1);
  bb_mvin((uintptr_t)X, 1, 1 + R * ROWS, 1);
  bb_mvin((uintptr_t)P, 2, PARAMS, 1);
  bb_layernorm(1, 2, 3, R, W, 1, 0, 0, 0);
  bb_layernorm(1, 2, 4, 1, W, 1, 0, RMS, 1);
  bb_mvout((uintptr_t)Y, 3, R * ROWS, 1);
  bb_mvout((uintptr_t)Z, 4, ROWS, 1);
  asm volatile("fence" ::: "memory");
  for (int i = 0; i < W; ++i) {
    if (Y[i] != fp32(sign(i) * gamma(i) + beta(i)))
      return 1 + i;
    if (Y[ROWS * 4 + i] != fp32(sign(i) * gamma(i) + beta(i)))
      return 100 + i;
    if (Z[i] != fp32(sign(i) * gamma(i)))
      return 200 + i;
  }
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  bb_mem_release(4);
  return 0;
}
