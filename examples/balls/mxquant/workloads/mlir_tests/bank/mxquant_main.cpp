#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <dma.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern "C" void _mlir_ciface_quant_block_32();
extern "C" void _mlir_ciface_quant_dynamic(int64_t count);
#include "../../reference.h"
static uint32_t input[32] __attribute__((aligned(64)));
static uint8_t actual[64] __attribute__((aligned(64)));
static uint8_t expected[64];
int main(void) {
  const uint32_t edges[] = {0,          0x80000000, 1,          0x80000001,
                            0x007fffff, 0x00800000, 0x3f880000, 0x3f980000,
                            0x43e00000, 0x7f7fffff, 0xff7fffff};
  const uint32_t subnormals[] = {0,      1,        0x1000,     0x2000,
                                 0x3000, 0x7fffff, 0x80000001, 0x807fffff};
  uint32_t random = 0x6d2b79f5;
  for (int test = 0; test < 68; ++test) {
    for (int i = 0; i < 32; ++i) {
      random ^= random << 13;
      random ^= random >> 17;
      random ^= random << 5;
      input[i] =
          test == 0   ? (i & 1 ? 0x80000000u : 0)
          : test == 1 ? edges[i % 11]
          : test == 2 ? (i ? (i & 1 ? 0x3f880000u : 0x3f980000u) : 0x43e00000u)
          : test == 3 ? subnormals[i % 8]
                      : ((random & 0x807fffffu) |
                         ((((test - 4) * 4 + (random >> 23 & 3)) % 255) << 23));
    }
    memset(actual, 0x5a, sizeof(actual));
    memset(expected, 0x5a, sizeof(expected));
    mxquant_reference32(input, expected);
    if (test == 0 &&
        (expected[0] != 0 || expected[1] != 128 || expected[32] != 127))
      return 1;
    if (test == 1 &&
        (expected[9] != 126 || expected[10] != 254 || expected[32] != 246))
      return 2;
    if (test == 2 &&
        (expected[0] != 126 || expected[1] != 56 || expected[2] != 58))
      return 2;
    uint32_t state = 0x1fu | (test % 5) << 5, after;
    asm volatile("csrw fcsr, %0" : : "r"(state));
    bb_dma_fence();
    bb_mem_alloc(3, 1, 1);
    bb_mem_alloc(4, 1, 1);
    bb_mvin((uintptr_t)input, 3, 8, 1);
    bb_mvin((uintptr_t)actual, 4, 4, 1);
    _mlir_ciface_quant_block_32();
    bb_mvout((uintptr_t)actual, 4, 4, 1);
    bb_fence();
    asm volatile("csrr %0, fcsr" : "=r"(after));
    if (after != state || memcmp(actual, expected, sizeof(actual)))
      return 3;
    bb_mem_release(3);
    bb_mem_release(4);
  }
  enum { BANK_BYTES = BANK_LINES * BANK_WIDTH / 8, MAX_COUNT = BANK_BYTES / 4 };
  static uint32_t capacity_input[MAX_COUNT] __attribute__((aligned(64)));
  static uint8_t capacity_actual[BANK_BYTES] __attribute__((aligned(64)));
  const int counts[] = {32, 480, 960, MAX_COUNT};
  for (int test = 0; test < 4; ++test) {
    int count = counts[test];
    for (int i = 0; i < MAX_COUNT; ++i)
      capacity_input[i] = (i & 1) ? 0xbf800000u : 0x3f800000u;
    memset(capacity_actual, 0x5a, sizeof(capacity_actual));
    bb_dma_fence();
    bb_mem_alloc(3, 1, 1);
    bb_mem_alloc(4, 1, 1);
    bb_mvin((uintptr_t)capacity_input, 3, BANK_LINES, 1);
    bb_mvin((uintptr_t)capacity_actual, 4, BANK_LINES, 1);
    _mlir_ciface_quant_dynamic(count);
    bb_mvout((uintptr_t)capacity_actual, 4, BANK_LINES, 1);
    bb_fence();
    for (int i = 0; i < BANK_BYTES; ++i) {
      uint8_t expected = i < count                ? ((i & 1) ? 0xf8 : 0x78)
                         : i < count + count / 32 ? 119
                                                  : 0x5a;
      if (capacity_actual[i] != expected)
        return 1;
    }
    bb_mem_release(3);
    bb_mem_release(4);
  }
  puts("mxquant bank MLIR finite/RNE/signed-zero/capacity/guard PASS");
  return 0;
}
