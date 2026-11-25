#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/im2col.h>
#include <stdint.h>

// 21x21 input, 3x3 stride 2, 100 windows padded to 104, three K groups.
static const int8_t input[441 * 16] __attribute__((aligned(64))) = {
    [0] = 11, [(10 * 21 + 10) * 16 + 7] = -23, [440 * 16 + 15] = 37};
static const int8_t expected[3 * 312 * 16]
    __attribute__((aligned(64))) = {[0] = 11,
                                    [(624 + 140) * 16 + 7] = -23,
                                    [(624 + 125) * 16 + 7] = -23,
                                    [166 * 16 + 7] = -23,
                                    [151 * 16 + 7] = -23,
                                    [(624 + 307) * 16 + 15] = 37};
static int8_t actual[3 * 312 * 16]
    __attribute__((aligned(64))) = {[0 ... 3 * 312 * 16 - 1] = 0x5a};
__attribute__((used, section(".bbsim.check"), aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {actual, expected, sizeof(actual)};

int main(void) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 3);
  bb_mvin((uintptr_t)input, 0, 441, 1);
  bb_im2col(0, 1, 21, 3, 2, 0, 0, 0, 0, 0, 100, 8);
  for (int group = 0; group < 3; ++group)
    bb_mvout_group((uintptr_t)(actual + group * 312 * 16), 1, group, 312, 1);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  return 0;
}
