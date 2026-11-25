#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/lut.h>
#include <limits.h>
#include <stdint.h>

static const int32_t table[1024] __attribute__((aligned(64))) = {
    [0] = 1 << 30, [1] = 1 << 29, [2] = 1 << 28, [3] = 1 << 27};
static const int32_t input[3][8] __attribute__((aligned(64))) = {
    {7, 6, 5, 4, 7, 6, 5, 4},
    {0, 0, 0, 0, 0, 0, 0, 0},
    {INT32_MIN, INT32_MAX, -1, 0, -512, 511, 7, 9}};
static uint8_t actual[6][32] __attribute__((aligned(64))) = {{0x5a}};
static const uint8_t expected[6][32] = {
    {32, 16, 8, 4, 32, 16, 8, 4}, {0},
    {2, 2, 2, 2, 2, 2, 2, 2},     {127, 127, 127, 127, 127, 127, 127, 127},
    {0, 127, 0, 0, 0, 0, 0, 0},   {0, 127, 0, 0, 0, 0, 0, 0}};
__attribute__((used, section(".bbsim.check"), aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {actual, expected, sizeof(expected)};

int main(void) {
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)table, 1, 256, 1);
  const uint32_t multipliers[6] = {120 << 16, 4 << 16,   12 << 16,
                                   1 << 29,   127 << 16, (1U << 30) - 1};
  const int sources[6] = {0, 1, 1, 1, 2, 2};
  for (int i = 0; i < 6; ++i) {
    bb_mvin((uintptr_t)input[sources[i]], 0, 2, 1);
    bb_lut_softmax_i8(0, 1, 2, 2, multipliers[i]);
    bb_mvout((uintptr_t)actual[i], 2, 2, 1);
  }
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}
