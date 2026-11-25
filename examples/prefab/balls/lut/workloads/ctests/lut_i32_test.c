#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/lut.h>
#include <limits.h>
#include <stdint.h>

static int32_t table[1024] __attribute__((aligned(64)));
static const int32_t input[8] __attribute__((aligned(64))) = {
    INT32_MIN, -512, -511, -1, 0, 1, 511, INT32_MAX};
static int32_t actual[8] __attribute__((aligned(64))) = {0x5a};
static const int32_t expected[8] = {1436, 1436, 1439, 2969,
                                    -100, -97,  1433, 1433};
__attribute__((used, section(".bbsim.check"), aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {actual, expected, sizeof(expected)};

int main(void) {
  for (int i = 0; i < 1024; ++i)
    table[i] = i * 3 - 100;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 0, 2, 1);
  bb_mvin((uintptr_t)table, 1, 256, 1);
  bb_lut_i32(0, 1, 2, 2);
  bb_mvout((uintptr_t)actual, 2, 2, 1);
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}
