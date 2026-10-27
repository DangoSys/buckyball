#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>

extern "C" int64_t _mlir_ciface_reduce_groups();
static_assert(THREAD_FRONTEND_SHARED_VBANK_BASE == 32);

static int checkF32Add() {
  alignas(64) static const uint32_t input[3][8] = {
      {0x3f800000, 0xc0200000, 0x7f800000, 0xff800000, 0, 0x80000000,
       0x7f7fffff, 1},
      {0x40000000, 0x40400000, 0xff800000, 0x7f800000, 0x80000000, 0,
       0xff7fffff, 1},
      {0xbf800000, 0x3f000000, 0x40a00000, 0xc0a00000, 0, 0x80000000,
       0x3f800000, 0x80000001}};
  alignas(64) static uint32_t output[8];
  bb_mem_alloc(32, 1, 1);
  bb_mem_alloc(33, 1, 1);
  bb_mem_alloc(34, 1, 1);
  bb_mem_alloc(35, 1, 1);
  bb_mvin((uintptr_t)input[0], 32, 2, 1);
  bb_mvin((uintptr_t)input[1], 33, 2, 1);
  bb_mvin((uintptr_t)input[2], 35, 2, 1);
  if (_mlir_ciface_reduce_groups() != 33)
    return 1;
  bb_mvout((uintptr_t)output, 33, 2, 1);
  for (unsigned i = 0; i < 8; ++i) {
    union {
      uint32_t bits;
      float value;
    } expected = {input[0][i]}, source;
    for (unsigned group = 1; group < 3; ++group) {
      source.bits = input[group][i];
      asm volatile("fadd.s %0, %1, %2, rne"
                   : "=f"(expected.value)
                   : "f"(expected.value), "f"(source.value));
    }
    if (output[i] != expected.bits)
      return 2;
  }
  bb_mem_release(32);
  bb_mem_release(33);
  bb_mem_release(34);
  bb_mem_release(35);
  return 0;
}

int main() {
  int result = checkF32Add();
  if (!result)
    puts("F32ADD Ball MLIR RNE/NaN/ordered sum PASS");
  return result;
}
