#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>
#include <params.h>

extern "C" int64_t _mlir_ciface_reduce_groups();
static_assert(BB_SHARED_BANK_BASE == 32);

static int checkF32Add() {
  alignas(64) static const uint32_t input[3][8] = {
      {0x3f800000, 0xc0200000, 0x7f800000, 0xff800000, 0, 0x80000000,
       0x7f7fffff, 1},
      {0x40000000, 0x40400000, 0xff800000, 0x7f800000, 0x80000000, 0,
       0xff7fffff, 1},
      {0xbf800000, 0x3f000000, 0x40a00000, 0xc0a00000, 0, 0x80000000,
       0x3f800000, 0x80000001}};
  alignas(64) static const uint32_t poison[8] = {
      0x7fa00001, 0x7fa00001, 0x7fa00001, 0x7fa00001,
      0x7fa00001, 0x7fa00001, 0x7fa00001, 0x7fa00001};
  alignas(64) static uint32_t output[8];
  bb_mem_alloc(32, 1, 3);
  bb_mem_alloc(33, 1, 1);
  bb_mem_alloc(34, 1, 1);
  bb_mvin((uintptr_t)poison, 33, 2, 1);
  for (unsigned group = 0; group < 3; ++group)
    bb_mvin_group((uintptr_t)input[group], 32, group, 2, 1);
  if (_mlir_ciface_reduce_groups() != 34)
    return 1;
  bb_mvout((uintptr_t)output, 34, 2, 1);
  for (unsigned i = 0; i < 8; ++i) {
    union {
      uint32_t bits;
      float value;
    } expected = {0}, source;
    for (unsigned group = 0; group < 3; ++group) {
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
  return 0;
}

int main() {
  int result = checkF32Add();
  if (!result)
    puts("F32ADD Bank MLIR grouped/RNE/NaN/first/SSA output PASS");
  return result;
}
