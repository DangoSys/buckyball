#include <bbhw/isa/isa.h>
#include <isa/f32add.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>
static const uint32_t inputs[3][8] __attribute__((aligned(64))) = {
    {0x3f800000, 0xc0200000, 0x7f800000, 0xff800000, 0, 0x80000000, 0x7f7fffff,
     1},
    {0x40000000, 0x40400000, 0xff800000, 0x7f800000, 0x80000000, 0, 0xff7fffff,
     1},
    {0xbf800000, 0x3f000000, 0x40a00000, 0xc0a00000, 0, 0x80000000, 0x3f800000,
     0x80000001}};
static const uint32_t poison[8] __attribute__((aligned(64))) = {
    0x7fc00001, 0x7fc00001, 0x7fc00001, 0x7fc00001,
    0x7fc00001, 0x7fc00001, 0x7fc00001, 0x7fc00001};
static uint32_t output[8] __attribute__((aligned(64)));
static uint32_t add(uint32_t a, uint32_t b) {
  union {
    uint32_t bits;
    float value;
  } lhs = {.bits = a}, rhs = {.bits = b}, result;
  __asm__ volatile("fadd.s %0, %1, %2, rne"
                   : "=f"(result.value)
                   : "f"(lhs.value), "f"(rhs.value));
  return result.bits;
}
int main(void) {
  unsigned source = BB_SHARED_BANK_BASE, accumulator = source + 1,
           target = source + 2;
  bb_mem_alloc(source, 1, 3);
  bb_mem_alloc(accumulator, 1, 1);
  bb_mem_alloc(target, 1, 1);
  bb_mvin((uintptr_t)poison, accumulator, 2, 1);
  for (unsigned group = 0; group < 3; ++group)
    bb_mvin_group((uintptr_t)inputs[group], source, group, 2, 1);
  for (unsigned group = 0; group < 3; ++group) {
    bb_f32add(source, accumulator, target, 2, group, group == 0);
    unsigned previous = accumulator;
    accumulator = target;
    target = previous;
  }
  bb_mvout((uintptr_t)output, accumulator, 2, 1);
  bb_fence();
  for (unsigned i = 0; i < 8; ++i) {
    uint32_t expected = 0;
    for (unsigned group = 0; group < 3; ++group)
      expected = add(expected, inputs[group][i]);
    if (output[i] != expected)
      __builtin_trap();
  }
  bb_mem_release(source);
  bb_mem_release(accumulator);
  bb_mem_release(target);
  puts("F32ADD grouped shared banks/RNE/canonical NaN PASS");
  return 0;
}
