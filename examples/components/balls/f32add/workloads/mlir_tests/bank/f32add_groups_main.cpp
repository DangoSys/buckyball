#include <CRunnerUtils.h>
#include <cstdint>
#include <cstdio>

using Rows = StridedMemRefType<uint32_t, 2>;
extern "C" void _mlir_ciface_reduce_groups(StridedMemRefType<uint32_t, 3> *,
                                           Rows *, Rows *, Rows *);

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
  // A whole-bank mvin reads row r of group g at (r * 2 + g) * 16 bytes.
  alignas(64) static uint32_t pair[2][2][4];
  for (unsigned group = 0; group < 2; ++group)
    for (unsigned i = 0; i < 8; ++i)
      pair[i / 4][group][i % 4] = input[group][i];
  StridedMemRefType<uint32_t, 3> pairRef = {
      &pair[0][0][0], &pair[0][0][0], 0, {2, 2, 4}, {8, 4, 1}};
  Rows lastRef = {
      (uint32_t *)input[2], (uint32_t *)input[2], 0, {2, 4}, {4, 1}};
  Rows poisonRef = {(uint32_t *)poison, (uint32_t *)poison, 0, {2, 4}, {4, 1}};
  Rows outputRef = {output, output, 0, {2, 4}, {4, 1}};
  _mlir_ciface_reduce_groups(&pairRef, &lastRef, &poisonRef, &outputRef);
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
  return 0;
}

int main() {
  int result = checkF32Add();
  if (!result)
    puts("F32ADD Bank MLIR grouped/RNE/NaN/first/SSA output PASS");
  return result;
}
