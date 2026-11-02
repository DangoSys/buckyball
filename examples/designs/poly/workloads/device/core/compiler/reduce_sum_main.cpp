#include <CRunnerUtils.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
using Matrix = StridedMemRefType<uint32_t, 2>;
extern "C" void _mlir_ciface_reduce_sum(Matrix *, Matrix *, Matrix *, Matrix *);
constexpr unsigned rows = THREAD_MEM_BANK_ENTRIES + 1;
alignas(64) static uint32_t inputs[3][rows * 4], output[rows * 4];
int main() {
  const uint32_t values[3][8] = {
      {0x60ad78ec, 0x3f800000, 0x7f800000, 0xff800000, 0, 0x80000000, 1,
       0x7f7fffff},
      {0xe0ad78ec, 0xc0200000, 0xff800000, 0x7f800000, 0x80000000, 0x80000000,
       1, 0xff7fffff},
      {0x40490fdb, 0x3f000000, 0x3f800000, 0xc0a00000, 0, 0x80000000,
       0x80000001, 0x3f800000}};
  for (unsigned group = 0; group < 3; ++group)
    for (unsigned i = 0; i < rows * 4; ++i)
      inputs[group][i] = values[group][i % 8];
  Matrix a{inputs[0], inputs[0], 0, {rows, 4}, {4, 1}};
  Matrix b{inputs[1], inputs[1], 0, {rows, 4}, {4, 1}};
  Matrix c{inputs[2], inputs[2], 0, {rows, 4}, {4, 1}};
  Matrix result{output, output, 0, {rows, 4}, {4, 1}};
  _mlir_ciface_reduce_sum(&a, &b, &c, &result);
  for (unsigned i = 0; i < rows * 4; ++i) {
    float expected, next;
    memcpy(&expected, &inputs[0][i], 4);
    for (unsigned group = 1; group < 3; ++group) {
      memcpy(&next, &inputs[group][i], 4);
      asm volatile("fadd.s %0, %1, %2, rne"
                   : "=f"(expected)
                   : "f"(expected), "f"(next));
    }
    uint32_t bits;
    memcpy(&bits, &expected, 4);
    if (output[i] != bits) {
      printf("Ordered sum mismatch at %u: %08x != %08x\n", i, output[i], bits);
      return 1;
    }
  }
  puts("Ordered FP32 sum with a bank-boundary tail PASS");
  return 0;
}
