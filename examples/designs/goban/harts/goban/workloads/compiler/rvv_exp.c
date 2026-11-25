#include <kernel.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *);
static float input[2][36], output[2][36];
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 36; ++col)
      input[row][col] = (col - 32) * 0.5f + row * 0.125f;
  input[0][0] = -INFINITY;
  input[0][1] = -104;
  input[0][2] = -90;
  input[0][3] = -87;
  input[0][4] = INFINITY;
  input[0][5] = NAN;
  struct matrix a = {input, input, 0, {2, 36}, {36, 1}};
  struct matrix out = {output, output, 0, {2, 36}, {36, 1}};
  _mlir_ciface_compute(&a, &out);
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 36; ++col) {
      float expected = expf(input[row][col]);
      float actual = output[row][col];
      int ok = isnan(expected) ? isnan(actual)
               : isinf(expected)
                   ? actual == expected
                   : isfinite(actual) &&
                         fabsf(actual - expected) <= 2e-6f * expected + 1e-37f;
      if (!ok) {
        printf("FAIL: vector exp row %d column %d\n", row, col);
        return 1;
      }
    }
  puts("PASS: vector exp normal range, underflow, special values and tail");
  return 0;
}
