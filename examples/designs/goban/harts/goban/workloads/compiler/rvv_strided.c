#include <kernel.h>
#include <stdint.h>
#include <stdio.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *);
static float input[4][32], output[4][32];
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 32; ++col) {
      input[row][col] = row * 32 + col;
      output[row][col] = -1;
    }
  struct matrix a = {input, input, 0, {4, 32}, {32, 1}};
  struct matrix out = {output, output, 0, {4, 32}, {32, 1}};
  _mlir_ciface_compute(&a, &out);
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 32; ++col)
      if (output[row][col] != (col < 16 ? -1 : input[row][col] + 1)) {
        printf("FAIL: strided thread transfer row %d column %d\n", row, col);
        return 1;
      }
  puts("PASS: strided thread input and output preserve row gaps");
  return 0;
}
