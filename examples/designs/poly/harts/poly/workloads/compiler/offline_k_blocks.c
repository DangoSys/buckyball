#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <testutils.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct blocks {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct matrix *, struct blocks *,
                                 struct matrix *);
static float a[3][3072], output[3][67];
static int8_t weights[3][68][1056];
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  struct matrix am = {a, a, 0, {3, 3072}, {3072, 1}};
  struct blocks wm = {weights, weights, 0, {3, 68, 1056}, {71808, 1056, 1}};
  struct matrix cm = {output, output, 0, {3, 67}, {67, 1}};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3072; ++k)
      a[i][k] = (i + k) % 5 - 2;
  for (int block = 0; block < 3; ++block)
    for (int j = 0; j < 67; ++j) {
      for (int k = 0; k < 1024; ++k)
        weights[block][j][k] = (j + block * 1024 + k) % 7 - 3;
      for (int k = 0; k < 32; ++k)
        weights[block][j][1024 + k] = 126 + (block * 32 + k) % 3;
    }
  uint64_t start = read_counter();
  _mlir_ciface_compute(&am, &wm, &cm);
  printf("Operator cycles: %lu\n", (unsigned long)(read_counter() - start));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 67; ++j) {
      float expected = 0;
      for (int k = 0; k < 3072; ++k) {
        int exponent = (k / 32) % 3;
        float scale = exponent == 0 ? 0.5f : exponent == 1 ? 1.0f : 2.0f;
        expected += a[i][k] * ((j + k) % 7 - 3) * scale;
      }
      if (output[i][j] != expected) {
        printf("Offline K block mismatch at %d,%d\n", i, j);
        return 1;
      }
    }
  return 0;
}
