#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute_compact_lhs(struct tensor *, struct tensor *,
                                             struct tensor *);
static float output[2 * 5 * 65];
static int8_t a[2 * 5 * 528];
static int8_t packed[2 * 65 * 528];
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute_rows(struct matrix *, struct matrix *,
                                      struct matrix *, struct matrix *);
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  const int columns[] = {1, 17, 32, 65};
  for (int run = 0; run < 4; ++run) {
    int k = 512, m = 3, n = columns[run];
    struct tensor am = {a, a, 0, {2, m, 528}, {m * 528, 528, 1}};
    struct tensor wm = {packed, packed, 0, {2, n, 528}, {n * 528, 528, 1}};
    struct tensor cm = {output, output, 0, {2, m, n}, {m * n, n, 1}};
    for (int i = 0; i < 2 * m * 528; ++i)
      a[i] = i % 528 < 512 ? i % 7 - 3 : 127;
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < n; ++j)
        for (int t = 0; t < 528; ++t)
          packed[(b * n + j) * 528 + t] = t < k ? (b + t + j) % 5 - 2 : 127;
    _mlir_ciface_compute_compact_lhs(&am, &wm, &cm);
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          float expected = 0;
          for (int t = 0; t < k; ++t)
            expected += a[(b * m + i) * 528 + t] * ((b + t + j) % 5 - 2);
          if (output[(b * m + i) * n + j] != expected) {
            printf("Compact LHS mismatch: N %d batch %d row %d column %d\n", n,
                   b, i, j);
            return 1;
          }
        }
  }
  const int heights[] = {1, 2, 5};
  for (int run = 0; run < 3; ++run) {
    int m = heights[run], n = 17;
    for (int i = 0; i < m * 528; ++i)
      a[i] = i % 528 < 512 ? i % 7 - 3 : 127;
    struct matrix am = {a, a, 0, {m, 528}, {528, 1}};
    struct matrix wm = {packed, packed, 0, {n, 528}, {528, 1}};
    struct matrix cm = {output, output, 0, {m, n}, {n, 1}};
    _mlir_ciface_compute_rows(&am, &wm, &cm, &cm);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float expected = 0;
        for (int t = 0; t < 512; ++t)
          expected += a[i * 528 + t] * (int8_t)packed[j * 528 + t];
        if (output[i * n + j] != expected) {
          printf("Packed matrix rows mismatch: M %d row %d column %d\n", m, i,
                 j);
          return 1;
        }
      }
  }
  puts("Compiled compact_lhs passed");
  return 0;
}
