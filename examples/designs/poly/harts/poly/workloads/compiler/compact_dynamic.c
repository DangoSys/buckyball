#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute_compact_dynamic(struct tensor *,
                                                 struct tensor *,
                                                 struct tensor *);
static float a[2 * 3 * 512], output[2 * 3 * 65];
static int8_t packed[2 * 65 * 528];
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  const int lengths[] = {19, 32, 64, 65};
  for (int run = 0; run < 4; ++run) {
    int k = lengths[run], m = 3, n = run % 2 ? 17 : 32;
    int padded = (k + 31) / 32 * 32, width = padded + padded / 32;
    struct tensor am = {a, a, 0, {2, m, k}, {m * k, k, 1}};
    struct tensor wm = {packed, packed, 0, {2, width, n}, {width * n, n, 1}};
    struct tensor cm = {output, output, 0, {2, m, n}, {m * n, n, 1}};
    for (int i = 0; i < 2 * m * k; ++i)
      a[i] = i % 7 - 3;
    for (int b = 0; b < 2; ++b)
      for (int t = 0; t < width; ++t)
        for (int j = 0; j < n; ++j)
          packed[(b * width + t) * n + j] = t < k        ? (b + t + j) % 5 - 2
                                            : t < padded ? 0
                                                         : 127;
    _mlir_ciface_compute_compact_dynamic(&am, &wm, &cm);
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          float expected = 0;
          for (int t = 0; t < k; ++t)
            expected += a[(b * m + i) * k + t] * ((b + t + j) % 5 - 2);
          if (output[(b * m + i) * n + j] != expected) {
            printf("Dynamic compact mismatch: K %d batch %d row %d column %d\n",
                   k, b, i, j);
            return 1;
          }
        }
  }
  puts("Compiled compact_dynamic passed");
  return 0;
}
