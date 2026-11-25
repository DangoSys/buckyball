#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute_head_view(struct tensor *, struct tensor *,
                                           struct tensor *);
static float a[2 * 3 * 65], w[65 * 64], output[2 * 3 * 32];
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  const int lengths[] = {19, 32, 64, 65};
  for (int run = 0; run < 4; ++run) {
    int k = lengths[run], m = 3, n = 32;
    struct tensor am = {a, a, 0, {2, m, k}, {m * k, k, 1}};
    struct tensor wm = {w, w, 0, {k, 1, 64}, {64, 64, 1}};
    struct tensor cm = {output, output, 0, {2, m, n}, {m * n, n, 1}};
    for (int i = 0; i < 2 * m * k; ++i)
      a[i] = i % 7 - 3;
    for (int i = 0; i < k * 64; ++i)
      w[i] = i % 5 - 2;
    _mlir_ciface_compute_head_view(&am, &wm, &cm);
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          float expected = 0;
          for (int t = 0; t < k; ++t)
            expected += a[(b * m + i) * k + t] * w[(t * 2 + b) * n + j];
          if (output[(b * m + i) * n + j] != expected) {
            printf("Head view mismatch: K %d batch %d row %d column %d\n", k, b,
                   i, j);
            return 1;
          }
        }
  }
  puts("Compiled head_view passed");
  return 0;
}
