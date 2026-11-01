#include <engine.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *,
                                 struct tensor *);
static float a[2 * 3 * 65], w[2 * 65 * 17], output[2 * 3 * 17];
int main(void) {
  struct engine engine;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct engine_slot slots[count];
  engine_init(&engine, slots, count);
  for (int run = 0; run < 2; ++run) {
    int m = run ? 2 : 3, k = run ? 65 : 19, n = run ? 17 : 5;
    struct tensor am = {a, a, 0, {2, m, k}, {m * k, k, 1}};
    struct tensor wm = {w, w, 0, {2, k, n}, {k * n, n, 1}};
    struct tensor cm = {output, output, 0, {2, m, n}, {m * n, n, 1}};
    for (int b = 0; b < 2; ++b) {
      for (int i = 0; i < m; ++i)
        for (int t = 0; t < k; ++t)
          a[(b * m + i) * k + t] = (b + i + t) % 7 - 3;
      for (int t = 0; t < k; ++t)
        for (int j = 0; j < n; ++j)
          w[(b * k + t) * n + j] = (b + t + j) % 5 - 2;
    }
    _mlir_ciface_compute(&am, &wm, &cm);
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          float expected = 0;
          for (int t = 0; t < k; ++t)
            expected += a[(b * m + i) * k + t] * w[(b * k + t) * n + j];
          if (output[(b * m + i) * n + j] != expected) {
            printf("Dynamic attention mismatch: run %d batch %d row %d column "
                   "%d\n",
                   run, b, i, j);
            return 1;
          }
        }
  }
  puts("Compiled attention: dynamic M/N/K and partial MX blocks passed");
  return 0;
}
