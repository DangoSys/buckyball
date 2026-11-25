#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *,
                                 struct tensor *);
extern void _mlir_ciface_compute_transposed_rows(struct tensor *,
                                                 struct tensor *,
                                                 struct tensor *);
extern void _mlir_ciface_compute_transposed_tail_rows(struct tensor *,
                                                      struct tensor *,
                                                      struct tensor *);
extern void _mlir_ciface_compute_residual(struct tensor *, struct tensor *,
                                          struct tensor *);
extern void _mlir_ciface_compute_strided(struct tensor *, struct tensor *,
                                         struct tensor *);
extern void _mlir_ciface_compute_column_tiles(struct tensor *, struct tensor *,
                                              struct tensor *);
extern void _mlir_ciface_compute_compact(struct tensor *, struct tensor *,
                                         struct tensor *);
static int8_t packed[2 * 65 * 528];
static float a[2 * 17 * 512], w[2 * 65 * 32], output[2 * 17 * 32];
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  const int lengths[] = {19, 32, 64, 65};
  const int columns[] = {1, 17, 32, 65};
  for (int run = 0; run < 18; ++run) {
    int compact = run >= 14;
    int strided = run == 9;
    int transposed = run >= 4 && run < 7;
    int m = transposed ? (run == 4 ? 1 : 17) : run % 2 ? 2 : 3;
    int k = compact      ? 64
            : run >= 10  ? lengths[run - 10]
            : run >= 7   ? (run == 7 ? 32 : 64)
            : transposed ? 64
                         : lengths[run % 4];
    int n = compact      ? columns[run - 14]
            : run >= 10  ? 32
            : transposed ? (run == 6 ? 17 : 32)
            : run % 2    ? 17
                         : 5;
    int stride = strided ? 128 : k, offset = strided ? 32 : 0;
    struct tensor am = {a, a, 0, {2, m, stride}, {m * stride, stride, 1}};
    struct tensor wm = {w, w, 0, {2, k, n}, {k * n, n, 1}};
    if (transposed)
      wm = (struct tensor){w, w, 0, {2, n, k}, {n * k, k, 1}};
    if (compact)
      wm = (struct tensor){packed, packed, 0, {2, n, 66}, {n * 66, 66, 1}};
    struct tensor cm = {output, output, 0, {2, m, n}, {m * n, n, 1}};
    if (strided)
      for (int i = 0; i < 2 * m * stride; ++i)
        a[i] = 99.0f;
    for (int b = 0; b < 2; ++b) {
      for (int i = 0; i < m; ++i)
        for (int t = 0; t < k; ++t)
          a[(b * m + i) * stride + offset + t] =
              run >= 7 && run < 9
                  ? ((b + i + t) % 7 - 3) * 0.25f + 0.0009765625f
                  : (b + i + t) % 7 - 3;
      for (int t = 0; t < k; ++t)
        for (int j = 0; j < n; ++j)
          if (compact)
            packed[(b * n + j) * 66 + t] = (b + t + j) % 5 - 2;
          else
            w[b * k * n + (transposed ? j * k + t : t * n + j)] =
                (b + t + j) % 5 - 2;
      if (compact)
        for (int j = 0; j < n; ++j)
          packed[(b * n + j) * 66 + 64] = packed[(b * n + j) * 66 + 65] = 127;
    }
    if (compact)
      _mlir_ciface_compute_compact(&am, &wm, &cm);
    else if (run >= 10)
      _mlir_ciface_compute_column_tiles(&am, &wm, &cm);
    else if (strided)
      _mlir_ciface_compute_strided(&am, &wm, &cm);
    else if (run >= 7)
      _mlir_ciface_compute_residual(&am, &wm, &cm);
    else if (run < 4)
      _mlir_ciface_compute(&am, &wm, &cm);
    else if (run != 6)
      _mlir_ciface_compute_transposed_rows(&am, &wm, &cm);
    else
      _mlir_ciface_compute_transposed_tail_rows(&am, &wm, &cm);
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          float expected = 0;
          for (int t = 0; t < k; ++t)
            expected +=
                a[(b * m + i) * stride + offset + t] *
                (compact ? (b + t + j) % 5 - 2
                         : w[b * k * n + (transposed ? j * k + t : t * n + j)]);
          if (output[(b * m + i) * n + j] != expected) {
            printf("Dynamic attention mismatch: run %d batch %d row %d column "
                   "%d\n",
                   run, b, i, j);
            return 1;
          }
        }
  }
  puts("Compiled attention: dynamic M/N/K, quantized layouts and residual "
       "products passed");
  return 0;
}
