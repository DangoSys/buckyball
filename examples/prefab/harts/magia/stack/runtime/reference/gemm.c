// magia_gemm in software: what the mesh computes, for runs without the mesh
// (the checks of the model recipes). RedMulE accumulates one product at a
// time in FP16, each step a fused multiply-add rounded to nearest even:
// z[m][k] = fma(x[m][n], w[n][k], z[m][k]) for n = 0 .. N-1.
#include <stdint.h>

void magia_gemm(uint64_t x, uint64_t w, uint64_t z, uint64_t m, uint64_t n,
                uint64_t k) {
  const _Float16 *xs = (const _Float16 *)x;
  const _Float16 *ws = (const _Float16 *)w;
  _Float16 *zs = (_Float16 *)z;
  // Every element of a row of Z takes the terms in the order above; the
  // columns are independent, so the row of W is the inner loop.
  for (uint64_t row = 0; row < m; ++row)
    for (uint64_t term = 0; term < n; ++term) {
      const _Float16 scale = xs[row * n + term];
      const _Float16 *restrict weights = ws + term * k;
      _Float16 *restrict sums = zs + row * k;
      for (uint64_t column = 0; column < k; ++column)
        sums[column] = __builtin_fmaf16(scale, weights[column], sums[column]);
    }
}
