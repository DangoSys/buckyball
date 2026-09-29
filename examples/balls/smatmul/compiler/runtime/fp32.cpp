#include <CRunnerUtils.h>
#include <algorithm>
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <isa/smatmul.h>

extern "C" void _mlir_ciface_fp32_matmul(UnrankedMemRefType<float> *lhs,
                                         UnrankedMemRefType<float> *rhs,
                                         UnrankedMemRefType<float> *result) {
  DynamicMemRefType<float> a(*lhs), b(*rhs), c(*result);
  const int rank = a.rank;
  if ((rank != 2 && rank != 3) || b.rank != rank || c.rank != rank) {
    fputs("FP32 matmul requires matrices or matched batches\n", stderr);
    abort();
  }
  const int64_t batches = rank == 3 ? a.sizes[0] : 1;
  const int64_t m = a.sizes[rank - 2], k = a.sizes[rank - 1],
                n = b.sizes[rank - 1];
  if (batches <= 0 || m <= 0 || k <= 0 || n <= 0 || k != b.sizes[rank - 2] ||
      c.sizes[rank - 2] != m || c.sizes[rank - 1] != n ||
      (rank == 3 && (b.sizes[0] != batches || c.sizes[0] != batches))) {
    fputs("FP32 matmul shape mismatch\n", stderr);
    abort();
  }
  constexpr int block = BANK_LINES * (BANK_WIDTH / 8) / (16 * sizeof(float));
  static_assert(block > 0 && block % 4 == 0 && block < 4096 &&
                BANK_WIDTH == 128);
  alignas(64) float ap[16 * block], bp[16 * block], cp[256];
  for (int bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  for (int64_t batch = 0; batch < batches; ++batch) {
    const float *av =
        a.data + a.offset + (rank == 3 ? batch * a.strides[0] : 0);
    const float *bv =
        b.data + b.offset + (rank == 3 ? batch * b.strides[0] : 0);
    float *cv = c.data + c.offset + (rank == 3 ? batch * c.strides[0] : 0);
    for (int64_t row = 0; row < m; row += 16)
      for (int64_t column = 0; column < n; column += 16) {
        int rows = m - row == 1 ? 1 : 16;
        for (int64_t inner = 0; inner < k; inner += block) {
          int count = int((std::min<int64_t>(block, k - inner) + 3) / 4 * 4);
          for (int i = 0; i < rows; ++i)
            for (int j = 0; j < count; ++j)
              ap[i * count + j] = row + i < m && inner + j < k
                                      ? av[(row + i) * a.strides[rank - 2] +
                                           (inner + j) * a.strides[rank - 1]]
                                      : 0;
          for (int i = 0; i < 16; ++i)
            for (int j = 0; j < count; ++j)
              bp[i * count + j] = column + i < n && inner + j < k
                                      ? bv[(inner + j) * b.strides[rank - 2] +
                                           (column + i) * b.strides[rank - 1]]
                                      : 0;
          bb_mvin((uintptr_t)ap, 0, rows * count / 4, 1);
          bb_mvin((uintptr_t)bp, 1, 16 * count / 4, 1);
          bb_smatmul_f32(0, 1, 2, rows, 16, count, inner == 0,
                         inner + count >= k, 0);
        }
        bb_mvout((uintptr_t)cp, 2, rows * 4, 1);
        bb_fence();
        for (int i = 0; i < rows && row + i < m; ++i)
          for (int j = 0; j < 16 && column + j < n; ++j)
            cv[(row + i) * c.strides[rank - 2] +
               (column + j) * c.strides[rank - 1]] = cp[i * 16 + j];
      }
  }
  for (int bank = 0; bank < 3; ++bank)
    bb_mem_release(bank);
}
