#include "images.h"
#include <CRunnerUtils.h>
#include <algorithm>
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <params.h>

namespace {
constexpr unsigned outputBank = 1, inputBank = 2, upBank = 3;
constexpr uint32_t bankBytes = BANK_LINES * (BANK_WIDTH / 8);
static_assert(bankBytes >= 4096 && bankBytes <= 65536 && bankBytes % 16 == 0);
} // namespace

extern "C" void _mlir_ciface_rvv_matmul(UnrankedMemRefType<float> *output,
                                        UnrankedMemRefType<float> *lhs,
                                        UnrankedMemRefType<float> *rhs) {
  DynamicMemRefType<float> out(*output), a(*lhs), b(*rhs);
  if ((out.rank != 2 && out.rank != 3) || a.rank != out.rank ||
      b.rank != out.rank) {
    fputs("RVV matmul requires matching rank-two or rank-three tensors\n",
          stderr);
    abort();
  }
  const int64_t axis = out.rank - 2;
  const int64_t rows = a.sizes[axis], inner = a.sizes[axis + 1];
  const int64_t cols = b.sizes[axis + 1];
  const int64_t batches = axis ? out.sizes[0] : 1;
  if (rows <= 0 || inner <= 0 || cols <= 0 || batches <= 0 ||
      b.sizes[axis] != inner || out.sizes[axis] != rows ||
      out.sizes[axis + 1] != cols ||
      (axis && (a.sizes[0] != batches || b.sizes[0] != batches))) {
    fputs("RVV matmul shape mismatch\n", stderr);
    abort();
  }
  alignas(16) float aTile[bankBytes / sizeof(float)];
  alignas(16) float bTile[bankBytes / sizeof(float)];
  alignas(16) float cTile[16 * 16];
  for (unsigned bank = 0; bank <= upBank; ++bank)
    bb_mem_alloc(bank, 1, 1);
  mvin_kernel(images::matmul.bytes, images::matmul.size, 0);
  for (int64_t batch = 0; batch < batches; ++batch) {
    const float *aSource =
        a.data + a.offset + (axis ? batch * a.strides[0] : 0);
    const float *bSource =
        b.data + b.offset + (axis ? batch * b.strides[0] : 0);
    float *cSource =
        out.data + out.offset + (axis ? batch * out.strides[0] : 0);
    for (int64_t row = 0; row < rows; row += 16) {
      const uint32_t m = std::min<int64_t>(16, rows - row);
      for (int64_t col = 0; col < cols; col += 16) {
        const uint32_t n = std::min<int64_t>(16, cols - col);
        const uint32_t count = m * n;
        float *cDirect =
            cSource + row * out.strides[axis] + col * out.strides[axis + 1];
        const bool directC =
            out.strides[axis + 1] == 1 && (m == 1 || out.strides[axis] == n) &&
            count % 4 == 0 && (reinterpret_cast<uintptr_t>(cDirect) & 15) == 0;
        if (directC) {
          bb_mvin((uintptr_t)cDirect, outputBank, count / 4, 1);
        } else {
          for (uint32_t r = 0; r < m; ++r)
            for (uint32_t c = 0; c < n; ++c)
              cTile[r * n + c] = cSource[(row + r) * out.strides[axis] +
                                         (col + c) * out.strides[axis + 1]];
          std::fill(cTile + count, cTile + (count + 3) / 4 * 4, 0.0f);
          bb_mvin((uintptr_t)cTile, outputBank, (count + 3) / 4, 1);
        }
        const int64_t tileK = bankBytes / sizeof(float) / std::max(m, n);
        for (int64_t begin = 0; begin < inner; begin += tileK) {
          const uint32_t k = std::min(tileK, inner - begin);
          const float *aDirect =
              aSource + row * a.strides[axis] + begin * a.strides[axis + 1];
          const float *bDirect =
              bSource + begin * b.strides[axis] + col * b.strides[axis + 1];
          const bool directA = a.strides[axis + 1] == 1 &&
                               (m == 1 || a.strides[axis] == k) &&
                               m * k % 4 == 0 &&
                               (reinterpret_cast<uintptr_t>(aDirect) & 15) == 0;
          const bool directB = b.strides[axis + 1] == 1 &&
                               (k == 1 || b.strides[axis] == n) &&
                               k * n % 4 == 0 &&
                               (reinterpret_cast<uintptr_t>(bDirect) & 15) == 0;
          if (directA) {
            bb_mvin((uintptr_t)aDirect, inputBank, m * k / 4, 1);
          } else {
            for (uint32_t r = 0; r < m; ++r) {
              const float *source = aSource + (row + r) * a.strides[axis] +
                                    begin * a.strides[axis + 1];
              if (a.strides[axis + 1] == 1)
                std::memcpy(aTile + r * k, source, k * sizeof(float));
              else
                for (uint32_t i = 0; i < k; ++i)
                  aTile[r * k + i] = source[i * a.strides[axis + 1]];
            }
            std::fill(aTile + m * k, aTile + (m * k + 3) / 4 * 4, 0.0f);
            bb_mvin((uintptr_t)aTile, inputBank, (m * k + 3) / 4, 1);
          }
          if (directB) {
            bb_mvin((uintptr_t)bDirect, upBank, k * n / 4, 1);
          } else {
            for (uint32_t i = 0; i < k; ++i) {
              const float *source = bSource + (begin + i) * b.strides[axis] +
                                    col * b.strides[axis + 1];
              if (b.strides[axis + 1] == 1)
                std::memcpy(bTile + i * n, source, n * sizeof(float));
              else
                for (uint32_t c = 0; c < n; ++c)
                  bTile[i * n + c] = source[c * b.strides[axis + 1]];
            }
            std::fill(bTile + k * n, bTile + (k * n + 3) / 4 * 4, 0.0f);
            bb_mvin((uintptr_t)bTile, upBank, (k * n + 3) / 4, 1);
          }
          kernel_launch call{
              images::matmul.entry,
              images::matmul.text_bytes,
              0x80002000,
              {outputBank << 16, inputBank << 16, upBank << 16, m, n, k},
              0};
          bb_mvin((uintptr_t)&call, 0, sizeof(call) / 16, 1);
          bb_fence();
          run_kernel(0, 0);
          bb_fence();
        }
        if (directC) {
          bb_mvout((uintptr_t)cDirect, outputBank, count / 4, 1);
          bb_fence();
        } else {
          bb_mvout((uintptr_t)cTile, outputBank, (count + 3) / 4, 1);
          bb_fence();
          for (uint32_t r = 0; r < m; ++r)
            for (uint32_t c = 0; c < n; ++c)
              cSource[(row + r) * out.strides[axis] +
                      (col + c) * out.strides[axis + 1]] = cTile[r * n + c];
        }
      }
    }
  }
  for (unsigned bank = 0; bank <= upBank; ++bank)
    bb_mem_release(bank);
}
