#include <CRunnerUtils.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" void _mlir_ciface_rvv_matmul(UnrankedMemRefType<float> *,
                                        UnrankedMemRefType<float> *,
                                        UnrankedMemRefType<float> *);

enum { M = 3, N = 13, K = 7 };
alignas(16) static float a[M * K], b[K * N], output[40];

int main() {
  for (unsigned r = 0; r < M; ++r)
    for (unsigned k = 0; k < K; ++k)
      a[r * K + k] = k ? (int(r + k) - 3) * 0.25f : 0x1.000002p0f;
  for (unsigned k = 0; k < K; ++k)
    for (unsigned c = 0; c < N; ++c)
      b[k * N + c] =
          k ? (c ? (int(k + c) % 9 - 4) * 0.125f : 0.0f) : 0x1.fffffcp-1f;
  for (unsigned i = 0; i < 40; ++i)
    output[i] = i < M * N ? -1.0f : 123.0f;
  StridedMemRefType<float, 2> aView{a, a, 0, {M, K}, {K, 1}};
  StridedMemRefType<float, 2> bView{b, b, 0, {K, N}, {N, 1}};
  StridedMemRefType<float, 2> out{output, output, 0, {M, N}, {N, 1}};
  UnrankedMemRefType<float> aRef{2, &aView}, bRef{2, &bView},
      outputRef{2, &out};
  _mlir_ciface_rvv_matmul(&outputRef, &aRef, &bRef);
  for (unsigned r = 0; r < M; ++r)
    for (unsigned c = 0; c < N; ++c) {
      float expected = -1.0f;
      for (unsigned k = 0; k < K; ++k) {
        volatile float product = a[r * K + k] * b[k * N + c];
        expected += product;
      }
      uint32_t actualBits, expectedBits;
      std::memcpy(&actualBits, output + r * N + c, 4);
      std::memcpy(&expectedBits, &expected, 4);
      if (actualBits != expectedBits) {
        std::printf("RVV matmul mismatch r=%u c=%u actual=%08x expected=%08x\n",
                    r, c, actualBits, expectedBits);
        return 1;
      }
    }
  if (output[39] != 123.0f)
    return 1;
  std::puts(
      "RVV MATMUL PASSED (separate multiply/add, tails, initial accumulator)");
  return 0;
}
