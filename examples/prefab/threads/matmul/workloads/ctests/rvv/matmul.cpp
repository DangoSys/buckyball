#include <cstdio>
#include <cstring>
#include <rvv/launch.h>

alignas(16) static const uint8_t image[] = {
#embed "matmul.image"
};
enum { M = 3, N = 13, K = 7 };
alignas(16) static float a[24], b[92], output[40];

int main() {
  for (unsigned r = 0; r < M; ++r)
    for (unsigned k = 0; k < K; ++k)
      a[r * K + k] = k ? (int(r + k) - 3) * 0.25f : 0x1.000002p0f;
  for (unsigned k = 0; k < K; ++k)
    for (unsigned c = 0; c < N; ++c)
      b[k * N + c] =
          k ? (c ? (int(k + c) % 9 - 4) * 0.125f : 0.0f) : 0x1.fffffcp-1f;
  for (unsigned i = 0; i < M * N; ++i)
    output[i] = -1.0f;
  rvv_load(image, sizeof(image));
  bb_mvin_group((uintptr_t)a, RVV_READ_BANK, 0, 6, 1);
  bb_mvin_group((uintptr_t)b, RVV_READ_BANK, 1, 23, 1);
  bb_mvin_group((uintptr_t)output, RVV_WRITE_BANK, 1, 10, 1);
  kernel_launch call{rvv_image_word(image, 2),
                     rvv_image_word(image, 1),
                     RVV_STACK,
                     {UINT64_C(3) << 32, 0, UINT64_C(1) << 32, M, N, K},
                     0};
  rvv_run(&call, sizeof(call));
  bb_mvout_group((uintptr_t)output, RVV_WRITE_BANK, 1, 10, 1);
  rvv_release();
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
  std::puts(
      "RVV MATMUL PASSED (separate multiply/add, tails, initial accumulator)");
  return 0;
}
