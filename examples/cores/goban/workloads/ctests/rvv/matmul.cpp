#include "images.h"
#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

enum { M = 3, N = 13, K = 7 };
alignas(16) static float a[M * K], b[K * N], output[40];
alignas(16) static float aTile[12], bTile[52];

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
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)output, 1, 10, 1);
  mvin_kernel(images::matmul.bytes, images::matmul.size, 0);
  for (unsigned begin = 0; begin < K;) {
    unsigned length = begin ? 4 : 3;
    std::memset(aTile, 0, sizeof(aTile));
    std::memset(bTile, 0, sizeof(bTile));
    for (unsigned r = 0; r < M; ++r)
      for (unsigned k = 0; k < length; ++k)
        aTile[r * length + k] = a[r * K + begin + k];
    for (unsigned k = 0; k < length; ++k)
      for (unsigned c = 0; c < N; ++c)
        bTile[k * N + c] = b[(begin + k) * N + c];
    alignas(16) kernel_launch call{images::matmul.entry,
                                   images::matmul.text_bytes,
                                   0x80002000,
                                   {1 << 16, 2 << 16, 3 << 16, M, N, length},
                                   0};
    bb_mvin((uintptr_t)aTile, 2, (M * length + 3) / 4, 1);
    bb_mvin((uintptr_t)bTile, 3, (length * N + 3) / 4, 1);
    bb_mvin((uintptr_t)&call, 0, sizeof(call) / 16, 1);
    bb_fence();
    run_kernel(0, 0);
    bb_fence();
    begin += length;
  }
  bb_mvout((uintptr_t)output, 1, 10, 1);
  bb_fence();
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
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mem_release(bank);
  std::puts("RVV MATMUL PASSED (separate multiply/add, tails, K chunks)");
  return 0;
}
