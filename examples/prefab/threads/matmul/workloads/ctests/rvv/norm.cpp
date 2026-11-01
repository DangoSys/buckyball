#include <cmath>
#include <cstdio>
#include <cstring>
#include <rvv/launch.h>

alignas(16) static const uint8_t image[] = {
#embed "norm.image"
};
enum { WIDTH = 61 };
alignas(16) static float input[64], weights[64], output[64], expected[WIDTH];

int main() {
  for (int col = 0; col < WIDTH; ++col) {
    weights[col] = 0.5f + (col % 7) * 0.125f;
    input[col] = (col % 31 - 15) * 0.125f;
  }
  float multiplier = 1.0f / WIDTH, epsilon = 1.0e-5f;
  uint32_t meanBits, epsilonBits;
  std::memcpy(&meanBits, &multiplier, 4);
  std::memcpy(&epsilonBits, &epsilon, 4);
  asm volatile("csrw fcsr, zero" ::: "memory");
  volatile float sum = 0;
  for (int col = 0; col < WIDTH; ++col) {
    volatile float square = input[col] * input[col];
    sum = sum + square;
  }
  volatile float mean = sum * multiplier;
  float scale = 1.0f / std::sqrt(mean + epsilon);
  for (int col = 0; col < WIDTH; ++col) {
    volatile float value = input[col] * scale;
    expected[col] = weights[col] * value;
  }
  uint32_t expectedFlags;
  asm volatile("csrr %0, fflags" : "=r"(expectedFlags)::"memory");
  rvv_load(image, sizeof(image));
  bb_mvin_group((uintptr_t)input, RVV_READ_BANK, 0, 16, 1);
  bb_mvin_group((uintptr_t)weights, RVV_READ_BANK, 1, 16, 1);
  kernel_launch call{rvv_image_word(image, 2),
                     rvv_image_word(image, 1),
                     RVV_STACK,
                     {UINT64_C(3) << 32, 0, UINT64_C(1) << 32, WIDTH, meanBits,
                      epsilonBits, RVV_FLAGS, 0},
                     0};
  rvv_run(&call, sizeof(call));
  bb_mvout_group((uintptr_t)output, RVV_WRITE_BANK, 1, 16, 1);
  uint64_t flags = rvv_flags();
  rvv_release();
  if (flags != expectedFlags ||
      std::memcmp(output, expected, WIDTH * sizeof(float))) {
    std::printf("RMSNorm mismatch flags=%u expected=%u\n", unsigned(flags),
                expectedFlags);
    return 1;
  }
  std::puts("RVV RMSNORM PASSED (ordered reduction, weights, tail, FP flags)");
  return 0;
}
