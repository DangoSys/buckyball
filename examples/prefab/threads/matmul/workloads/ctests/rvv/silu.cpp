#include <cmath>
#include <cstdio>
#include <rvv/launch.h>

alignas(16) static const uint8_t image[] = {
#embed "silu.image"
};
alignas(16) static float input[40], output[40];

int main() {
  for (unsigned i = 0; i < 40; ++i)
    input[i] = (int(i) - 18) * 0.75f;
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding)::"memory");
  rvv_load(image, sizeof(image));
  bb_mvin_group((uintptr_t)input, RVV_READ_BANK, 0, 10, 1);
  kernel_launch call{rvv_image_word(image, 2),
                     rvv_image_word(image, 1),
                     RVV_STACK,
                     {UINT64_C(3) << 32, 0, 37, RVV_FLAGS, rounding << 5},
                     0};
  rvv_run(&call, sizeof(call));
  bb_mvout_group((uintptr_t)output, RVV_WRITE_BANK, 1, 10, 1);
  rvv_release();
  // Independent libm reference: approximate RVV exp is checked with tolerance.
  for (unsigned i = 0; i < 37; ++i) {
    float expected = input[i] / (1.0f + std::exp(-input[i]));
    if (!std::isfinite(output[i]) ||
        std::fabs(output[i] - expected) > 2e-6f * (1 + std::fabs(expected))) {
      std::printf("SiLU mismatch at %u: %g expected %g\n", i, output[i],
                  expected);
      return 1;
    }
  }
  std::puts("RVV SILU PASSED (libm tolerance, vector tail)");
  return 0;
}
