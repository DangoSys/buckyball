#include "images.h"
#include <bbhw/isa/isa.h>
#include <cmath>
#include <cstdio>

alignas(16) static float input[40], output[40];

int main() {
  for (unsigned i = 0; i < 40; ++i) {
    input[i] = (int(i % 19) - 9) * 0.25f;
    output[i] = 123.0f;
  }
  for (unsigned bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  alignas(16) struct {
    kernel_launch call;
    float log_alpha[2];
    float log_beta[2];
  } descriptor{{images::snake.entry,
                images::snake.text_bytes,
                0x80002000,
                {1 << 16, 2 << 16, 2, 19, 48, 56},
                0},
               {-0.5f, 0.5f},
               {0.25f, -0.25f}};
  bb_mvin((uintptr_t)input, 2, 10, 1);
  bb_mvin((uintptr_t)output, 1, 10, 1);
  bb_mvin((uintptr_t)&descriptor, 0, sizeof(descriptor) / 16, 1);
  mvin_kernel(images::snake.bytes, images::snake.size, 1);
  bb_fence();
  run_kernel(0, 1);
  bb_mvout((uintptr_t)output, 1, 10, 1);
  bb_fence();
  // Independent libm exp/sin reference with tolerance for RVV approximations.
  for (unsigned channel = 0; channel < 2; ++channel) {
    float alpha = std::exp(descriptor.log_alpha[channel]);
    float inverse = 1 / (std::exp(descriptor.log_beta[channel]) + 1e-9f);
    for (unsigned i = 0; i < 19; ++i) {
      unsigned index = channel * 19 + i;
      float sine = std::sin(input[index] * alpha);
      float expected = input[index] + sine * sine * inverse;
      if (!std::isfinite(output[index]) ||
          std::fabs(output[index] - expected) >
              2e-6f * (1 + std::fabs(expected))) {
        std::printf("Snake mismatch at %u: %g expected %g\n", index,
                    output[index], expected);
        return 1;
      }
    }
  }
  for (unsigned i = 38; i < 40; ++i)
    if (output[i] != 123.0f)
      return 1;
  std::puts(
      "RVV SNAKE PASSED (libm tolerance, channel coefficients, vector tail)");
  return 0;
}
