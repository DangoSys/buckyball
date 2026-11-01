#include <cmath>
#include <cstdio>
#include <cstring>
#include <rvv/launch.h>

alignas(16) static const uint8_t image[] = {
#embed "snake.image"
};
alignas(16) static float input[40], output[40];
// The launch descriptor followed by log(alpha) and log(beta) of each channel.
alignas(16) static uint8_t descriptor[sizeof(kernel_launch) + 16];

int main() {
  for (unsigned i = 0; i < 40; ++i)
    input[i] = (int(i % 19) - 9) * 0.25f;
  float logAlpha[2]{-0.5f, 0.5f}, logBeta[2]{0.25f, -0.25f};
  rvv_load(image, sizeof(image));
  bb_mvin_group((uintptr_t)input, RVV_READ_BANK, 0, 10, 1);
  kernel_launch call{rvv_image_word(image, 2),
                     rvv_image_word(image, 1),
                     RVV_STACK,
                     {UINT64_C(3) << 32, 0, 2, 19,
                      RVV_DESCRIPTOR + sizeof(call),
                      RVV_DESCRIPTOR + sizeof(call) + 8},
                     0};
  std::memcpy(descriptor, &call, sizeof(call));
  std::memcpy(descriptor + sizeof(call), logAlpha, 8);
  std::memcpy(descriptor + sizeof(call) + 8, logBeta, 8);
  rvv_run(descriptor, sizeof(descriptor));
  bb_mvout_group((uintptr_t)output, RVV_WRITE_BANK, 1, 10, 1);
  rvv_release();
  // Independent libm exp/sin reference with tolerance for RVV approximations.
  for (unsigned channel = 0; channel < 2; ++channel) {
    float alpha = std::exp(logAlpha[channel]);
    float inverse = 1 / (std::exp(logBeta[channel]) + 1e-9f);
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
  std::puts("RVV SNAKE PASSED (libm tolerance, channel coefficients, vector "
            "tail)");
  return 0;
}
