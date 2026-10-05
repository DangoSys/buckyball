#include "images.h"
#include <bbhw/isa/isa.h>
#include <cmath>
#include <cstdio>

alignas(16) static float gate[40], up[40], output[40];

int main() {
  for (unsigned i = 0; i < 40; ++i) {
    gate[i] = (int(i) - 18) * 0.75f;
    up[i] = (int(i % 9) - 4) * 0.5f;
    output[i] = 123.0f;
  }
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  alignas(16) kernel_launch call{images::swiglu.entry,
                                 images::swiglu.text_bytes,
                                 0x80002000,
                                 {1 << 16, 2 << 16, 3 << 16, 37},
                                 0};
  bb_mvin((uintptr_t)gate, 2, 10, 1);
  bb_mvin((uintptr_t)up, 3, 10, 1);
  bb_mvin((uintptr_t)output, 1, 10, 1);
  bb_mvin((uintptr_t)&call, 0, sizeof(call) / 16, 1);
  mvin_kernel(images::swiglu.bytes, images::swiglu.size, 0);
  bb_fence();
  run_kernel(0, 0);
  bb_mvout((uintptr_t)output, 1, 10, 1);
  bb_fence();
  // Independent libm reference: approximate RVV exp is checked with tolerance.
  for (unsigned i = 0; i < 37; ++i) {
    float expected = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
    if (!std::isfinite(output[i]) ||
        std::fabs(output[i] - expected) > 2e-6f * (1 + std::fabs(expected))) {
      std::printf("SwiGLU mismatch at %u: %g expected %g\n", i, output[i],
                  expected);
      return 1;
    }
  }
  for (unsigned i = 37; i < 40; ++i)
    if (output[i] != 123.0f)
      return 1;
  std::puts("RVV SWIGLU PASSED (libm tolerance, vector tail)");
  return 0;
}
