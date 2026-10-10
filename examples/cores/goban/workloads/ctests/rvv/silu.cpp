#include <CRunnerUtils.h>
#include <cmath>
#include <cstdio>

extern "C" void _mlir_ciface_rvv_silu(UnrankedMemRefType<float> *,
                                      UnrankedMemRefType<float> *);

alignas(16) static float input[40], output[40];

int main() {
  for (unsigned i = 0; i < 40; ++i) {
    input[i] = (int(i) - 18) * 0.75f;
    output[i] = 123.0f;
  }
  StridedMemRefType<float, 1> in{input, input, 0, {37}, {1}};
  StridedMemRefType<float, 1> out{output, output, 0, {37}, {1}};
  UnrankedMemRefType<float> inputRef{1, &in}, outputRef{1, &out};
  _mlir_ciface_rvv_silu(&outputRef, &inputRef);
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
  for (unsigned i = 37; i < 40; ++i)
    if (output[i] != 123.0f)
      return 1;
  std::puts("RVV SILU PASSED (libm tolerance, vector tail)");
  return 0;
}
