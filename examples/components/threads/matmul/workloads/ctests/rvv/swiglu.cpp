#include <CRunnerUtils.h>
#include <cmath>
#include <cstdio>

extern "C" void _mlir_ciface_rvv_swiglu(UnrankedMemRefType<float> *,
                                        UnrankedMemRefType<float> *,
                                        UnrankedMemRefType<float> *);

alignas(16) static float gate[40], up[40], output[40];

int main() {
  for (unsigned i = 0; i < 40; ++i) {
    gate[i] = (int(i) - 18) * 0.75f;
    up[i] = (int(i % 9) - 4) * 0.5f;
    output[i] = 123.0f;
  }
  StridedMemRefType<float, 1> gateView{gate, gate, 0, {37}, {1}};
  StridedMemRefType<float, 1> upView{up, up, 0, {37}, {1}};
  StridedMemRefType<float, 1> out{output, output, 0, {37}, {1}};
  UnrankedMemRefType<float> gateRef{1, &gateView}, upRef{1, &upView},
      outputRef{1, &out};
  _mlir_ciface_rvv_swiglu(&outputRef, &gateRef, &upRef);
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
