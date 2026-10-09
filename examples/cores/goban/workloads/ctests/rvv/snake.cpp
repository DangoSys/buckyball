#include <CRunnerUtils.h>
#include <cmath>
#include <cstdio>

extern "C" void _mlir_ciface_rvv_snake(UnrankedMemRefType<float> *,
                                       UnrankedMemRefType<float> *,
                                       UnrankedMemRefType<float> *,
                                       UnrankedMemRefType<float> *);

alignas(16) static float input[40], output[40];

int main() {
  for (unsigned i = 0; i < 40; ++i) {
    input[i] = (int(i % 19) - 9) * 0.25f;
    output[i] = 123.0f;
  }
  float logAlpha[2]{-0.5f, 0.5f}, logBeta[2]{0.25f, -0.25f};
  StridedMemRefType<float, 2> in{input, input, 0, {2, 19}, {19, 1}};
  StridedMemRefType<float, 2> out{output, output, 0, {2, 19}, {19, 1}};
  StridedMemRefType<float, 1> alpha{logAlpha, logAlpha, 0, {2}, {1}};
  StridedMemRefType<float, 1> beta{logBeta, logBeta, 0, {2}, {1}};
  UnrankedMemRefType<float> inputRef{2, &in}, outputRef{2, &out},
      alphaRef{1, &alpha}, betaRef{1, &beta};
  _mlir_ciface_rvv_snake(&outputRef, &inputRef, &alphaRef, &betaRef);
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
  for (unsigned i = 38; i < 40; ++i)
    if (output[i] != 123.0f)
      return 1;
  std::puts(
      "RVV SNAKE PASSED (libm tolerance, channel coefficients, vector tail)");
  return 0;
}
