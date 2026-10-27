#include <CRunnerUtils.h>
#include <cmath>
#include <cstdio>
#include <cstring>

extern "C" void _mlir_ciface_rvv_norm(UnrankedMemRefType<float> *,
                                      UnrankedMemRefType<float> *,
                                      UnrankedMemRefType<float> *, uint32_t,
                                      uint32_t);
extern "C" void _mlir_ciface_rvv_norm_no_weight(UnrankedMemRefType<float> *,
                                                UnrankedMemRefType<float> *,
                                                uint32_t, uint32_t);
constexpr int capacity = THREAD_MEM_BANK_ENTRIES * THREAD_MEM_BANK_WIDTH / 32;
constexpr int maximum = 2 * capacity + 5;
alignas(16) static float input[2 * (maximum + 4) + 4];
alignas(16) static float output[2 * maximum + 4], expected[2 * maximum];
alignas(16) static float weights[maximum];

int main() {
  for (int width : {capacity - 1, capacity, capacity + 3, maximum}) {
    for (bool weighted : {false, true}) {
      for (float &value : output)
        value = -999.0f;
      for (int col = 0; col < width; ++col)
        weights[col] = 0.5f + (col % 7) * 0.125f;
      for (int row = 0; row < 2; ++row)
        for (int col = 0; col < width; ++col)
          input[2 + row * (width + 4) + col] =
              ((col + row * 3) % 31 - 15) * 0.125f;
      float multiplier = 1.0f / width, epsilon = 1.0e-5f;
      uint32_t meanBits, epsilonBits;
      std::memcpy(&meanBits, &multiplier, 4);
      std::memcpy(&epsilonBits, &epsilon, 4);
      asm volatile("csrw fcsr, zero" ::: "memory");
      for (int row = 0; row < 2; ++row) {
        volatile float sum = 0;
        for (int col = 0; col < width; ++col) {
          float value = input[2 + row * (width + 4) + col];
          volatile float square = value * value;
          sum = sum + square;
        }
        volatile float mean = sum * multiplier;
        float scale = 1.0f / std::sqrt(mean + epsilon);
        for (int col = 0; col < width; ++col) {
          volatile float value = input[2 + row * (width + 4) + col] * scale;
          expected[row * width + col] = weighted ? weights[col] * value : value;
        }
      }
      uint32_t expectedFlags;
      asm volatile("csrr %0, fflags" : "=r"(expectedFlags)::"memory");
      StridedMemRefType<float, 2> in{
          input, input, 2, {2, width}, {width + 4, 1}};
      StridedMemRefType<float, 2> out{
          output, output, 2, {2, width}, {width, 1}};
      StridedMemRefType<float, 1> weight{weights, weights, 0, {width}, {1}};
      UnrankedMemRefType<float> inRef{2, &in}, outRef{2, &out},
          wRef{1, &weight};
      asm volatile("csrw fcsr, zero" ::: "memory");
      if (weighted)
        _mlir_ciface_rvv_norm(&outRef, &inRef, &wRef, meanBits, epsilonBits);
      else
        _mlir_ciface_rvv_norm_no_weight(&outRef, &inRef, meanBits, epsilonBits);
      uint32_t flags;
      asm volatile("csrr %0, fflags" : "=r"(flags)::"memory");
      if (flags != expectedFlags ||
          std::memcmp(output + 2, expected, 2 * width * sizeof(float))) {
        std::printf(
            "RMSNorm mismatch width=%d weighted=%d flags=%u expected=%u\n",
            width, weighted, flags, expectedFlags);
        return 1;
      }
      if (output[0] != -999 || output[1] != -999 ||
          output[2 * width + 2] != -999 || output[2 * width + 3] != -999)
        return 1;
    }
  }
  std::puts(
      "RVV RMSNORM PASSED (multi-bank chunks, strided rows, tails, FP flags)");
  return 0;
}
