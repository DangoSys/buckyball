#include <kernel.h>
#include <stdint.h>
#include <stdio.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct layout {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" void _mlir_ciface_compute(matrix *, layout *, matrix *, matrix *,
                                     matrix *, matrix *);
struct inputs {
  alignas(64) float a[2][20];
  float as[2], mult[2], ws[20], expected[2][20];
  int8_t w[20][20];
  alignas(64) int8_t packed[2][16][2][16];
};
static constexpr inputs data = [] {
  inputs result{};
  for (int i = 0; i < 2; ++i) {
    result.as[i] = i == 0 ? 0.5f : 1.0f;
    result.mult[i] = i == 0 ? 2 : 1;
    for (int k = 0; k < 20; ++k)
      result.a[i][k] = ((i + k) % 7 - 3) * result.as[i];
  }
  for (int j = 0; j < 20; ++j) {
    result.ws[j] = j % 2 ? 0.5f : 1.0f;
    for (int k = 0; k < 20; ++k) {
      result.w[j][k] = (j * 3 + k) % 5 - 2;
      result.packed[j / 16][k % 16][k / 16][j % 16] = result.w[j][k];
    }
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 20; ++j)
      for (int k = 0; k < 20; ++k)
        result.expected[i][j] += result.a[i][k] * result.w[j][k] * result.ws[j];
  return result;
}();
alignas(64) static float output[2][20];
int main() {
  kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  matrix a = {(void *)data.a, (void *)data.a, 0, {2, 20}, {20, 1}};
  layout weights = {
      (void *)data.packed, (void *)data.packed, 0, {2, 16, 32}, {512, 32, 1}};
  matrix multiplier = {(void *)data.mult, (void *)data.mult, 0, {2, 1}, {1, 1}};
  matrix activationStep = {(void *)data.as, (void *)data.as, 0, {2, 1}, {1, 1}};
  matrix weightStep = {(void *)data.ws, (void *)data.ws, 0, {20, 1}, {1, 1}};
  matrix result = {output, output, 0, {2, 20}, {20, 1}};
  _mlir_ciface_compute(&a, &weights, &multiplier, &activationStep, &weightStep,
                       &result);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 20; ++j)
      if (output[i][j] != data.expected[i][j]) {
        printf("Padded DMA copy mismatch at %d,%d\n", i, j);
        return 1;
      }
  puts("Compiled padded DMA copies: zero tails, row strides and output trim "
       "passed");
  return 0;
}
