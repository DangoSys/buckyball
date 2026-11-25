#include <cstdint>
#include <cstdio>
#include <kernel.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct vector {
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
struct cube {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" void _mlir_ciface_compute(matrix *, matrix *, vector *, matrix *);
extern "C" void _mlir_ciface_compute_pipeline(matrix *, matrix *, matrix *);
extern "C" void _mlir_ciface_compute_strided(cube *, cube *);
alignas(64) static float input[5 * 65], weights[65], output[5 * 65 + 16];
alignas(64) static const float scales[5] = {0.25, 1, 4, 16, 64};
static constexpr float inverse_scales[5] = {2, 1, 0.5, 0.25, 0.125};
alignas(64) static int8_t packed_input[5 + 5 * 33], packed_weights[3 + 32 * 33];

int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  const int widths[] = {3, 7, 32, 33, 65};
  const int heights[] = {1, 2, 5};
  for (int rows : heights)
    for (int width : widths) {
      for (int i = 0; i < rows * width; ++i)
        input[i] = (i % 17 - 8) * 0.25f;
      for (int i = 0; i < width; ++i)
        weights[i] = (i - 16) * 0.125f;
      for (float &x : output)
        x = 17;
      matrix a{input, input, 0, {rows, width}, {width, 1}};
      auto *scale = const_cast<float *>(scales);
      matrix s{scale, scale, 0, {rows, 1}, {1, 1}};
      vector w{weights, weights, 0, {width}, {1}};
      matrix out{output, output, 0, {rows, width}, {width, 1}};
      _mlir_ciface_compute(&a, &s, &w, &out);
      for (int i = 0; i < rows * width; ++i)
        if (output[i] !=
            input[i] * inverse_scales[i / width] + weights[i % width]) {
          printf("Dynamic GP mismatch at width %d index %d\n", width, i);
          return 1;
        }
      for (unsigned i = rows * width; i < sizeof(output) / sizeof(*output); ++i)
        if (output[i] != 17)
          return 1;
    }
  for (int rows : heights) {
    for (unsigned i = 0; i < 5 * 33; ++i)
      packed_input[5 + i] = i % 33 < 32 ? 1 : 122;
    for (unsigned i = 0; i < 32 * 33; ++i)
      packed_weights[3 + i] = i % 33 < 32 ? 1 : 127;
    for (float &x : output)
      x = 17;
    matrix a{packed_input, packed_input, 5, {rows, 33}, {33, 1}};
    matrix w{packed_weights, packed_weights, 3, {32, 33}, {33, 1}};
    matrix out{output, output, 3, {rows, 32}, {32, 1}};
    _mlir_ciface_compute_pipeline(&a, &w, &out);
    for (int i = 0; i < rows * 32; ++i)
      if (output[3 + i] != 27) {
        printf("Row pipeline mismatch at row count %d index %d\n", rows, i);
        return 2;
      }
    if (output[0] != 17 || output[3 + rows * 32] != 17)
      return 3;
  }
  for (int rows : heights) {
    for (int i = 0; i < 2 * rows * 32; ++i)
      input[i] = (i % 17 - 8) * .25f;
    for (float &x : output)
      x = 17;
    cube a{input, input, 0, {2, rows, 32}, {rows * 32, 32, 1}};
    cube out{output, output, 4, {2, rows, 32}, {rows * 32, 32, 1}};
    _mlir_ciface_compute_strided(&a, &out);
    for (int i = 0; i < 2 * rows * 32; ++i)
      if (output[4 + i] != 2 * input[i])
        return 4;
    if (output[0] != 17 || output[4 + 2 * rows * 32] != 17)
      return 5;
  }
  return 0;
}
