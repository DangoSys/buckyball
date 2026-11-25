#include <cstdint>
#include <cstdio>
#include <cstring>
#include <kernel.h>

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
struct Data {
  float a[68][16], multipliers[68], steps[68], weightSteps[16],
      expected[68][16];
  int8_t weights[1][16][16];
};
constexpr Data data = [] {
  Data d{};
  for (int column = 0; column < 16; ++column) {
    d.weightSteps[column] = column & 1 ? 0.5f : 1.0f;
    for (int k = 0; k < 16; ++k)
      d.weights[0][k][column] = (column * 3 + k) % 5 - 2;
  }
  for (int row = 0; row < 68; ++row) {
    d.steps[row] = 0.25f * (1 << (row % 3));
    d.multipliers[row] = 1.0f / d.steps[row];
    for (int k = 0; k < 16; ++k)
      d.a[row][k] = ((row + k) % 7 - 3) * d.steps[row];
    for (int column = 0; column < 16; ++column) {
      int sum = 0;
      for (int k = 0; k < 16; ++k)
        sum += ((row + k) % 7 - 3) * d.weights[0][k][column];
      d.expected[row][column] =
          float(sum) * (d.steps[row] * d.weightSteps[column]);
    }
  }
  return d;
}();
static float output[68][16];

int main() {
  kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  matrix a{const_cast<float *>(&data.a[0][0]),
           const_cast<float *>(&data.a[0][0]),
           0,
           {68, 16},
           {16, 1}};
  layout w{const_cast<int8_t *>(&data.weights[0][0][0]),
           const_cast<int8_t *>(&data.weights[0][0][0]),
           0,
           {1, 16, 16},
           {256, 16, 1}};
  matrix mult{const_cast<float *>(data.multipliers),
              const_cast<float *>(data.multipliers),
              0,
              {68, 1},
              {1, 1}};
  matrix as{const_cast<float *>(data.steps),
            const_cast<float *>(data.steps),
            0,
            {68, 1},
            {1, 1}};
  matrix ws{const_cast<float *>(data.weightSteps),
            const_cast<float *>(data.weightSteps),
            0,
            {16, 1},
            {1, 1}};
  matrix out{output, output, 0, {68, 16}, {16, 1}};
  _mlir_ciface_compute(&a, &w, &mult, &as, &ws, &out);
  if (std::memcmp(output, data.expected, sizeof(output))) {
    std::puts("Compiled row-partitioned matmul: output bits differ");
    return 1;
  }
  std::puts("Compiled row-partitioned matmul: private quantization and output "
            "tail passed");
  return 0;
}
