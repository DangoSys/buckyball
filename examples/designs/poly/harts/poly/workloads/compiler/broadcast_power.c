#include <kernel.h>
#include <stdint.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct vector {
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
void _mlir_ciface_compute(struct matrix *, struct matrix *, struct vector *,
                          struct matrix *);
static const float input[5 * 32]
    __attribute__((aligned(64))) = {[0 ... 5 * 32 - 1] = 1};
static const float scales[5]
    __attribute__((aligned(64))) = {0.25f, 1, 4, 16, 64};
static const float expected[5] = {2, 1, 0.5f, 0.25f, 0.125f};
static const float weights[32] __attribute__((aligned(64))) = {0};
static float output[5 * 32 + 4]
    __attribute__((aligned(64))) = {[0 ... 5 * 32 + 3] = 17};

int main(void) {
  struct kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  struct matrix a = {(void *)input, (void *)input, 0, {5, 32}, {32, 1}};
  struct matrix s = {(void *)scales, (void *)scales, 0, {5, 1}, {1, 1}};
  struct vector w = {(void *)weights, (void *)weights, 0, {32}, {1}};
  struct matrix out = {output, output + 2, 0, {5, 32}, {32, 1}};
  _mlir_ciface_compute(&a, &s, &w, &out);
  for (unsigned row = 0; row < 5; ++row)
    if (output[2 + row * 32] != expected[row] ||
        output[33 + row * 32] != expected[row])
      return 1;
  if (output[0] != 17 || output[1] != 17 || output[162] != 17 ||
      output[163] != 17)
    return 2;
  return 0;
}
