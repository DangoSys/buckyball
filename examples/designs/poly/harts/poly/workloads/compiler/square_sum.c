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
void _mlir_ciface_square_sum(struct matrix *, struct vector *);

static const float input[5 * 65]
    __attribute__((aligned(64))) = {4096.0f, [1 ... 5 * 65 - 1] = 1.0f};
static float output[11] __attribute__((aligned(64))) = {[0 ... 10] = 17};

int main(void) {
  struct kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  const unsigned widths[] = {3, 32, 65};
  for (unsigned test = 0; test < 3; ++test) {
    for (unsigned i = 0; i < 11; ++i)
      output[i] = 17;
    struct matrix a = {
        (void *)input, (void *)input, 0, {5, widths[test]}, {65, 1}};
    struct vector out = {output, output, 3, {5}, {1}};
    _mlir_ciface_square_sum(&a, &out);
    // Ordered FP32 addition loses each unit after 4096 squared; a tree sum does
    // not.
    if (output[3] != 16777216.0f)
      return 1;
    for (unsigned row = 1; row < 5; ++row)
      if (output[3 + row] != widths[test])
        return 1;
    for (unsigned i = 0; i < 11; ++i)
      if ((i < 3 || i >= 8) && output[i] != 17)
        return 2;
  }
  return 0;
}
