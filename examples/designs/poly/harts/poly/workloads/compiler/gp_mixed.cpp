#include <cmath>
#include <cstdint>
#include <kernel.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern "C" void _mlir_ciface_compute_mixed(matrix *, matrix *, matrix *);
alignas(64) static float input[480], narrow[488], wide[488];
static float expected[17];
uint64_t mixed_cycles[2];

int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  for (unsigned i = 0; i < 480; ++i)
    input[i] = (int(i % 17) - 8) * .25f;
  for (unsigned i = 0; i < 488; ++i)
    narrow[i] = wide[i] = 17;
  matrix in{input, input, 0, {15, 32}, {32, 1}};
  matrix n{narrow, narrow, 4, {15, 32}, {32, 1}};
  matrix w{wide, wide, 4, {15, 32}, {32, 1}};
  asm volatile("rdcycle %0" : "=r"(mixed_cycles[0]));
  _mlir_ciface_compute_mixed(&in, &n, &w);
  asm volatile("rdcycle %0" : "=r"(mixed_cycles[1]));
  for (unsigned i = 0; i < 17; ++i)
    expected[i] = ((int(i) - 8) * .25f) * ((int(i) - 8) * .25f);
  for (unsigned i = 0; i < 480; ++i)
    if (narrow[4 + i] != 2 * input[i] || wide[4 + i] != expected[i % 17])
      return 1;
  for (unsigned i = 0; i < 4; ++i)
    if (narrow[i] != 17 || wide[i] != 17 || narrow[484 + i] != 17 ||
        wide[484 + i] != 17)
      return 2;
  return 0;
}
