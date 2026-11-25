#include <kernel.h>
#include <stdint.h>
#define BUCKYBALL_STREAM
#include <isa.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern void _mlir_ciface_compute(struct matrix *, struct matrix *,
                                 struct matrix *);
static const int8_t a[26][1056] __attribute__((aligned(16))) = {
    [0 ... 24] = {[0 ... 1023] = 64, [1024 ... 1055] = 121},
    [25] = {[0 ... 1023] = 96, [1024 ... 1055] = 122}};
static const int8_t w[64][1056] __attribute__((aligned(16))) = {
    [0 ... 31] = {[0 ... 1023] = 64, [1024 ... 1055] = 121},
    [32 ... 63] = {[0 ... 1023] = 64, [1024 ... 1055] = 122}};
static float output[26][64]
    __attribute__((aligned(16))) = {[0 ... 25] = {[0 ... 63] = -999}};
static const float expected[26][64] = {
    [0 ... 24] = {[0 ... 31] = 1024, [32 ... 63] = 2048},
    [25] = {[0 ... 31] = 3072, [32 ... 63] = 6144}};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {output, expected,
                                                              sizeof(output)};
static void readback(void *argument) {
  (void)argument;
  bb_mem_alloc(31, 1, 1);
  bb_mvin((uintptr_t)output, 31, sizeof(output) / 16, 1);
  bb_mem_release(31);
}
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  struct matrix am = {(void *)a, (void *)a, 0, {26, 1056}, {1056, 1}};
  struct matrix wm = {(void *)w, (void *)w, 0, {64, 1056}, {1056, 1}};
  struct matrix om = {output, output, 0, {26, 64}, {64, 1}};
  _mlir_ciface_compute(&am, &wm, &om);
  const unsigned rows[] = {0, 14, 15, 25}, columns[] = {0, 31, 32, 63};
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned column = 0; column < 4; ++column)
      if (output[rows[row]][columns[column]] !=
          expected[rows[row]][columns[column]])
        return 1;
  stream_begin(1, 0, 0);
  stream_job(kernel.slots[2].signature, 2, readback, 0, 0);
  stream_submit();
  stream_wait();
  return 0;
}
