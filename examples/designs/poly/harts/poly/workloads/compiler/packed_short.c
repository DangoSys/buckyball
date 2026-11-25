#include <kernel.h>
#include <stdint.h>
#define BUCKYBALL_STREAM
#include <isa.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct weights {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct matrix *, struct weights *,
                                 struct matrix *);
static const int8_t a[5][132] __attribute__((aligned(16))) = {
    [0 ... 3] = {[0 ... 127] = 64, [128 ... 131] = 121},
    [4] = {[0 ... 127] = 96, [128 ... 131] = 122}};
static const int8_t w[1][8][132] __attribute__((aligned(16))) = {
    [0] = {[0 ... 3] = {[0 ... 127] = 64, [128 ... 131] = 121},
           [4 ... 7] = {[0 ... 127] = 64, [128 ... 131] = 122}}};
static float output[5][8] __attribute__((aligned(16)));
static const float expected[5][8] = {
    [0 ... 3] = {[0 ... 3] = 128, [4 ... 7] = 256},
    [4] = {[0 ... 3] = 384, [4 ... 7] = 768}};
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
  struct matrix am = {(void *)a, (void *)a, 0, {5, 132}, {132, 1}};
  struct weights wm = {(void *)w, (void *)w, 0, {1, 8, 132}, {1056, 132, 1}};
  struct matrix om = {output, output, 0, {5, 8}, {8, 1}};
  _mlir_ciface_compute(&am, &wm, &om);
  for (unsigned row = 0; row < 5; ++row)
    if (output[row][0] != expected[row][0] ||
        output[row][7] != expected[row][7])
      return 1;
  stream_begin(1, 0, 0);
  stream_job(kernel.slots[2].signature, 2, readback, 0, 0);
  stream_submit();
  stream_wait();
  return 0;
}
