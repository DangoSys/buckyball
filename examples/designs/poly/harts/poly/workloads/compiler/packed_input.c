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
static const int8_t a[5][4224]
    __attribute__((aligned(16))) = {[0 ... 3] = {[0 ... 2047] = 64,
                                                 [2048 ... 4095] = 96,
                                                 [4096 ... 4159] = 121,
                                                 [4160 ... 4223] = 122},
                                    [4] = {[0 ... 2047] = 96,
                                           [2048 ... 4095] = 64,
                                           [4096 ... 4159] = 122,
                                           [4160 ... 4223] = 121}};
static const int8_t w[2][8][2112] __attribute__((aligned(16))) = {
    [0] = {[0 ... 7] = {[0 ... 2047] = 64, [2048 ... 2111] = 121}},
    [1] = {[0 ... 3] = {[0 ... 2047] = 64, [2048 ... 2111] = 121},
           [4 ... 7] = {[0 ... 2047] = 64, [2048 ... 2111] = 122}}};
static float output[5][8] __attribute__((aligned(16)));
static const float expected[5][8] = {
    [0 ... 3] = {[0 ... 3] = 8192, [4 ... 7] = 14336},
    [4] = {[0 ... 3] = 8192, [4 ... 7] = 10240}};
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
  struct matrix am = {(void *)a, (void *)a, 0, {5, 4224}, {4224, 1}};
  struct weights wm = {(void *)w, (void *)w, 0, {2, 8, 2112}, {16896, 2112, 1}};
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
