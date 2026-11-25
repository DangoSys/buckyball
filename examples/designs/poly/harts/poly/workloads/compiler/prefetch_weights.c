#include <kernel.h>
#include <stdint.h>
#define BUCKYBALL_STREAM
#include <isa.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct vector {
  void *allocated, *aligned;
  int64_t offset, size, stride;
};
extern void _mlir_ciface_compute(struct matrix *, struct vector *,
                                 struct vector *, struct matrix *);
static const int8_t a[8][48]
    __attribute__((aligned(16))) = {[0 ... 6] = {[0 ... 31] = 64, [32] = 121},
                                    [7] = {[0 ... 31] = 96, [32] = 122}};
static const int8_t w0[32][48]
    __attribute__((aligned(16))) = {[0 ... 31] = {[0 ... 31] = 64, [32] = 121}};
static const int8_t w1[33][48]
    __attribute__((aligned(16))) = {[0 ... 31] = {[0 ... 31] = 64, [32] = 122}};
static float output[8][32]
    __attribute__((aligned(16))) = {[0 ... 7] = {[0 ... 31] = -999}};
static const float expected[8][32] = {
    [0 ... 6] = {[0 ... 31] = 2048}, [7] = {[0 ... 31] = 6144}};
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
  struct matrix am = {(void *)a, (void *)a, 0, {8, 48}, {48, 1}};
  struct vector wm0 = {(void *)w0, (void *)w0, 0, sizeof(w0), 1};
  struct vector wm1 = {(void *)w1, (void *)w1, 0, sizeof(w1), 1};
  struct matrix om = {output, output, 0, {8, 32}, {32, 1}};
  _mlir_ciface_compute(&am, &wm0, &wm1, &om);
  if (output[0][0] != 2048 || output[0][31] != 2048 || output[7][0] != 6144 ||
      output[7][31] != 6144)
    return 1;
  stream_begin(1, 0, 0);
  stream_job(kernel.slots[2].signature, 2, readback, 0, 0);
  stream_submit();
  stream_wait();
  return 0;
}
