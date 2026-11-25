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
static float a[33][32] __attribute__((aligned(16))) = {
    [0 ... 31] = {[0 ... 31] = 1}, [32] = {[0 ... 31] = 3}};
static int8_t w[256][48] __attribute__((aligned(16))) = {
    [0 ... 63] = {[0 ... 31] = 64, [32] = 121},
    [64 ... 127] = {[0 ... 31] = 64, [32] = 122},
    [128 ... 191] = {[0 ... 31] = 96, [32] = 122},
    [192 ... 255] = {[0 ... 31] = 64, [32] = 123}};
static float output[33][256]
    __attribute__((aligned(16))) = {[0 ... 32] = {[0 ... 255] = -999}};
static const float expected[33][256] = {[0 ... 31] = {[0 ... 63] = 32,
                                                      [64 ... 127] = 64,
                                                      [128 ... 191] = 96,
                                                      [192 ... 255] = 128},
                                        [32] = {[0 ... 63] = 96,
                                                [64 ... 127] = 192,
                                                [128 ... 191] = 288,
                                                [192 ... 255] = 384}};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {output, expected,
                                                              sizeof(output)};
enum { READBACK_BYTES = OUTPUT_READBACK_ROWS * 16 };
static void readback(void *argument) {
  (void)argument;
  bb_mem_alloc(31, 1, 1);
  for (unsigned first = 0; first < sizeof(output); first += READBACK_BYTES) {
    unsigned bytes = sizeof(output) - first;
    if (bytes > READBACK_BYTES)
      bytes = READBACK_BYTES;
    bb_mvin((uintptr_t)((unsigned char *)output + first), 31, bytes / 16, 1);
  }
  bb_mem_release(31);
}
int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  struct matrix am = {a[0], a[0], 0, {33, 32}, {32, 1}};
  struct matrix wm = {w[0], w[0], 0, {256, 48}, {48, 1}};
  struct matrix out = {output[0], output[0], 0, {33, 256}, {256, 1}};
  _mlir_ciface_compute(&am, &wm, &out);
  // Check every row at both sides of each column-tile boundary coherently.
  // Traced regressions check the full result from the DMA readback.
  const unsigned columns[] = {0, 127, 128, 255};
  for (unsigned row = 0; row < 33; ++row)
    for (unsigned column = 0; column < 4; ++column)
      if (output[row][columns[column]] != expected[row][columns[column]])
        return 1;
  stream_begin(1, 0, 0);
  stream_job(kernel.slots[2].signature, 2, readback, 0, 0);
  stream_submit();
  stream_wait();
  return 0;
}
