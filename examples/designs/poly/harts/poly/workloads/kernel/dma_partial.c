#include <stdint.h>
#include <stdio.h>
#include <thread.h>
static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "dma_partial.bin"
};

static const uint8_t input[192] __attribute__((aligned(64))) = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,        15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,        32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,        49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, [131] = 0, 13, 26, 39};
static uint8_t output[64] __attribute__((aligned(64))) = {[0 ... 63] = 0xa5};

int main(void) {
  const unsigned thread = 1;
  uint64_t arguments[] = {(uintptr_t)input, (uintptr_t)output};
  uint64_t base = thread_query(thread, THREAD_FIELD_TLS_BASE);
  thread_acquire();
  thread_tss_write(0, image, sizeof(image));
  thread_load(thread, THREAD_CODE, 0, 0, (sizeof(image) + 7) & ~7u);
  thread_tss_write(0, arguments, sizeof(arguments));
  thread_load(thread, THREAD_TLS, 0, 0, sizeof(arguments));
  struct thread_task task = {1,
                             0,
                             sizeof(image),
                             base,
                             base +
                                 thread_query(thread, THREAD_FIELD_TLS_BYTES),
                             thread_query(thread, THREAD_FIELD_SIGNATURE)};
  thread_start(thread, &task);
  int cancelled;
  if (thread_wait(thread, task.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
  for (unsigned i = 0; i < sizeof(output); ++i) {
    unsigned row = i / 20, column = i % 20;
    uint8_t expected = row >= 2      ? 0xa5
                       : column < 16 ? input[row * 16 + column]
                                     : input[32 + row * 16 + column - 16];
    if (i >= 48)
      expected = i < 52 ? input[131 + i - 48] : 0;
    if (output[i] != expected) {
      printf("Partial DMA mismatch at %u: %u != %u\n", i, output[i], expected);
      return 1;
    }
  }
  return 0;
}
