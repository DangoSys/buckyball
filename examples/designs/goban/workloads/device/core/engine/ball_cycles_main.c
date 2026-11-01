#include <stdint.h>
#include <stdio.h>
#include <thread.h>

static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "ball_cycles.bin"
};
static int8_t a[256] __attribute__((aligned(64)));
static int8_t b[256] __attribute__((aligned(64)));
static int32_t bias[16] __attribute__((aligned(64)));
static int32_t output[256] __attribute__((aligned(64)));

int main(void) {
  for (unsigned i = 0; i < 256; ++i)
    a[i] = b[i] = 1;
  for (unsigned i = 0; i < 16; ++i)
    bias[i] = 3;
  uint32_t thread = thread_find(THREAD_SIGNATURE, 0);
  uint64_t base = thread_query(thread, THREAD_FIELD_TLS_BASE);
  struct thread_task task = {1,
                             0,
                             sizeof(image),
                             base,
                             base +
                                 thread_query(thread, THREAD_FIELD_TLS_BYTES),
                             THREAD_SIGNATURE};
  uint64_t args[] = {(uintptr_t)a, (uintptr_t)b, (uintptr_t)bias,
                     (uintptr_t)output};
  thread_tss_write(0, image, sizeof(image));
  thread_load(thread, THREAD_CODE, 0, 0, (sizeof(image) + 7) & ~(size_t)7);
  thread_tss_write(0, args, sizeof(args));
  thread_load(thread, THREAD_TLS, 0, 0, sizeof(args));
  thread_acquire();
  thread_start(thread, &task);
  int cancelled;
  if (thread_wait(thread, task.id, &cancelled) || cancelled)
    return 1;
  thread_release();
  thread_store(thread, THREAD_TLS, 32, 0, 112);
  for (unsigned i = 0; i < 14; ++i)
    printf("Ball timing %u: %u cycles\n", i, (unsigned)thread_tss_read(i * 8));
  for (unsigned i = 0; i < 256; ++i)
    if (output[i] != 19)
      return 2;
  return 0;
}
