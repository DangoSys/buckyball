#include <stdint.h>
#include <stdio.h>
#include <thread.h>

static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "ant_cycles.bin"
};

int main(void) {
  uint32_t thread = thread_find(THREAD_SIGNATURE, 0);
  uint64_t base = thread_query(thread, THREAD_FIELD_TLS_BASE);
  struct thread_task task = {1,
                             0,
                             sizeof(image),
                             base,
                             base +
                                 thread_query(thread, THREAD_FIELD_TLS_BYTES),
                             THREAD_SIGNATURE};
  thread_tss_write(0, image, sizeof(image));
  thread_load(thread, THREAD_CODE, 0, 0, (sizeof(image) + 7) & ~(size_t)7);
  thread_acquire();
  thread_start(thread, &task);
  int cancelled;
  if (thread_wait(thread, task.id, &cancelled) || cancelled)
    return 1;
  thread_release();
  thread_store(thread, THREAD_TLS, 0, 0, 64);
  const unsigned expected[] = {68, 116, 100, 228, 1124, 612, 100, 68};
  int failed = 0;
  for (unsigned i = 0; i < 8; ++i) {
    uint64_t cycles = thread_tss_read(i * 8);
    printf("Ant timing %u: %u cycles, expected %u\n", i, (unsigned)cycles,
           expected[i]);
    failed |= cycles != expected[i];
  }
  return failed;
}
