#include "ant_shared.h"
#include <thread.h>
extern const unsigned char ant_image[], ant_image_end[];
unsigned shared_done[POLY_THREAD_CORES];
void shared_exercise(unsigned core) {
  const uint32_t contexts[] = {thread_find(THREAD_SIGNATURE, 0),
                               thread_find(THREAD_SIGNATURE, 1)};
  size_t bytes = ant_image_end - ant_image;
  uint64_t value = 100 + core,
           shared = thread_query(contexts[0], THREAD_FIELD_TSS_BASE),
           zero[2] = {0};
  struct thread_task tasks[2];
  for (unsigned context = 0; context < 2; ++context) {
    uint64_t base = thread_query(contexts[context], THREAD_FIELD_TLS_BASE);
    if (bytes > thread_query(contexts[context], THREAD_FIELD_CODE_BYTES) ||
        thread_query(contexts[context], THREAD_FIELD_SIGNATURE) !=
            THREAD_SIGNATURE)
      __builtin_trap();
    tasks[context] = (struct thread_task){
        context + 1,
        0,
        bytes,
        base,
        base + thread_query(contexts[context], THREAD_FIELD_TLS_BYTES),
        THREAD_SIGNATURE};
    struct ant_shared_args args = {shared, context == 0, context ? 0 : value};
    thread_tss_write(0, ant_image, bytes);
    thread_load(contexts[context], THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
    thread_tss_write(0, &args, sizeof(args));
    thread_load(contexts[context], THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
  }
  thread_tss_write(0, zero, sizeof(zero));
  thread_acquire();
  thread_start(contexts[1], &tasks[1]);
  thread_start(contexts[0], &tasks[0]);
  for (unsigned context = 0; context < 2; ++context) {
    int cancelled;
    if (thread_wait(contexts[context], tasks[context].id, &cancelled) !=
            value ||
        cancelled)
      __builtin_trap();
  }
  thread_release();
  for (unsigned context = 0; context < 2; ++context) {
    thread_store(contexts[context], THREAD_TLS, 16, 0, 8);
    if (thread_tss_read(0) != value)
      __builtin_trap();
  }
  thread_tss_write(0, zero, sizeof(zero));
  thread_acquire();
  thread_start(contexts[1], &tasks[1]);
  thread_cancel(contexts[1]);
  int cancelled;
  thread_wait(contexts[1], tasks[1].id, &cancelled);
  if (!cancelled)
    __builtin_trap();
  thread_release();
  uint64_t ready[2] = {value + 7, 1};
  thread_tss_write(0, ready, sizeof(ready));
  ++tasks[1].id;
  thread_acquire();
  thread_start(contexts[1], &tasks[1]);
  if (thread_wait(contexts[1], tasks[1].id, &cancelled) != value + 7 ||
      cancelled)
    __builtin_trap();
  thread_release();
  __atomic_store_n(&shared_done[core], 1, __ATOMIC_RELEASE);
}
