#include "ant_corelink_reduce.h"
#include <thread.h>
extern const unsigned char ant_image[], ant_image_end[];
void ant_corelink_reduce_phase(unsigned core,
                               const struct ant_corelink_reduce_args *args) {
  unsigned context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = ant_image_end - ant_image;
  thread_tss_write(0, args, sizeof(*args));
  thread_load(context, THREAD_TLS, 0, 0, (sizeof(*args) + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
  struct thread_task task = {core * 3 + args->phase + 1,
                             0,
                             bytes,
                             base,
                             base +
                                 thread_query(context, THREAD_FIELD_TLS_BYTES),
                             THREAD_SIGNATURE};
  thread_acquire();
  thread_start(context, &task);
  int cancelled;
  if (thread_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
}
void ant_corelink_reduce_prepare(unsigned core, float *input, float *output) {
  size_t bytes = ant_image_end - ant_image;
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  thread_tss_write(0, ant_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  for (unsigned i = 0; i < 272; ++i)
    input[i] = 0.0f;
  input[0] = core % 2 ? -(float)(core + 1) * 0.5f : (float)(core + 1) * 1.25f;
  for (unsigned column = 0; column < 16; ++column)
    input[16 + column * 16] = (float)(column + 1);
  for (unsigned i = 0; i < 8; ++i)
    output[i] = -77.0f;
}
