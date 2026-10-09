#include "ant_tlink_reduce.h"
#include <ant.h>
#include <params.h>
extern const unsigned char ant_image[], ant_image_end[];
void ant_tlink_reduce_phase(unsigned tile,
                            const struct ant_tlink_reduce_args *args) {
  unsigned context = BB_ANT_CONTEXT;
  size_t bytes = ant_image_end - ant_image;
  ant_write(context, ANT_TLS, 0, args, sizeof(*args));
  uint64_t base = ant_query(context, ANT_TLS_BASE);
  struct ant_task task = {tile * 3 + args->phase + 1,
                          0,
                          bytes,
                          base,
                          base + ant_query(context, ANT_TLS_BYTES),
                          CORE_SIGNATURE};
  ant_acquire();
  ant_start(context, &task);
  int cancelled;
  if (ant_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  ant_release();
}
void ant_tlink_reduce_prepare(unsigned tile, float *input, float *output) {
  size_t bytes = ant_image_end - ant_image;
  if (bytes > ant_query(BB_ANT_CONTEXT, ANT_CODE_BYTES) ||
      ant_query(BB_ANT_CONTEXT, ANT_SIGNATURE) != CORE_SIGNATURE)
    __builtin_trap();
  ant_write(BB_ANT_CONTEXT, ANT_CODE, 0, ant_image, bytes);
  for (unsigned i = 0; i < 272; ++i)
    input[i] = 0.0f;
  input[0] = tile % 2 ? -(float)(tile + 1) * 0.5f : (float)(tile + 1) * 1.25f;
  for (unsigned column = 0; column < 16; ++column)
    input[16 + column * 16] = (float)(column + 1);
  for (unsigned i = 0; i < 8; ++i)
    output[i] = -77.0f;
}
