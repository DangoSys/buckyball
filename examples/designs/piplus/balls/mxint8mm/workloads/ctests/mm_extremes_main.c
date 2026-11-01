#include "task.h"
#include "vectors.h"
#include <engine.h>
#include <stdio.h>
#include <string.h>

/* Runs mm_extremes on the core's thread and compares C with the reference of
 * vectors.py bit for bit. */
int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  struct ball_task args = {(uintptr_t)mm_extremes_a, (uintptr_t)mm_extremes_b,
                           (uintptr_t)mm_extremes_c};
  struct engine engine;
  struct engine_slot slots[1];
  engine_init(&engine, slots, 1);
  if (engine.thread_count != 1)
    return 1;
  struct engine_job job = {.thread = 0,
                           .task = {.id = 1, .signature = slots[0].signature},
                           .opaque = &args};
  struct engine_result result;
  if (engine_run(&engine, &job, 1, 0, 0, &result))
    return 2;
  if (result.state != ENGINE_SUCCEEDED || result.value != 0)
    return 3;
  if (memcmp(mm_extremes_c + MM_EXTREMES_C_BASE * 4, mm_extremes_expected,
             sizeof(mm_extremes_expected)))
    return 4;
  printf("piplus core hart %u: mxint8mm mm_extremes PASS\n", (unsigned)hart);
  return 0;
}
