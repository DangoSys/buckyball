#include "task.h"
#include "vectors.h"
#include <engine.h>
#include <stdio.h>
#include <string.h>

/* Runs quant_edges on the core's thread and compares the packed rows with
 * quantize_mxint8 of quant.py bit for bit. */
int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  struct ball_task args = {(uintptr_t)quant_edges_input,
                           (uintptr_t)quant_edges_output};
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
  if (memcmp(quant_edges_output + QUANT_EDGES_OUT_BASE * 16,
             quant_edges_expected, sizeof(quant_edges_expected)))
    return 4;
  printf("piplus core hart %u: mxint8quant quant_edges PASS\n", (unsigned)hart);
  return 0;
}
