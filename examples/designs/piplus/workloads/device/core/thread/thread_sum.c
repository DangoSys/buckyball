#include "sum.h"
#include <engine.h>
#include <stddef.h>
#include <stdio.h>

/* Runs on a core with one hardware thread: the CPU computes, then hands the
 * same sum to its thread, over 2 words and over all of them. The thread
 * leaves each total in its TLS, read back here, and returns its own cycles
 * (mcycle), which must grow with the work. */
int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  struct sum_task args[2];
  uint64_t expected[2] = {0, 0}, totals[2], cycles[2];
  for (unsigned job = 0; job < 2; ++job) {
    args[job].count = job ? SUM_WORDS : 2;
    args[job].total = 0;
    for (unsigned i = 0; i < SUM_WORDS; ++i) {
      args[job].words[i] = (hart << 32) | (i * 3 + 1);
      if (i < args[job].count)
        expected[job] += args[job].words[i] * (i + 1);
    }
  }
  printf("piplus core hart %u: CPU sum %08x%08x\n", (unsigned)hart,
         (unsigned)(expected[1] >> 32), (unsigned)expected[1]);

  struct engine engine;
  struct engine_slot slots[1];
  engine_init(&engine, slots, 1);
  if (engine.thread_count != 1)
    return 1;
  for (unsigned job = 0; job < 2; ++job) {
    struct engine_job request = {
        .thread = 0,
        .task = {.id = job + 1, .signature = slots[0].signature},
        .opaque = &args[job]};
    struct engine_result result;
    engine_begin(&engine, &request, 1, 0, 0, &result);
    engine_wait(&engine);
    if (result.state != ENGINE_SUCCEEDED)
      return 2;
    // The task sits at the start of the thread's TLS.
    thread_store(0, THREAD_TLS, offsetof(struct sum_task, total), 0, 8);
    totals[job] = thread_tss_read(0);
    cycles[job] = result.value;
    engine_end(&engine);
    if (totals[job] != expected[job])
      return 3;
  }
  printf("piplus core hart %u: thread sum %08x%08x, %u cycles for 2 words, "
         "%u for %u\n",
         (unsigned)hart, (unsigned)(totals[1] >> 32), (unsigned)totals[1],
         (unsigned)cycles[0], (unsigned)cycles[1], SUM_WORDS);
  if (!cycles[0] || cycles[1] <= cycles[0])
    return 4;
  return 0;
}
