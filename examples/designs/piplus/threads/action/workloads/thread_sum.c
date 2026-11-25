#include "sum.h"
#include <kernel.h>
#include <stddef.h>
#include <stdio.h>

/* Runs on a hart with up to four hardware threads: the CPU computes, then
 * hands the same sum to each thread in turn, over 2 words and over all of
 * them. A thread leaves each total in its TLS, read back here, and returns its
 * own cycles (mcycle), which must grow with the work. One line at the end:
 * every character of the UART costs the hart a round trip to the device. */
int main(void) {
  uintptr_t hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  struct sum_task args[2];
  uint64_t expected[2] = {0, 0}, totals[2], cycles[2];
  for (unsigned job = 0; job < 2; ++job) {
    args[job].count = job ? SUM_WORDS : 2;
    args[job].total = 0;
    for (unsigned i = 0; i < SUM_WORDS; ++i) {
      args[job].words[i] = (hartid << 32) | (i * 3 + 1);
      if (i < args[job].count)
        expected[job] += args[job].words[i] * (i + 1);
    }
  }
  struct kernel kernel;
  struct kernel_slot slots[4];
  kernel_init(&kernel, slots, 4);
  if (kernel.thread_count == 0)
    return 1;
  uint64_t slowest[2] = {0, 0};
  for (uint32_t thread = 0; thread < kernel.thread_count; ++thread) {
    for (unsigned job = 0; job < 2; ++job) {
      args[job].total = 0;
      struct kernel_job request = {
          .thread = thread,
          .task = {.id = job + 1, .signature = slots[thread].signature},
          .opaque = &args[job]};
      struct kernel_result result;
      kernel_begin(&kernel, &request, 1, 0, 0, &result);
      kernel_wait(&kernel);
      if (result.state != KERNEL_SUCCEEDED)
        return 2;
      // The task sits at the start of the thread's TLS.
      thread_store(thread, THREAD_TLS, offsetof(struct sum_task, total), 0, 8);
      totals[job] = thread_tss_read(0);
      cycles[job] = result.value;
      kernel_end(&kernel);
      if (totals[job] != expected[job])
        return 3;
    }
    if (!cycles[0] || cycles[1] <= cycles[0])
      return 4;
    for (unsigned job = 0; job < 2; ++job)
      if (cycles[job] > slowest[job])
        slowest[job] = cycles[job];
  }
  printf("piplus hart hart %u: %u threads, thread sum %08x%08x, at most %u "
         "cycles for 2 words, %u for %u\n",
         (unsigned)hartid, (unsigned)kernel.thread_count,
         (unsigned)(expected[1] >> 32), (unsigned)expected[1],
         (unsigned)slowest[0], (unsigned)slowest[1], SUM_WORDS);
  return 0;
}
