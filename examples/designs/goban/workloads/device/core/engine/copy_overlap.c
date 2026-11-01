#include "copy.h"
#include <engine.h>
#include <stdio.h>

enum { ROWS = 512, WORDS = ROWS * (THREAD_MEM_BANK_WIDTH / 64) };
_Static_assert(THREAD_MEM_BANK_ENTRIES >= ROWS,
               "Copy overlap requires 512 bank rows");
static const uint64_t input[4][WORDS]
    __attribute__((aligned(64))) = {{[0 ... WORDS - 1] = 1},
                                    {[0 ... WORDS - 1] = 2},
                                    {[0 ... WORDS - 1] = 3},
                                    {[0 ... WORDS - 1] = 4}};
static uint64_t output[4][WORDS]
    __attribute__((section(".data"), aligned(64))) = {{0}};

static uint64_t cycle(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mcycle" : "=r"(value)::"memory");
  return value;
}

int main(void) {
  struct engine engine;
  struct engine_slot slots[4];
  struct copy_task args[4];
  struct engine_job jobs[4];
  struct engine_result results[4];
  engine_init(&engine, slots, 4);
  for (unsigned thread = 0; thread < 4; ++thread) {
    args[thread] = (struct copy_task){(uintptr_t)input[thread],
                                      (uintptr_t)output[thread], ROWS};
    jobs[thread] = (struct engine_job){
        .thread = thread,
        .task = {.id = thread + 1, .signature = slots[thread].signature},
        .opaque = &args[thread]};
  }
  uint64_t start = cycle();
  if (engine_run(&engine, jobs, 4, 0, 0, results))
    return 1;
  uint64_t elapsed = cycle() - start;
  printf("Core copy overlap: %u cycles\n", (unsigned)elapsed);
  for (unsigned thread = 0; thread < 4; ++thread) {
    if (results[thread].state != ENGINE_SUCCEEDED || results[thread].value)
      return 2;
    for (unsigned word = 0; word < WORDS; ++word)
      if (output[thread][word] != thread + 1)
        return 3;
  }
  return 0;
}
