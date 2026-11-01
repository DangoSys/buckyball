#include "copy.h"
#include <engine.h>

static uint64_t input[16] __attribute__((aligned(64)));
static uint64_t intermediate[5][16] __attribute__((aligned(64)));
static uint64_t output[2][16] __attribute__((aligned(64)));

int main(void) {
  struct engine engine;
  struct engine_slot slots[4];
  engine_init(&engine, slots, 4);
  if (engine.thread_count != 4)
    return 1;
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    for (unsigned word = 0; word < 16; ++word) {
      input[word] = ((uint64_t)(iteration + 1) << 32) | (word + 1);
      output[0][word] = output[1][word] = 0;
      for (unsigned buffer = 0; buffer < 5; ++buffer)
        intermediate[buffer][word] = UINT64_C(0xa5a5a5a5a5a5a5a5);
    }
    struct copy_task args[] = {
        {(uintptr_t)input, (uintptr_t)intermediate[0], 8},
        {(uintptr_t)intermediate[1], (uintptr_t)output[0], 8},
        {(uintptr_t)intermediate[2], (uintptr_t)intermediate[3], 8},
        {(uintptr_t)intermediate[4], (uintptr_t)output[1], 8}};
    struct engine_job jobs[4];
    struct engine_result results[4];
    for (unsigned thread = 0; thread < 4; ++thread)
      jobs[thread] = (struct engine_job){
          .thread = thread,
          .task = {.id = thread + 1, .signature = slots[thread].signature},
          .opaque = &args[thread]};
    struct engine_transfer edges[] = {
        {0, 1, args[0].output, args[1].input, sizeof(input), 0},
        {0, 2, args[0].output, args[2].input, sizeof(input), 0},
        {2, 3, args[2].output, args[3].input, sizeof(input), 0}};
    if (engine_run(&engine, jobs, 4, edges, 3, results))
      return 2;
    for (unsigned thread = 0; thread < 4; ++thread)
      if (results[thread].state != ENGINE_SUCCEEDED || results[thread].value)
        return 3;
    for (unsigned word = 0; word < 16; ++word) {
      if (output[0][word] != input[word] || output[1][word] != input[word])
        return 4;
      for (unsigned buffer = 0; buffer < 5; ++buffer)
        if (intermediate[buffer][word] != UINT64_C(0xa5a5a5a5a5a5a5a5))
          return 5;
    }
  }
  return 0;
}
