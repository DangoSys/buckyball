// Two runtime jobs on the core under test: the first moves data through a
// bank of its accelerator, the second depends on it and checks the data.
#include <bbhw/isa/isa.h>
#include <instances.h>
#include <params.h>
#include <runtime.h>
#include <stdint.h>
#include <stdio.h>

static uint64_t input[2] __attribute__((aligned(64)));
static uint64_t output[2] __attribute__((aligned(64)));

static int move(void *core, void *argument) {
  (void)core;
  (void)argument;
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)input, 1, 1, 1);
  bb_mvout((uintptr_t)output, 1, 1, 1);
  bb_mem_release(1);
  return 0;
}

static int check(void *core, void *argument) {
  (void)core;
  (void)argument;
  return output[0] != input[0] || output[1] != input[1];
}

int main(void) {
  input[0] = 0x0123456789abcdefULL;
  input[1] = ~input[0];
  const uint64_t signature = bb_cores[0].signature;
  const struct runtime_core core =
      runtime_core(bb_cores[0].hart, signature, NULL);
  static const size_t after_move[] = {0};
  const struct runtime_job jobs[] = {
      {signature, RUNTIME_ANY, move, NULL, NULL, 0},
      {signature, RUNTIME_ANY, check, NULL, after_move, 1},
  };
  struct runtime_result results[2];
  struct runtime runtime;
  runtime_init(&runtime, &core, 1, jobs, 2, results);
  runtime_baremetal_run(&runtime, 0);
  for (unsigned job = 0; job < 2; ++job) {
    if (results[job].state != RUNTIME_SUCCEEDED || results[job].status) {
      printf("runtime job %u state=%d status=%d\n", job, results[job].state,
             results[job].status);
      return 1;
    }
  }
  puts("Runtime job PASS");
  return 0;
}
