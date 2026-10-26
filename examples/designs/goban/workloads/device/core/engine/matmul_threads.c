#include <engine.h>
#include <instances.h>
#include <matmul.h>
#include <runtime.h>
#include <stdint.h>

static int8_t a[4][256] __attribute__((aligned(64)));
static int8_t b[4][256] __attribute__((aligned(64)));
static int32_t bias[4][16] __attribute__((aligned(64)));
static int32_t output[4][256] __attribute__((aligned(64)));
static struct matmul_task args[4];
static struct engine_job jobs[4];
static struct engine_result results[4];

static int compute(void *context, void *argument) {
  (void)argument;
  struct engine *engine = context;
  if (engine->thread_count != 4)
    return 1;
  for (unsigned tid = 0; tid < 4; ++tid) {
    for (unsigned i = 0; i < 256; ++i) {
      a[tid][i] = (i * 3 + tid) % 13 - 6;
      b[tid][i] = (i * 5 + tid) % 11 - 5;
    }
    for (unsigned i = 0; i < 16; ++i)
      bias[tid][i] = i - 8 + tid;
    args[tid] =
        (struct matmul_task){(uintptr_t)a[tid], (uintptr_t)b[tid],
                             (uintptr_t)bias[tid], (uintptr_t)output[tid]};
    jobs[tid] = (struct engine_job){
        .thread = tid,
        .task = {.id = tid + 1, .signature = engine->slots[tid].signature},
        .opaque = &args[tid]};
  }
  engine_run(engine, jobs, 4, results);
  for (unsigned tid = 0; tid < 4; ++tid) {
    if (results[tid].state != ENGINE_SUCCEEDED || results[tid].value)
      return 2;
    for (unsigned row = 0; row < 16; ++row)
      for (unsigned col = 0; col < 16; ++col) {
        int32_t expected = bias[tid][col];
        for (unsigned k = 0; k < 16; ++k)
          expected += a[tid][row * 16 + k] * b[tid][k * 16 + col];
        if (output[tid][row * 16 + col] != expected)
          return 3;
      }
  }
  return 0;
}

int main(void) {
  struct runtime_core core =
      runtime_core(0, bb_cores[bb_hart_slot(0)].signature, 0);
  struct runtime_job job = {
      .signature = core.signature, .core = 0, .entry = compute};
  struct runtime_result result;
  struct runtime runtime;
  runtime_init(&runtime, &core, 1, &job, 1, &result);
  runtime_baremetal_run(&runtime, 0);
  return result.state == RUNTIME_SUCCEEDED ? result.status : 4;
}
