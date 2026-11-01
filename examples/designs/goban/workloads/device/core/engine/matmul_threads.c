#include <bb_cores.h>
#include <bb_isa.h>
#include <engine.h>
#include <runtime.h>
#include <stdint.h>
#include <stream.h>

// The matmul threads run the engine's stream interpreter: a job is a stream
// of Ball instructions, recorded here and executed by its thread.
static struct stream *recording;
#undef BUCKYBALL_INSTRUCTION_R_R
#define BUCKYBALL_INSTRUCTION_R_R(rs1, rs2, func7)                             \
  (recording->instructions[recording->count++] =                               \
       (struct stream_instruction){(func7), (uint64_t)(rs1), (uint64_t)(rs2)})
#include <isa/smatmul.h>

enum { INSTRUCTIONS = 16 };
static struct {
  uint64_t count;
  struct stream_instruction instructions[INSTRUCTIONS];
} streams[4] __attribute__((aligned(8)));

static int8_t a[4][256] __attribute__((aligned(64)));
static int8_t b[4][256] __attribute__((aligned(64)));
static int32_t bias[4][16] __attribute__((aligned(64)));
static int32_t output[4][256] __attribute__((aligned(64)));
static struct engine_job jobs[4];
static struct engine_result results[4];

// One 16x16x16 INT8 product with an INT32 bias on one thread.
static void record(unsigned tid) {
  recording = (struct stream *)&streams[tid];
  recording->count = 0;
  // mset: allocate banks 0-3 with one row and one column group.
  for (unsigned bank = 0; bank < 4; ++bank)
    BUCKYBALL_INSTRUCTION_R_R(
        BB_BANK0(bank), FIELD(1, 0, 4) | FIELD(1, 5, 9) | FIELD(1, 10, 10),
        BB_MSET_FUNC7);
  // mvin: bias to bank 0, A to bank 1, B to bank 2, stride 1.
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK2(0) | BB_ITER(4),
      FIELD((uintptr_t)bias[tid], 0, 38) | FIELD(1, 39, 57), BB_MVIN_FUNC7);
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK2(1) | BB_ITER(16),
                            FIELD((uintptr_t)a[tid], 0, 38) | FIELD(1, 39, 57),
                            BB_MVIN_FUNC7);
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK2(2) | BB_ITER(16),
                            FIELD((uintptr_t)b[tid], 0, 38) | FIELD(1, 39, 57),
                            BB_MVIN_FUNC7);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, 3, 16, 16, 16, 1, 1, 0);
  // mvout: bank 3 to the output, stride 1.
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(3) | BB_ITER(64),
      FIELD((uintptr_t)output[tid], 0, 38) | FIELD(1, 39, 57), BB_MVOUT_FUNC7);
  // mset: release banks 0-3.
  for (unsigned bank = 0; bank < 4; ++bank)
    BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(bank), 0, BB_MSET_FUNC7);
}

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
    record(tid);
    jobs[tid] = (struct engine_job){
        .thread = tid,
        .task = {.id = tid + 1, .signature = engine->slots[tid].signature},
        .opaque = &streams[tid]};
  }
  engine_run(engine, jobs, 4, 0, 0, results);
  for (unsigned tid = 0; tid < 4; ++tid) {
    // The stream interpreter returns the thread's cycles in the job.
    if (results[tid].state != ENGINE_SUCCEEDED || !results[tid].value)
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
