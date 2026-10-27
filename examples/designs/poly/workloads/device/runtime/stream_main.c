#include <bb_cores.h>
#include <bbhw/isa/isa.h>
#include <isa/mxmm.h>
#include <linux.h>
#include <stdio.h>
#include <stdlib.h>
#include <stream.h>
#include <workspace.h>
enum { M = 16, N = 16, K = 32 };
static float a[M * K] __attribute__((aligned(4096)));
static float b[N * K] __attribute__((aligned(4096)));
static float output[M * N] __attribute__((aligned(4096)));
static void job(void *argument) {
  (void)argument;
  for (unsigned bank = 3; bank <= 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)a, 3, sizeof(a) / 16, 1);
  // A DMA call has completed before the CPU is allowed to reuse its source.
  for (unsigned i = 0; i < M * K; ++i)
    a[i] = -1;
  bb_mvin((uintptr_t)b, 4, sizeof(b) / 16, 1);
  bb_mxmm_f32(3, 4, 5, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)output, 5, sizeof(output) / 16, 1);
  for (unsigned row = 0; row < M; ++row)
    for (unsigned col = 0; col < N; ++col)
      if (output[row * N + col] != (float)(K * (row + 1) * (col + 1)))
        abort();
  for (unsigned bank = 3; bank <= 5; ++bank)
    bb_mem_release(bank);
}
// Runs on the core; the stream executes its Ball instructions on a hardware
// thread.
static int run(void *engine, void *argument) {
  (void)engine;
  (void)argument;
  void *workspace = aligned_alloc(64, 1024 * 1024);
  if (!workspace)
    abort();
  workspace_init(workspace, 1024 * 1024);
  workspace_begin(workspace, 1024 * 1024);
  stream_run(THREAD_SIGNATURE, job, NULL);
  free(workspace);
  return 0;
}
int main(void) {
  for (unsigned row = 0; row < M; ++row)
    for (unsigned k = 0; k < K; ++k)
      a[row * K + k] = row + 1;
  for (unsigned col = 0; col < N; ++col)
    for (unsigned k = 0; k < K; ++k)
      b[col * K + k] = col + 1;
  for (unsigned i = 0; i < M * N; ++i)
    output[i] = -1;
  if (runtime_linux_job(bb_cores[1].hart, bb_cores[1].signature, run, NULL))
    abort();
  puts("Stream DMA lifetime/Mxmm PASS");
  return 0;
}
