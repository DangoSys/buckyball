#define _GNU_SOURCE
#include <ant_stream.h>
#include <bbhw/isa/isa.h>
#include <isa/mxmm.h>
#include <params.h>
#include <runtime.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <topology.h>
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
int main(void) {
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(BB_MAIN_CORES, &affinity);
  if (sched_setaffinity(0, sizeof(affinity), &affinity))
    abort();
  runtime_init(1024 * 1024);
  void *workspace = aligned_alloc(64, 1024 * 1024);
  if (!workspace)
    abort();
  workspace_init(workspace, 1024 * 1024);
  workspace_begin(workspace, 1024 * 1024);
  for (unsigned row = 0; row < M; ++row)
    for (unsigned k = 0; k < K; ++k)
      a[row * K + k] = row + 1;
  for (unsigned col = 0; col < N; ++col)
    for (unsigned k = 0; k < K; ++k)
      b[col * K + k] = col + 1;
  for (unsigned i = 0; i < M * N; ++i)
    output[i] = -1;
  task_run(CORE_SIGNATURE, job, NULL);
  puts("Ant stream DMA lifetime/Mxmm PASS");
  free(workspace);
  return 0;
}
