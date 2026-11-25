#include <bb_harts.h>
#include <bbhw/isa/isa.h>
#include <isa/mxint8matmul.h>
#include <linux.h>
#include <stdio.h>
#include <stdlib.h>
#include <stream.h>
#include <workspace.h>
enum { M = 16, N = 16, K = 32, ROW = MXINT8_ROWS(K) * 16 };
/* Packed MXINT8 rows: K codes, K/32 exponent bytes, zero padding. */
static uint8_t a[M * ROW] __attribute__((aligned(4096)));
static uint8_t b[N * ROW] __attribute__((aligned(4096)));
static float output[M * N] __attribute__((aligned(4096)));
static void job(void *argument) {
  (void)argument;
  for (unsigned bank = 3; bank <= 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)a, 3, sizeof(a) / 16, 1);
  // A DMA call has completed before the CPU is allowed to reuse its source.
  for (unsigned i = 0; i < sizeof(a); ++i)
    a[i] = 0;
  bb_mvin((uintptr_t)b, 4, sizeof(b) / 16, 1);
  bb_mxint8mm(3, 4, 5, M, N, K, 0, 0, 0, 0);
  bb_mvout((uintptr_t)output, 5, sizeof(output) / 16, 1);
  for (unsigned row = 0; row < M; ++row)
    for (unsigned col = 0; col < N; ++col)
      if (output[row * N + col] != (float)(K * (1u << (row % 4 + col % 4))))
        abort();
  for (unsigned bank = 3; bank <= 5; ++bank)
    bb_mem_release(bank);
}
// Runs on the hart; the stream executes its Ball instructions on a hardware
// thread.
static int run(void *kernel, void *argument) {
  (void)kernel;
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
  // Codes 2^(row % 4) and 2^(col % 4) at scale 2^0 (exponent byte 127):
  // C[row][col] = K * 2^(row % 4 + col % 4).
  for (unsigned row = 0; row < M; ++row) {
    for (unsigned k = 0; k < K; ++k)
      a[row * ROW + k] = 1u << (row % 4);
    a[row * ROW + K] = 127;
  }
  for (unsigned col = 0; col < N; ++col) {
    for (unsigned k = 0; k < K; ++k)
      b[col * ROW + k] = 1u << (col % 4);
    b[col * ROW + K] = 127;
  }
  for (unsigned i = 0; i < M * N; ++i)
    output[i] = -1;
  if (runtime_linux_job(bb_harts[1].hartid, bb_harts[1].signature, run, NULL))
    abort();
  puts("Stream DMA lifetime/MXINT8 matmul PASS");
  return 0;
}
