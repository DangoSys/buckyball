#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct cube {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" void _mlir_ciface_compute(matrix *, cube *, matrix *);
constexpr int M = 160, K = 256, N = 48, KQ = K + K / 32;
alignas(64) static float input[M * K];
alignas(64) static int8_t weights[N * KQ];
alignas(64) static float expected[M * N];
alignas(64) static float output[M * N] = {17};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {output, expected,
                                                              sizeof(output)};
static void readback(void *) {
  // One bank holds 1024 rows of 16 bytes.
  bb_mem_alloc(31, 1, 1);
  for (unsigned row = 0; row < sizeof(output) / 16; row += 1024)
    bb_mvin((uintptr_t)output + row * 16, 31,
            sizeof(output) / 16 - row < 1024 ? sizeof(output) / 16 - row : 1024,
            1);
  bb_mem_release(31);
}
int main() {
  // Small integers: x*x+x stays below 128, so its MXINT8 quantization is exact.
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      input[m * K + k] = (m * 3 + k) % 11 - 3;
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < KQ; ++k)
      weights[n * KQ + k] = k < K ? (n + k * 2) % 5 - 2 : 127;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        float x = input[m * K + k];
        sum += (x * x + x) * weights[n * KQ + k];
      }
      expected[m * N + n] = sum;
    }
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  matrix x{input, input, 0, {M, K}, {K, 1}};
  cube w{weights, weights, 0, {1, N, KQ}, {N * KQ, KQ, 1}};
  matrix y{output, output, 0, {M, N}, {N, 1}};
  _mlir_ciface_compute(&x, &w, &y);
  for (int i = 0; i < M * N; i += 517)
    if (output[i] != expected[i])
      return 1;
  stream_begin(1, 0, 0);
  stream_job(owner.slots[0].signature, 0, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
