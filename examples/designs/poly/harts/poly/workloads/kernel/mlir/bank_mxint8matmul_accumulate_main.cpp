#include "mlir.h"
#include <array>
#include <thread.h>

extern const unsigned char thread_image[], thread_image_end[];

/* C[5,12] += A[5,96] . B[12,96]^T through the bank operation of the
 * compiler, onto an old C of integers. Codes in [-15, 15] and block exponents
 * within 2 of 1 make every block term and every sum exact in FP32, so C is
 * known bit for bit (as in the mxint8matmul C test mm_accumulate). */
#define M 5
#define N 12
#define K 96
#define ROW 112

alignas(64) static constexpr auto a = [] {
  std::array<uint8_t, M * ROW> data{};
  for (int m = 0; m < M; ++m)
    for (int i = 0; i < K; ++i) {
      data[m * ROW + i] = (uint8_t)((m * 7 + i * 3) % 31 - 15);
      data[m * ROW + K + i / 32] = 127 + (m + i / 32) % 2;
    }
  return data;
}();
alignas(64) static constexpr auto b = [] {
  std::array<uint8_t, N * ROW> data{};
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < K; ++i) {
      data[n * ROW + i] = (uint8_t)((n * 5 + i * 11) % 29 - 14);
      data[n * ROW + K + i / 32] = 127 - (n + 2 * (i / 32)) % 3;
    }
  return data;
}();
alignas(64) static auto c = [] {
  std::array<std::array<float, M * N>, POLY_THREAD_HARTS> data{};
  for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
    for (int i = 0; i < M * N; ++i)
      data[hart][i] = (float)((i / N) * 13 - (i % N) * 7);
  return data;
}();
static unsigned done[POLY_THREAD_HARTS];

static const float expected[M * N] = {
    -162.50f, 615.50f,  567.75f,  376.50f,   -259.75f, -462.50f, -56.00f,
    -221.50f, 566.25f,  280.00f,  -31.00f,   -580.00f, 764.50f,  -349.50f,
    293.50f,  5.25f,    -87.50f,  -1103.00f, 855.50f,  145.50f,  -194.00f,
    -795.50f, -404.50f, -546.50f, 544.00f,   495.25f,  207.25f,  -454.50f,
    -339.00f, -89.50f,  765.50f,  1052.50f,  -451.50f, -436.00f, -840.25f,
    -190.00f, 639.75f,  621.00f,  -971.50f,  251.25f,  -404.50f, -302.50f,
    355.75f,  585.50f,  62.00f,   199.25f,   -106.50f, -472.50f, 227.50f,
    1119.00f, 319.50f,  -76.50f,  -271.00f,  -398.50f, -366.00f, 528.50f,
    514.75f,  26.00f,   1.25f,    -319.25f};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    c[0].data(), expected, sizeof(expected)};

static void exercise(unsigned hart) {
  const uint32_t context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = thread_image_end - thread_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  struct mlir_mxmm_args args = {(uintptr_t)a.data(), (uintptr_t)b.data(),
                                (uintptr_t)c[hart].data()};
  thread_tss_write(0, thread_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  thread_tss_write(0, &args, sizeof(args));
  thread_load(context, THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE),
           capacity = thread_query(context, THREAD_FIELD_TLS_BYTES);
  struct thread_task task = {hart + 1,        0, bytes, base, base + capacity,
                             THREAD_SIGNATURE};
  thread_acquire();
  thread_start(context, &task);
  int cancelled;
  if (thread_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
  for (int i = 0; i < M * N; ++i)
    if (c[hart][i] != expected[i])
      __builtin_trap();
  __atomic_store_n(&done[hart], 1, __ATOMIC_RELEASE);
}

int main(void) {
  uintptr_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid == WORKLOAD_REPORT_HART) {
    for (unsigned hart = 0; hart < POLY_THREAD_HARTS; ++hart)
      while (!__atomic_load_n(&done[hart], __ATOMIC_ACQUIRE)) {
      }
  } else {
    exercise(hartid - 1);
  }
  return 0;
}
