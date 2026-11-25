#include "mlir.h"
#include <thread.h>

extern const unsigned char thread_image[], thread_image_end[];

/* Three rows of K = 544 FP32 quantized through the bank operation of the
 * compiler, every block of its own power-of-two scale. Checked against the
 * definition (as in the mxint8quantize C test quant_random): the block
 * exponent from the largest magnitude, each code within half a step of its
 * value, the clamp at 127, the zero padding. */
#define ROWS 3
#define K 544
#define STRIDE 576

static float input[POLY_THREAD_HARTS][ROWS * K] __attribute__((aligned(64)));
static uint8_t output[POLY_THREAD_HARTS][ROWS * STRIDE]
    __attribute__((section(".data"), aligned(64))) = {{0}};
static unsigned done[POLY_THREAD_HARTS];

/* 2^n, -126 <= n <= 127. */
static float power(int n) {
  uint32_t bits = (uint32_t)(n + 127) << 23;
  float value;
  __builtin_memcpy(&value, &bits, 4);
  return value;
}

static void check(const float *in, const uint8_t *out) {
  for (int row = 0; row < ROWS; ++row) {
    const uint8_t *packed = out + row * STRIDE;
    for (int block = 0; block < K / 32; ++block) {
      const float *x = in + row * K + block * 32;
      uint32_t largest = 0;
      for (int i = 0; i < 32; ++i) {
        uint32_t bits;
        __builtin_memcpy(&bits, &x[i], 4);
        if ((bits & 0x7fffffff) > largest)
          largest = bits & 0x7fffffff;
      }
      int biased = (int)(largest >> 23);
      int e = largest == 0 ? 0 : biased == 0 ? -127 : biased - 133;
      if (e < -127)
        e = -127;
      if (packed[K + block] != (uint8_t)(e + 127))
        __builtin_trap();
      for (int i = 0; i < 32; ++i) {
        float scaled = x[i] * power(-e);
        float target = scaled > 127.0f    ? 127.0f
                       : scaled < -127.0f ? -127.0f
                                          : scaled;
        float error = (float)(int8_t)packed[block * 32 + i] - target;
        if (error > 0.5f || error < -0.5f)
          __builtin_trap();
      }
    }
    for (int i = K + K / 32; i < STRIDE; ++i)
      if (packed[i])
        __builtin_trap();
  }
}

static void exercise(unsigned hart) {
  const uint32_t context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = thread_image_end - thread_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  uint32_t state = 2024;
  for (int i = 0; i < ROWS * K; ++i) {
    state = state * 1664525u + 1013904223u;
    float unit = (float)(state >> 8) / (float)(1u << 23) - 1.0f;
    input[hart][i] = unit * power((int)((i / 32) * 7 % 61) - 30);
  }
  struct mlir_mxquant_args args = {(uintptr_t)input[hart],
                                   (uintptr_t)output[hart]};
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
  check(input[hart], output[hart]);
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
