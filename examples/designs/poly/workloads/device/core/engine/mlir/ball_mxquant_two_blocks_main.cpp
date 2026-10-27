#include "mlir.h"
#include "reference.h"
#include <thread.h>

extern const unsigned char thread_image[], thread_image_end[];

// Input: two blocks of 32 FP32. Output: the 64 codes, the two scales from
// byte 64, then guard bytes up to the end of row 4. Block 0 holds signed
// zeros, subnormals, rounding ties and the largest finite; block 1 one large
// element that sets the scale and small ones that round.
struct buffers {
  uint32_t input[16 * 4];
  uint8_t output[5 * 16];
};
struct all {
  buffers core[POLY_THREAD_CORES];
};
static constexpr all prepare() {
  all data{};
  for (unsigned core = 0; core < POLY_THREAD_CORES; ++core) {
    buffers &buffer = data.core[core];
    const uint32_t edges[] = {0,          0x80000000, 1,          0x80000001,
                              0x007fffff, 0x00800000, 0x3f880000, 0x3f980000,
                              0x43e00000, 0x7f7fffff, 0xff7fffff};
    for (unsigned i = 0; i < 32; ++i) {
      buffer.input[i] = edges[i % 11];
      buffer.input[32 + i] =
          i ? (i & 1 ? 0x3f880000u : 0xbf980000u) : 0x43e00000u;
    }
    for (unsigned i = 0; i < 5 * 16; ++i)
      buffer.output[i] = 0x5a;
  }
  return data;
}
alignas(64) static all data = prepare();
static unsigned done[POLY_THREAD_CORES];

static void run(unsigned core, const void *args, size_t size) {
  const uint32_t context = thread_find(THREAD_SIGNATURE, 0);
  size_t bytes = thread_image_end - thread_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  thread_tss_write(0, thread_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  thread_tss_write(0, args, size);
  thread_load(context, THREAD_TLS, 0, 0, (size + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE),
           capacity = thread_query(context, THREAD_FIELD_TLS_BYTES);
  struct thread_task task = {core + 1,        0, bytes, base, base + capacity,
                             THREAD_SIGNATURE};
  thread_acquire();
  thread_start(context, &task);
  int cancelled;
  if (thread_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
}

static void exercise(unsigned core) {
  buffers &buffer = data.core[core];
  struct mlir_mxquant_args args = {(uintptr_t)buffer.input,
                                   (uintptr_t)buffer.output};
  run(core, &args, sizeof(args));
  uint8_t expected[2][33];
  mxquant_reference32(buffer.input, expected[0]);
  mxquant_reference32(buffer.input + 32, expected[1]);
  for (unsigned block = 0; block < 2; ++block) {
    for (unsigned i = 0; i < 32; ++i)
      if (buffer.output[block * 32 + i] != expected[block][i])
        __builtin_trap();
    if (buffer.output[64 + block] != expected[block][32])
      __builtin_trap();
  }
  for (unsigned i = 66; i < 80; ++i)
    if (buffer.output[i] != 0x5a)
      __builtin_trap();
  __atomic_store_n(&done[core], 1, __ATOMIC_RELEASE);
}

int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == WORKLOAD_REPORT_HART) {
    for (unsigned core = 0; core < POLY_THREAD_CORES; ++core)
      while (!__atomic_load_n(&done[core], __ATOMIC_ACQUIRE)) {
      }
  } else {
    exercise(hart - 1);
  }
  return 0;
}
