// A hardware-thread slice with BEMU capture and host-side RTL verification.
// The GVN host checks the DMA readback; guest exit alone is not a data check.
extern "C" {
#include <runtime.h>
}
#include <stdint.h>
#define BUCKYBALL_STREAM
#include <isa.h>
extern "C" {
void gvn_entry(void);
alignas(64) unsigned char gvn_in[] = {
#embed "gvn_in.bin"
};
alignas(64) unsigned char gvn_table[] __attribute__((section(".data"))) = {
#embed "gvn_table.bin"
};
alignas(64) unsigned char gvn_out[] = {
#embed "gvn_out.bin"
};
}
alignas(64) static const unsigned char expected[] = {
#embed "gvn_expect.bin"
};
// Four 64-bit words: the device hart the slice runs on, its thread, the
// thread's verify bank (its last private one) and the most bytes one mvin
// moves on it.
alignas(8) static const unsigned char gvn_thread[] = {
#embed "gvn_thread.bin"
};
static_assert(sizeof(gvn_thread) == 32);
static uint64_t word(int index) {
  uint64_t value;
  __builtin_memcpy(&value, gvn_thread + 8 * index, 8);
  return value;
}
#define X(n) DEVICE_HARTS_##n##_HART_ID,
static const uint64_t hart_ids[] = {DEVICE_HARTS};
#undef X
#define X(n) DEVICE_HARTS_##n##_SIGNATURE,
static const uint64_t hart_signatures[] = {DEVICE_HARTS};
#undef X
extern "C" {
alignas(8) unsigned char gvn_ranges[] = {
#embed "gvn_ranges.bin"
};
volatile uint64_t gvn_range_count = sizeof(gvn_ranges) / 16;
}
static_assert(sizeof(gvn_ranges) % 16 == 0);
__attribute__((used, section(".gvn.bank_output"),
               aligned(8))) static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check = {gvn_out, expected, sizeof(expected)};
extern "C" void _mlir_ciface_buddyTraceCycleStartPath(int64_t, int64_t, int64_t,
                                                      int64_t, int64_t,
                                                      int64_t);
extern "C" void _mlir_ciface_buddyTraceCycleEndPath(int64_t, int64_t, int64_t,
                                                    int64_t, int64_t, int64_t);
// The hart's worker runs on a stack of its own: the boot stack of a hart is
// 4 KiB, too small for compiled graphs.
static uint8_t stack[1 << 20] __attribute__((aligned(16))) = {0xa5};
static int done;
static void readback(void *) {
  uint64_t bank = word(2), dma_bytes = word(3);
  bb_mem_alloc(bank, 1, 1);
  for (uint64_t range = 0; range < gvn_range_count; ++range) {
    uint64_t offset, length;
    __builtin_memcpy(&offset, gvn_ranges + range * 16, 8);
    __builtin_memcpy(&length, gvn_ranges + range * 16 + 8, 8);
    // The whole blocks as one recorded loop, so that a large output fits the
    // stream of the job; then the rest.
    uint64_t blocks = length / dma_bytes;
    if (blocks) {
      stream_loop(blocks);
      stream_offset(0, 1, dma_bytes);
      bb_mvin(uintptr_t(gvn_out + offset), bank, dma_bytes / 16, 1);
      stream_end();
    }
    if (length % dma_bytes)
      bb_mvin(uintptr_t(gvn_out + offset + blocks * dma_bytes), bank,
              (length % dma_bytes + 15) / 16, 1);
  }
  bb_mem_release(bank);
}
static int job(void *, void *) {
  gvn_entry();
  _mlir_ciface_buddyTraceCycleEndPath(INT64_MAX, 1, INT64_MAX, 0, 0, 0);
  stream_begin(1, 0, 0);
  uint64_t thread = word(1);
  stream_job(kernel_current()->slots[thread].signature, thread, readback,
             nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
static void run(void) {
  struct runtime_hart hart =
      runtime_hart(hart_ids[word(0)], hart_signatures[word(0)], 0);
  _mlir_ciface_buddyTraceCycleStartPath(INT64_MAX, 1, INT64_MAX, 0, 0, 0);
  if (hart.execute(hart.context, job, nullptr))
    __builtin_trap();
}
int main() {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == hart_ids[word(0)]) {
    register uintptr_t top asm("t1") = (uintptr_t)(stack + sizeof(stack));
    asm volatile("mv t0, sp\n"
                 "addi sp, t1, -16\n"
                 "sd t0, 0(sp)\n"
                 "jalr %1\n"
                 "ld sp, 0(sp)\n"
                 : "+r"(top)
                 : "r"(run)
                 : "ra", "t0", "t2", "t3", "t4", "t5", "t6", "a0", "a1", "a2",
                   "a3", "a4", "a5", "a6", "a7", "ft0", "ft1", "ft2", "ft3",
                   "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11",
                   "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
                   "memory");
    __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
  }
  if (hart != WORKLOAD_REPORT_HART)
    return 0;
  while (!__atomic_load_n(&done, __ATOMIC_ACQUIRE)) {
  }
  return 0;
}
