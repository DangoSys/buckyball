#include "ant_mesh.h"
#include <instances.h>
#include <params.h>
#include <thread.h>

enum {
  BANKS = VIRTUAL_BANK_NUM - BB_SHARED_BANK_BASE,
  ROW_BYTES = BANK_WIDTH / 8
};
_Static_assert(BANKS >= 2 && BANK_WIDTH == 128,
               "Mesh tests require two 128-bit input rows");
static uint8_t input[BB_THREAD_CORE_COUNT][BANKS * ROW_BYTES]
    __attribute__((aligned(64)));
static uint8_t output[BB_THREAD_CORE_COUNT]
                     [5 * 2 * ROW_BYTES + BANKS * ROW_BYTES]
    __attribute__((aligned(64)));
static unsigned done[BB_THREAD_CORE_COUNT];
extern const unsigned char ant_image[], ant_image_end[];
static void exercise(unsigned core) {
  if (thread_query(0, THREAD_FIELD_COUNT) != 5)
    __builtin_trap();
  for (unsigned i = 0; i < sizeof(input[core]); ++i)
    input[core][i] = POLY_MESH_PRIVATE
                         ? (i < ROW_BYTES ? 0x30 + i : 0xa0 + i - ROW_BYTES)
                         : (i / ROW_BYTES * 17 + i % ROW_BYTES);
  unsigned count = POLY_MESH_PRIVATE ? 5 : 1;
  size_t bytes = ant_image_end - ant_image;
  struct thread_task tasks[5];
  // Both Attention and FFN execute the same common bank-ISA payload.
  for (unsigned context = 0; context < count; ++context) {
    uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
    if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES))
      __builtin_trap();
    struct ant_mesh_args args = {(uintptr_t)input[core],
                                 (uintptr_t)output[core], context};
    thread_tss_write(0, ant_image, bytes);
    thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
    thread_tss_write(0, &args, sizeof(args));
    thread_load(context, THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
    tasks[context] = (struct thread_task){
        context + 1,
        0,
        bytes,
        base,
        base + thread_query(context, THREAD_FIELD_TLS_BYTES),
        thread_query(context, THREAD_FIELD_SIGNATURE)};
  }
  thread_acquire();
  for (unsigned context = 0; context < count; ++context)
    thread_start(context, &tasks[context]);
  for (unsigned context = 0; context < count; ++context) {
    int cancelled;
    if (thread_wait(context, tasks[context].id, &cancelled) || cancelled)
      __builtin_trap();
  }
  thread_release();
  if (POLY_MESH_PRIVATE) {
    for (unsigned target = 0; target < 5; ++target)
      for (unsigned byte = 0; byte < ROW_BYTES; ++byte)
        if (output[core][target * 2 * ROW_BYTES + ROW_BYTES + byte] !=
            input[core][ROW_BYTES + byte])
          __builtin_trap();
  } else {
    for (unsigned i = 0; i < sizeof(input[core]); ++i)
      if (output[core][i] != input[core][i])
        __builtin_trap();
  }
  __atomic_store_n(&done[core], 1, __ATOMIC_RELEASE);
}
int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart == BB_REPORT_HART) {
    for (unsigned core = 0; core < BB_THREAD_CORE_COUNT; ++core)
      while (!__atomic_load_n(&done[core], __ATOMIC_ACQUIRE)) {
      }
  } else {
    exercise(hart - 1);
  }
  return 0;
}
