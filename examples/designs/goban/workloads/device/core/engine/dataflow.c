#include "copy.h"
#include <engine.h>

static uint64_t input[16] __attribute__((aligned(64)));
static uint64_t intermediate[5][16] __attribute__((aligned(64)));
static uint64_t output[2][16] __attribute__((aligned(64)));
static uint64_t root[512] __attribute__((aligned(4096)));

int main(void) {
  struct engine engine;
  struct engine_slot slots[4];
  engine_init(&engine, slots, 4);
  if (engine.thread_count != 4)
    return 1;
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    for (unsigned word = 0; word < 16; ++word) {
      input[word] = ((uint64_t)(iteration + 1) << 32) | (word + 1);
      output[0][word] = output[1][word] = 0;
      for (unsigned buffer = 0; buffer < 5; ++buffer)
        intermediate[buffer][word] = UINT64_C(0xa5a5a5a5a5a5a5a5);
    }
    struct copy_task args[] = {
        {(uintptr_t)input, (uintptr_t)intermediate[0], 8},
        {(uintptr_t)intermediate[1], (uintptr_t)output[0], 8},
        {(uintptr_t)intermediate[2], (uintptr_t)intermediate[3], 8},
        {(uintptr_t)intermediate[4], (uintptr_t)output[1], 8}};
    if (iteration != 0) {
      // Logical edge addresses deliberately lie outside the device's DDR
      // region.
      uintptr_t logical = UINT64_C(0x1000000000) + iteration * 0x10000;
      args[0].output = logical;
      args[1].input = logical + 0x1000;
      args[2].input = logical + 0x2000;
      args[2].output = logical + 0x3000;
      args[3].input = logical + 0x4000;
    }
    struct engine_job jobs[4];
    struct engine_result results[4];
    for (unsigned thread = 0; thread < 4; ++thread)
      jobs[thread] = (struct engine_job){
          .thread = thread,
          .task = {.id = thread + 1, .signature = slots[thread].signature},
          .opaque = &args[thread]};
    struct engine_transfer edges[] = {
        {0, 1, args[0].output, args[1].input, sizeof(input), 0},
        {0, 2, args[0].output, args[2].input, sizeof(input), 0},
        {2, 3, args[2].output, args[3].input, sizeof(input), 0}};
    if (iteration == 2) {
      for (unsigned page = 0; page < 4; ++page)
        root[page] = ((uint64_t)page << 28) | 0xcf;
      root[4] = (UINT64_C(0x80000000) >> 2) | 0xcf;
      args[0].input += UINT64_C(0x80000000);
      args[1].output += UINT64_C(0x80000000);
      args[3].output += UINT64_C(0x80000000);
      uintptr_t all = ~(uintptr_t)0;
      uintptr_t config = 0x1f;
      uintptr_t satp = (UINT64_C(8) << 60) | ((uintptr_t)root >> 12);
      __asm__ volatile("csrw pmpaddr0, %0; csrw pmpcfg0, %1; fence rw, rw; "
                       "csrw satp, %2; sfence.vma" ::"r"(all),
                       "r"(config), "r"(satp)
                       : "memory");
      uintptr_t status;
      __asm__ volatile("csrr %0, mstatus" : "=r"(status));
      status = (status & ~(UINT64_C(3) << 11)) | (UINT64_C(1) << 17) |
               (UINT64_C(1) << 11);
      __asm__ volatile("csrw mstatus, %0" ::"r"(status) : "memory");
    }
    int result = engine_run(&engine, jobs, 4, edges, 3, results);
    if (iteration == 2) {
      uintptr_t mprv = UINT64_C(1) << 17;
      __asm__ volatile(
          "csrc mstatus, %0; csrw satp, zero; sfence.vma" ::"r"(mprv)
          : "memory");
    }
    if (result)
      return 2;
    for (unsigned thread = 0; thread < 4; ++thread)
      if (results[thread].state != ENGINE_SUCCEEDED || results[thread].value)
        return 3;
    for (unsigned word = 0; word < 16; ++word) {
      if (output[0][word] != input[word] || output[1][word] != input[word])
        return 4;
      for (unsigned buffer = 0; buffer < 5; ++buffer)
        if (intermediate[buffer][word] != UINT64_C(0xa5a5a5a5a5a5a5a5))
          return 5;
    }
  }
  return 0;
}
