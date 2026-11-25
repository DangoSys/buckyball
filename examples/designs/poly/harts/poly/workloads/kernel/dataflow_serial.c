#include "copy.h"
#include <bb_harts.h>
#include <kernel.h>

static uint64_t input[BB_NUM_HARTS][16] __attribute__((aligned(64)));
static uint64_t intermediate[BB_NUM_HARTS][8][16] __attribute__((aligned(64)));
static uint64_t output[BB_NUM_HARTS][16] __attribute__((aligned(64)));
static struct kernel_job jobs[BB_NUM_HARTS][5];
static unsigned active, completed[BB_NUM_HARTS], errors[BB_NUM_HARTS];

int main(void) {
  uintptr_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  const unsigned hart = bb_hart_slot(hartid);
  if (hartid == WORKLOAD_REPORT_HART) {
    volatile uint32_t *uart = (volatile uint32_t *)0x60020000;
    for (unsigned peer = 0; peer < BB_NUM_HARTS; ++peer) {
      if (bb_harts[peer].hartid == WORKLOAD_REPORT_HART)
        continue;
      *uart = 'C';
      for (int shift = 28; shift >= 0; shift -= 4)
        *uart = "0123456789abcdef"[(bb_harts[peer].hartid >> shift) & 15];
      *uart = '\n';
      __atomic_store_n(&active, peer + 1, __ATOMIC_RELEASE);
      while (!__atomic_load_n(&completed[peer], __ATOMIC_ACQUIRE)) {
      }
      if (errors[peer])
        return errors[peer];
      *uart = 'P';
      *uart = '\n';
    }
    return 0;
  }
  while (__atomic_load_n(&active, __ATOMIC_ACQUIRE) != hart + 1) {
  }
  unsigned status = 0;
  struct kernel kernel;
  struct kernel_slot slots[5];
  kernel_init(&kernel, slots, 5);
  if (kernel.thread_count != 5) {
    status = 1;
    goto finished;
  }
  for (unsigned word = 0; word < 16; ++word) {
    input[hart][word] = ((uint64_t)hartid << 32) | (word + 1);
    output[hart][word] = 0;
    for (unsigned buffer = 0; buffer < 8; ++buffer)
      intermediate[hart][buffer][word] = UINT64_C(0xa5a5a5a5a5a5a5a5);
  }
  struct copy_task args[5];
  struct kernel_result results[5];
  struct kernel_transfer edges[4];
  for (unsigned thread = 0; thread < 5; ++thread) {
    args[thread] = (struct copy_task){
        (uintptr_t)(thread ? intermediate[hart][2 * thread - 1] : input[hart]),
        (uintptr_t)(thread == 4 ? output[hart]
                                : intermediate[hart][2 * thread]),
        8};
    jobs[hart][thread].thread = thread;
    jobs[hart][thread].task.id = thread + 1;
    jobs[hart][thread].task.signature = slots[thread].signature;
    jobs[hart][thread].opaque = &args[thread];
    if (thread)
      edges[thread - 1] = (struct kernel_transfer){thread - 1,
                                                   thread,
                                                   args[thread - 1].output,
                                                   args[thread].input,
                                                   sizeof(input[hart]),
                                                   0};
  }
  if (kernel_run(&kernel, jobs[hart], 5, edges, 4, results)) {
    status = 2;
    goto finished;
  }
  for (unsigned thread = 0; thread < 5; ++thread)
    if (results[thread].state != KERNEL_SUCCEEDED || results[thread].value) {
      status = 3;
      goto finished;
    }
  for (unsigned word = 0; word < 16; ++word) {
    if (output[hart][word] != input[hart][word]) {
      status = 4;
      goto finished;
    }
    for (unsigned buffer = 0; buffer < 8; ++buffer)
      if (intermediate[hart][buffer][word] != UINT64_C(0xa5a5a5a5a5a5a5a5)) {
        status = 5;
        goto finished;
      }
  }
finished:
  errors[hart] = status;
  __atomic_store_n(&completed[hart], 1, __ATOMIC_RELEASE);
  return status;
}
