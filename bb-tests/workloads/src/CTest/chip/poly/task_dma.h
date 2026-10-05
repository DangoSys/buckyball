#ifndef TASK_DMA_H
#define TASK_DMA_H
#include <bbhw/isa/isa.h>
#include <multicore.h>
#include <params.h>
#include <stddef.h>
#include <stdint.h>
#include <topology.h>
// Compute tile 1: its controller submits tasks to its hidden workers.
enum {
  TASK_TILE = 1,
  TASK_WORKERS = BB_COMPUTE_CORES - 1,
  DMA_BYTES = 64,
  DMA_ROWS = DMA_BYTES / (BANK_WIDTH / 8)
};
struct dma_argument {
  uintptr_t input, output, result;
  uint32_t seed;
};
extern void buckyball_task_worker(void) __attribute__((noreturn));
extern char __tls_start[], __tdata_end[], __tls_end[];
static inline void task_tls(uint8_t *dest) {
  size_t initialized = (uintptr_t)__tdata_end - (uintptr_t)__tls_start;
  size_t total = (uintptr_t)__tls_end - (uintptr_t)__tls_start;
  for (size_t i = 0; i < total; i++)
    dest[i] = i < initialized ? ((volatile uint8_t *)__tls_start)[i] : 0;
}
// Workers may be different cores, so a task carries its worker's own signature.
static inline int submit_task(unsigned worker, void (*kernel)(void *),
                              void *arg, void *stack, void *tls) {
  uintptr_t signature = bb_task_core_signature(worker);
  uintptr_t gp;
  asm volatile("mv %0, gp" : "=r"(gp));
  uintptr_t descriptor[] = {(uintptr_t)kernel,
                            (uintptr_t)arg,
                            (uintptr_t)stack,
                            (uintptr_t)tls,
                            0,
                            gp,
                            signature};
  for (unsigned field = 0; field < 7; field++)
    if (bb_task_write_descriptor(field, descriptor[field]))
      return 1;
  return bb_task_submit(worker, signature) != 0;
}
static inline void task_finish(uintptr_t status) __attribute__((noreturn));
static inline void task_finish(uintptr_t status) {
  register uintptr_t result asm("a0") = status;
  register uintptr_t operation asm("a7") = 0xbb01;
  asm volatile("ecall" : "+r"(result) : "r"(operation) : "memory");
  __builtin_unreachable();
}
#endif
