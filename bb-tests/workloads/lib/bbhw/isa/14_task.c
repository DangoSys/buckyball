#ifndef BB_TASK_ISA_H
#define BB_TASK_ISA_H

#include "isa.h"

static inline uintptr_t bb_task_submit(uintptr_t worker, uintptr_t signature) {
  uintptr_t result;
  asm volatile(".insn r 0x2b, 7, 0, %0, %1, %2"
               : "=r"(result)
               : "r"(worker), "r"(signature)
               : "memory");
  return result;
}

static inline uintptr_t bb_task_status(uintptr_t worker) {
  uintptr_t result;
  asm volatile(".insn r 0x2b, 7, 1, %0, %1, %2"
               : "=r"(result)
               : "r"(worker), "r"((uintptr_t)0)
               : "memory");
  return result;
}

static inline uintptr_t bb_task_worker_count(void) {
  uintptr_t result;
  asm volatile(".insn r 0x2b, 7, 3, %0, %1, %2"
               : "=r"(result)
               : "r"((uintptr_t)0), "r"((uintptr_t)0)
               : "memory");
  return result;
}

static inline uintptr_t bb_task_core_signature(uintptr_t core) {
  uintptr_t result;
  asm volatile(".insn r 0x2b, 7, 4, %0, %1, %2"
               : "=r"(result)
               : "r"(core), "r"((uintptr_t)0)
               : "memory");
  return result;
}

static inline uintptr_t bb_task_write_descriptor(uintptr_t field,
                                                 uintptr_t value) {
  uintptr_t result;
  asm volatile(".insn r 0x2b, 7, 7, %0, %1, %2"
               : "=r"(result)
               : "r"(field), "r"(value)
               : "memory");
  return result;
}

#endif
