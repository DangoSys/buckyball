#ifndef BBSW_BAREMETAL_MULTITHREAD_H
#define BBSW_BAREMETAL_MULTITHREAD_H

#include <bb_cores.h>
#include <stdint.h>

static inline uint32_t bb_get_hart_id(void) {
  uint32_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  return hart;
}

#endif
