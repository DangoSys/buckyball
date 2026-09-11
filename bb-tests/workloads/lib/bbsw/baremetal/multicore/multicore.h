#ifndef BBSW_BAREMETAL_MULTICORE_H
#define BBSW_BAREMETAL_MULTICORE_H

#include <stdint.h>
#include <topology.h>

static inline uint32_t bb_get_hart_id(void) {
  uint32_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  return hart;
}

static inline core_id_t bb_get_core_id(void) {
  return bb_topology_core_id(bb_get_hart_id());
}

#endif
