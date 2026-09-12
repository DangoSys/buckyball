#ifndef BBSW_LINUX_MULTICORE_H
#define BBSW_LINUX_MULTICORE_H

#include <stdint.h>
#include <sys/syscall.h>
#include <topology.h>
#include <unistd.h>

static inline uint32_t bb_get_hart_id(void) {
  unsigned cpu;
  if (syscall(SYS_getcpu, &cpu, 0, 0) != 0)
    __builtin_trap();
  return cpu;
}

static inline core_id_t bb_get_core_id(void) {
  return bb_topology_core_id(bb_get_hart_id());
}

#endif
