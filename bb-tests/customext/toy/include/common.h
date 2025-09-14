#ifndef _COMMON_H
#define _COMMON_H

#include <cassert>
#include <cstdio>
#include <riscv/mmu.h>
#include <riscv/trap.h>

using namespace std;

// Forward declaration
class toy_t;

#define dprintf(p, ...)                                                        \
  {                                                                            \
    if (p->get_log_commits_enabled())                                          \
      printf(__VA_ARGS__);                                                     \
  }

// Template functions for DRAM access
template <class T> T read_from_dram(processor_t *p, reg_t addr) {
  T value = 0;
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    value |= p->get_mmu()->load<uint8_t>(addr + byte_idx) << (byte_idx * 8);
  }
  return value;
}

template <class T> void write_to_dram(processor_t *p, reg_t addr, T data) {
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    p->get_mmu()->store<uint8_t>(addr + byte_idx,
                                 (data >> (byte_idx * 8)) & 0xFF);
  }
}

#endif // _COMMON_H
