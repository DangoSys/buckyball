#ifndef TESTUTILS_H
#define TESTUTILS_H

#include <stdint.h>

// profile tool
static inline uint64_t read_cycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

#endif // TESTUTILS_H
