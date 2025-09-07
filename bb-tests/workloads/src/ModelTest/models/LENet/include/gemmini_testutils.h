#ifndef GEMMINI_TESTUTILS_H
#define GEMMINI_TESTUTILS_H

#include <stdint.h>

// gemmini profile tool
static inline uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

#endif // GEMMINI_TESTUTILS_H
