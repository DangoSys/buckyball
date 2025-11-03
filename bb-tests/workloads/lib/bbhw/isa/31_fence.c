#include "isa.h"

// =========================== for CTest ===========================
// FENCE instruction has no parameters, define assembly macro directly

// FENCE instruction low-level implementation
#ifndef __x86_64__
#define FENCE_RAW()                                                            \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 31, x0, x0, x0"                \
               :                                                               \
               :                                                               \
               : "memor"                                                       \
                 "y")
#else
// Do not execute RISC-V instructions on x86 platform
#define FENCE_RAW()
#endif

// FENCE instruction high-level API implementation
void bb_fence(void) { FENCE_RAW(); }
