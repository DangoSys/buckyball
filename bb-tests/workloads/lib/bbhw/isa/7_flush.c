#include "isa.h"

// =========================== for CTest ===========================
// FLUSH instruction low-level implementation
#ifndef __x86_64__
#define FLUSH_RAW()                                                            \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 7, x0, x0, x0" ::: "memory")
#else
// Do not execute RISC-V instructions on x86 platform
#define FLUSH_RAW()
#endif

// FLUSH instruction high-level API implementation
void bb_flush(void) { FLUSH_RAW(); }
