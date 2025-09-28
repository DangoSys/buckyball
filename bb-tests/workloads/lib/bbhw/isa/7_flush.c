#include "isa.h"

// =========================== for CTest ===========================
// FLUSH指令低级实现
#ifndef __x86_64__
#define FLUSH_RAW()                                                            \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 7, x0, x0, x0" ::: "memory")
#else
#define FLUSH_RAW() /* x86平台下不执行RISC-V指令 */
#endif

// FLUSH指令高级API实现
void bb_flush(void) { FLUSH_RAW(); }
