#include "isa.h"

// =========================== for CTest ===========================
// FENCE指令没有参数，直接定义汇编宏

// FENCE指令低级实现
#ifndef __x86_64__
#define FENCE_RAW()                                                            \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 31, x0, x0, x0"                \
               :                                                               \
               :                                                               \
               : "memor"                                                       \
                 "y")
#else
#define FENCE_RAW() /* x86平台下不执行RISC-V指令 */
#endif

// FENCE指令高级API实现
void bb_fence(void) { FENCE_RAW(); }
