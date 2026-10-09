/* RV64 reset trampoline at 0x10000. The board holds the CPU in reset while
 * host software writes and verifies the 128 KiB program BRAM.
 * Explicit zero extension avoids LUI sign extension of 0x80000000 on RV64.
 */
.section .text.init,"ax",@progbits
.globl _start
.option norvc
_start:
  lui t0, 0x80000
  slli t0, t0, 32
  srli t0, t0, 32
  jr t0
