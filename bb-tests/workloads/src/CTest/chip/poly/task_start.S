#include <topology.h>
/* Compute tile 1 runs the test: its controller (hart BB_MAIN_CORES) clears BSS, runs main
 * and reports; its workers (from hart BB_MAIN_CORES + BB_COMPUTE_TILES) enter main to serve
 * tasks. Every other hart parks. */
#define TASK_CONTROLLER BB_MAIN_CORES
#define TASK_WORKER0 (BB_MAIN_CORES + BB_COMPUTE_TILES)
.section .text.init,"ax",@progbits
.globl _start
_start:
  csrr s0, mhartid
  li t0, TASK_CONTROLLER
  beq s0, t0, .Lsetup
  li t0, TASK_WORKER0
  bltu s0, t0, .Lpark
  li t0, TASK_WORKER0 + BB_COMPUTE_CORES - 1
  bgeu s0, t0, .Lpark
.Lsetup:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  li sp, 0x90000000
  slli t0, s0, 12
  sub sp, sp, t0
  la t0, __tls_start
  la t1, __tdata_end
  la t2, __tls_end
  sub t3, t2, t0
  addi t3, t3, 15
  andi t3, t3, -16
  sub sp, sp, t3
  mv tp, sp
  mv t4, tp
.Lcopy_tls:
  bgeu t0, t1, .Lzero_tls
  lbu t5, 0(t0)
  sb t5, 0(t4)
  addi t0, t0, 1
  addi t4, t4, 1
  j .Lcopy_tls
.Lzero_tls:
  bgeu t0, t2, .Lcsr
  sb zero, 0(t4)
  addi t0, t0, 1
  addi t4, t4, 1
  j .Lzero_tls
.Lcsr:
  la t0, .Ltrap
  csrw mtvec, t0
  csrw mie, zero
  li t0, 0x1a000  /* FS Initial and XS Dirty before any controller/task custom instruction. */
  csrs mstatus, t0
  csrw fcsr, zero
  li t0, TASK_CONTROLLER
  bne s0, t0, .Lmain
  la t0, __bss_start
  la t1, __bss_end
.Lclear:
  bgeu t0, t1, .Lmain
  sd zero, 0(t0)
  addi t0, t0, 8
  j .Lclear
.Lmain:
  call main
  li t0, TASK_CONTROLLER
  bne s0, t0, .Lpark
  li t0, 0x60000000
  sw a0, 0(t0)
.Lpark:
  wfi
  j .Lpark
.align 4
.Ltrap:
  csrr t0, mcause
  ori t0, t0, 0x100
  li t1, 0x60000000
  sw t0, 0(t1)
  j .Lpark
