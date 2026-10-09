.section .text.init,"ax",@progbits
.globl _start
.option norvc
_start:
  la sp, __stack_top
  la t0, trap
  csrw mtvec, t0
  la t0, __bss_start
  la t1, __bss_end
1:
  bgeu t0, t1, 2f
  sd zero, 0(t0)
  addi t0, t0, 8
  j 1b
2:
  call main
3:
  wfi
  j 3b
.balign 4
trap:
  csrr a0, mcause
  csrr a1, mepc
  csrr a2, mtval
  call trap_report
  j 3b
