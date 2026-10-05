.option norelax
.option norvc
.section .text.start,"ax",@progbits
.globl _start
_start:
  li s0, 0x10000000
  la s1, data
  li s2, 0
  li s3, 0
  la t0, trap_handler
  csrw mtvec, t0
  li s7, 7
  li t0, 0x80
  csrw mie, t0
  li t0, 8
  csrs mstatus, t0
  j tests

.section .text.trap,"ax",@progbits
trap_handler:
  // t4/t5/t6 are reserved to this handler, so interrupted operands survive.
  csrr t4, mcause
  li t5, 0x8000000000000000
  or t5, t5, s7
  bne t4, t5, fail
  csrr t4, mtval
  bnez t4, fail
  li t5, 1
  sll t5, t5, s7
  csrr t6, mip
  and t6, t6, t5
  beqz t6, fail
  ld t4, 64(s1)
  addi t4, t4, 1
  sd t4, 64(s1)
  sd t5, 16(s0)
  addi s3, s3, 1
  mret

.section .text.tests,"ax",@progbits
tests:
irq_load:
  ld s8, 0(s1)
  li t0, 0x8877665544332211
  bne s8, t0, fail
timer_wait:
  li t0, 1
  bne s3, t0, timer_wait
  li s2, 1
  sd s2, 8(s0)
  li s7, 3
  li t0, 8
  csrw mie, t0
  addi t2, s1, 8
  li t0, 1
irq_amo:
  amoadd.d.aqrl s8, t0, (t2)
  bnez s8, fail
  ld t1, 8(s1)
  bne t1, t0, fail
software_wait:
  li t0, 2
  bne s3, t0, software_wait
  li s2, 2
  sd s2, 8(s0)
  li s7, 11
  li t0, 0x800
  csrw mie, t0
  li t0, 0x55
irq_store:
  sd t0, 16(s1)
external_wait:
  li t0, 3
  bne s3, t0, external_wait
  ld t0, 16(s1)
  li t1, 0x55
  bne t0, t1, fail
  ld t0, 64(s1)
  li t1, 3
  bne t0, t1, fail
  ld t0, 8(s1)
  li t1, 1
  bne t0, t1, fail
  li s2, 3
  sd s2, 8(s0)
  sd zero, 0(s0)
1: j 1b
fail:
  ori t0, s2, 0x100
  sd t0, 0(s0)
2: j 2b

.section .data,"aw",@progbits
data:
  .dword 0x8877665544332211
  .dword 0, 0, 0, 0, 0, 0, 0
  .dword 0
