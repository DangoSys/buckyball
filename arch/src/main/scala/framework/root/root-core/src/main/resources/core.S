.option norelax
.section .text.start,"ax",@progbits
.globl _start
_start:
  la t0, trap_handler
  csrw mtvec, t0
  li s0, 0x10000000
  la s1, data
  li s9, -1
  j tests

.section .text.trap,"ax",@progbits
.option push
.option norvc
trap_handler:
  csrr t0, mcause
  bne t0, s9, fail
  csrr t0, mtval
  bne t0, s10, fail
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  li s9, -1
  mret
.option pop

.section .text.tests,"ax",@progbits
tests:
  li s2, 1
  sd s2, 8(s0)
  li t0, 0x8877665544332211
  sd t0, 0(s1)
  ld t1, 0(s1)
  bne t0, t1, fail
  lb t0, 7(s1)
  li t1, -120
  bne t0, t1, fail
  lbu t0, 7(s1)
  li t1, 0x88
  bne t0, t1, fail
  lh t0, 6(s1)
  li t1, -30601
  bne t0, t1, fail
  lhu t0, 6(s1)
  li t1, 0x8877
  bne t0, t1, fail
  lw t0, 4(s1)
  li t1, 0xffffffff88776655
  bne t0, t1, fail
  lwu t0, 4(s1)
  li t1, 0x88776655
  bne t0, t1, fail
  li t0, 0xaa
  sb t0, 1(s1)
  li t0, 0xbeef
  sh t0, 2(s1)
  li t0, 0xcafebabe
  sw t0, 4(s1)
  ld t0, 0(s1)
  li t1, 0xcafebabebeefaa11
  bne t0, t1, fail

  li s2, 2
  sd s2, 8(s0)
  li t0, 0x7fffffff
  sw t0, 0(s1)
  li t0, 1
  amoadd.w t1, t0, (s1)
  li t2, 0x7fffffff
  bne t1, t2, fail
  ld t1, 0(s1)
  li t2, 0xcafebabe80000000
  bne t1, t2, fail
  addi t3, s1, 4
  amoadd.w t1, t0, (t3)
  li t2, 0xffffffffcafebabe
  bne t1, t2, fail
  ld t1, 0(s1)
  li t2, 0xcafebabf80000000
  bne t1, t2, fail
  li t0, 0x1122334455667788
  amoswap.d t1, t0, (s1)
  bne t1, t2, fail
  ld t1, 0(s1)
  bne t1, t0, fail

  li s2, 3
  sd s2, 8(s0)
  li t4, 32
  j lr_word
  .balign 512
lr_word:
  lr.w t0, (s1)
  addi t0, t0, 1
  sc.w t1, t0, (s1)
  beqz t1, word_done
  addi t4, t4, -1
  bnez t4, lr_word
  j fail
word_done:
  ld t0, 0(s1)
  li t1, 0x1122334455667789
  bne t0, t1, fail
  li t4, 32
  j lr_double
  .balign 512
lr_double:
  lr.d t0, (s1)
  addi t0, t0, 1
  sc.d t1, t0, (s1)
  beqz t1, double_done
  addi t4, t4, -1
  bnez t4, lr_double
  j fail
double_done:
  ld t0, 0(s1)
  li t1, 0x112233445566778a
  bne t0, t1, fail

  li s2, 4
  sd s2, 8(s0)
  li t0, 1
  beqz t0, fail
  li t0, 0
  bnez t0, fail
  li t0, 0xdeadbeef
  beq zero, zero, branch_done
  sd t0, 0(s1)
branch_done:
  ld t0, 0(s1)
  li t1, 0x112233445566778a
  bne t0, t1, fail
  jal ra, patch_target
  li t0, 7
  bne a0, t0, fail
  la t0, patch_target
  li t1, 0x00900513
  sw t1, 0(t0)
  fence.i
  jal ra, patch_target
  li t0, 9
  bne a0, t0, fail

  li s2, 5
  sd s2, 8(s0)
.option push
.option norvc
  li s9, 11
  li s10, 0
  ecall
  li t1, -1
  bne s9, t1, fail
  li t3, 0x20000000
  li s9, 5
  mv s10, t3
  ld t0, 0(t3)
  li t1, -1
  bne s9, t1, fail
  li s9, 7
  sd t0, 0(t3)
  li t1, -1
  bne s9, t1, fail
.option pop

  li s2, 6
  sd s2, 8(s0)
  srli t0, s1, 2
  csrw pmpaddr0, t0
  addi t0, t0, 8
  csrw pmpaddr1, t0
  li t0, 0x8800
  csrw pmpcfg0, t0
.option push
.option norvc
  li s9, 5
  mv s10, s1
  ld t0, 0(s1)
  li t1, -1
  bne s9, t1, fail
  li s9, 7
  sd t0, 0(s1)
  li t1, -1
  bne s9, t1, fail
.option pop
  li s2, 7
  sd s2, 8(s0)
  sd zero, 0(s0)
1: j 1b

fail:
  ori t0, s2, 0x100
  sd t0, 0(s0)
2: j 2b

.balign 8
patch_target:
  .word 0x00700513
  ret

.section .data,"aw",@progbits
.balign 8
data:
  .quad 0
