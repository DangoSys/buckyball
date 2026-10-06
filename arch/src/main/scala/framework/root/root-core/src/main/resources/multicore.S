.option norelax
.option norvc
.macro check_payload
  ld t0, 0(s1)
  li t1, 0x8877665544332211
  bne t0, t1, fail
  ld t0, 32(s1)
  li t1, 0x0102030405060708
  bne t0, t1, fail
  ld t0, 40(s1)
  li t1, 0x1122334455667788
  bne t0, t1, fail
  ld t0, 48(s1)
  li t1, 0x8000000000000001
  bne t0, t1, fail
  ld t0, 56(s1)
  li t1, 0xffeeddccbbaa9988
  bne t0, t1, fail
.endm
.section .text.start,"ax",@progbits
.globl _start
_start:
  li s0, 0x10000000
  la s1, shared
  li s2, 0
  la t0, trap_handler
  csrw mtvec, t0
  csrw mie, zero
  csrr s5, mhartid
  beqz s5, producer
  li t0, 1
  beq s5, t0, consumer
  j fail

.section .text.trap,"ax",@progbits
trap_handler:
  j fail

.section .text.tests,"ax",@progbits
producer:
  li t0, 0x8877665544332211
  sd t0, 0(s1)
  sd zero, 24(s1)
  li t0, 0x0102030405060708
  sd t0, 32(s1)
  li t0, 0x1122334455667788
  sd t0, 40(s1)
  li t0, 0x8000000000000001
  sd t0, 48(s1)
  li t0, 0xffeeddccbbaa9988
  sd t0, 56(s1)
  fence rw, rw
  li t0, 1
  sd t0, 8(s1)
  li s2, 1
  sd s2, 8(s0)
  li s3, 4096
producer_wait:
  ld t0, 16(s1)
  bnez t0, producer_check
  addi s3, s3, -1
  bnez s3, producer_wait
  j fail
producer_check:
  li t1, 1
  bne t0, t1, fail
  fence r, rw
  ld t0, 24(s1)
  bne t0, t1, fail
  check_payload
  j finish
consumer:
  li s3, 4096
consumer_wait:
  ld t0, 8(s1)
  bnez t0, consumer_check
  addi s3, s3, -1
  bnez s3, consumer_wait
  j fail
consumer_check:
  li t1, 1
  bne t0, t1, fail
  fence r, rw
  check_payload
  li s2, 1
  sd s2, 8(s0)
  addi t2, s1, 24
  li t0, 1
  amoadd.d.aqrl t1, t0, (t2)
  bnez t1, fail
  ld t1, 24(s1)
  bne t1, t0, fail
  fence rw, w
  sd t0, 16(s1)
finish:
  li s2, 2
  sd s2, 8(s0)
  sd zero, 0(s0)
1: j 1b
fail:
  ori t0, s2, 0x100
  sd t0, 0(s0)
2: j 2b

.section .data,"aw",@progbits
.balign 64
shared:
  .dword 0, 0, 0, 0, 0, 0, 0, 0
