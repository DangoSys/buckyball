.option norelax
.option norvc

.macro expect_trap cause, value
  li s9, \cause
  li s10, \value
.endm

.macro check_trap failure
  li t0, -1
  bne s9, t0, \failure
.endm

.section .text.start,"ax",@progbits
.globl _start
_start:
  li s0, 0x10000000
  li s2, 1
  li s9, -1
  li s11, 0
  la t0, trap_handler
  csrw mtvec, t0
  csrw medeleg, zero
  csrw mideleg, zero
  csrw mie, zero
  // The standalone Core has no RoCC accelerator; TIME must still be readable.
  csrr t0, time
  beqz t0, fail
  li t0, 2
  csrw mcounteren, t0
  csrw scounteren, t0
  li t0, 0x20002800
  csrw pmpaddr0, t0
  li t0, 0x20002803
  csrw pmpaddr1, t0
  li t0, -1
  csrw pmpaddr2, t0
  li t0, 0x1f191d
  csrw pmpcfg0, t0
  li t0, 0x8000000000080010
  csrw satp, t0
  sfence.vma

  // MPRV=S translates data while machine instruction fetch remains physical.
  li t0, 0x21800
  csrc mstatus, t0
  li t0, 0x20800
  csrs mstatus, t0
  li t1, 0x40008000
  ld t2, 0(t1)
  li t0, 0x20000
  csrc mstatus, t0
  li t0, 0x1122334455667788
  bne t2, t0, fail
  sd s2, 8(s0)
  li t0, 0x40004000
  csrw mepc, t0
  mret

.section .text.trap,"ax",@progbits
trap_handler:
  csrr t3, mcause
  bne t3, s9, fail
  csrr t0, mtval
  bne t0, s10, fail
  addi s11, s11, 1
  li s9, -1
  li t1, 1
  beq t3, t1, return_boundary
  li t1, 8
  beq t3, t1, return_supervisor
  li t1, 9
  beq t3, t1, finish
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
return_boundary:
  la t0, boundary_return
  li t1, 0x40000000
  sub t0, t0, t1
  csrw mepc, t0
  mret
return_supervisor:
  sd s2, 8(s0)
  // U ECALL returns through M to an explicit S continuation.
  li t0, 0x1800
  csrc mstatus, t0
  li t0, 0x800
  csrs mstatus, t0
  la t0, supervisor_return
  li t1, 0x40000000
  sub t0, t0, t1
  csrw mepc, t0
  mret
finish:
  li t0, 6
  bne s11, t0, fail
  li s2, 3
  sd s2, 8(s0)
  sd zero, 0(s0)
1: j 1b
fail:
  ori t0, s2, 0x100
  sd t0, 0(s0)
2: j 2b

.section .text.supervisor,"ax",@progbits
supervisor:
  csrr t0, time
  beqz t0, supervisor_fail
  li s2, 2
  li s1, 0x40008000
  li t0, 0x8877665544332211
  sd t0, 0(s1)
  ld t1, 0(s1)
  bne t0, t1, supervisor_fail
  // Exercise both fetch and data through a canonical upper-half Sv39 alias.
  li s3, 0xffffffffc0008000
  la t0, high_alias
  li t1, 0xffffffff80000000
  add t0, t0, t1
  jalr ra, 0(t0)
  // The allowed RX range is exactly one 8-byte fetch word.
  li s4, 0
  li t0, 0x4000a000
  jalr ra, 0(t0)
  li t0, 1
  bne s4, t0, supervisor_fail
  expect_trap 1, 0x4000a008
  li t0, 0x4000a008
  jr t0
boundary_return:
  check_trap supervisor_fail
  // Change the actual leaf via a writable page-table alias, then fence.
  li t0, 0x40012040
  li t1, 0x200020c3
  sd t1, 0(t0)
  sfence.vma s1, zero
  expect_trap 15, 0x40008000
trap_store:
  sd zero, 0(s1)
  check_trap supervisor_fail
  ld t0, 0(s1)
  li t1, 0x8877665544332211
  bne t0, t1, supervisor_fail
  // Supervisor cannot read a user page with SUM clear.
  expect_trap 13, 0x40006000
  li t0, 0x40006000
trap_sum:
  ld t1, 0(t0)
  check_trap supervisor_fail
  li t0, 0x40006000
  csrw sepc, t0
  li t0, 0x100
  csrc sstatus, t0
  sret
supervisor_return:
  check_trap supervisor_fail
  expect_trap 9, 0
trap_supervisor_ecall:
  ecall
3: j 3b
supervisor_fail:
  ebreak
  j supervisor_fail
high_alias:
  ld t1, 0(s3)
  li t2, 0x8877665544332211
  bne t1, t2, supervisor_fail
  ret

.section .text.user,"ax",@progbits
user:
  csrr t0, time
  beqz t0, user_fail
  // User access to a supervisor-only leaf must fault.
  expect_trap 13, 0x40008000
  li t0, 0x40008000
trap_user_load:
  ld t1, 0(t0)
  check_trap user_fail
  expect_trap 8, 0
trap_user_ecall:
  ecall
4: j 4b
user_fail:
  ebreak
  j user_fail

.section .data,"aw",@progbits
  .dword 0x1122334455667788

.section .text.boundary,"ax",@progbits
  addi s4, s4, 1
  ret
trap_boundary:
  nop

.section .pagetables,"aw",@progbits
.balign 4096
root:
  .dword 0
  .dword 0x20004401
  .zero 4072
  .dword 0x20004401
middle:
  .dword 0x20004801
  .zero 4088
leaf:
  .zero 32
  .dword 0x200010cb
  .dword 0
  .dword 0x200018db
  .dword 0
  .dword 0x200020c7
  .dword 0
  .dword 0x200028cb
  .zero 56
  .dword 0x200048c7
  .zero 3944
