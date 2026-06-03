# Buckyball Linux BootROM.

.section .text.hang, "ax", @progbits
.globl _hang
_hang:
  j       _start

.section .text.start, "ax", @progbits
.globl _start
_start:
  la      t0, trap_entry
  csrw    mtvec, t0

  csrw    mie, zero
  li      t0, 0x8
  csrc    mstatus, t0

  li      t0, 0x1800
  csrs    mstatus, t0

  li      t0, 0x80000000
  csrw    mepc, t0

  csrr    a0, mhartid
  la      a1, _dtb
  mret

trap_entry:
  j       trap_entry

  .balign 8
