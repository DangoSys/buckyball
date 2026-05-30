# Buckyball Linux BootROM
# Jumps to OpenSBI fw_payload at 0x80000000
# Passes hartid in a0, DTB address in a1

.section .text.hang, "ax", @progbits
.globl _hang
_hang:
    # Reset vector: setup trap handler and wait for interrupt
    la      a0, _start
    csrw    mtvec, a0

    # Check if supervisor mode supported
    csrr    t0, misa
    srai    t0, t0, 0x12
    andi    t0, t0, 1
    beqz    t0, 1f

    # Clear delegation and spurious interrupts
    csrw    mideleg, zero
    li      a0, 0x222
    csrc    mip, a0

1:
    # Enable MSIP interrupt
    li      a0, 0x8
    csrw    mie, a0
    csrs    mstatus, a0

wfi_loop:
    wfi
    j       wfi_loop

.section .text.start, "ax", @progbits
.globl _start
_start:
    # Multi-core sync via CLINT at 0x2000000
    li      a1, 0x2000000
    csrr    a0, mhartid
    bnez    a0, 1f

    # Hart 0: trigger others
    addi    a2, a1, 4
    li      a3, 1
2:  sw      a3, 0(a2)
    addi    a2, a2, 4
    lw      a3, -4(a2)
    bnez    a3, 2b
    j       3f

    # Other harts: wait for hart 0
1:  lw      t0, 0(a1)
    bnez    t0, 1b
    slli    a0, a0, 2
    add     a1, a0, a1

    # Clear MSIP and jump to OpenSBI
3:  sw      zero, 0(a1)
    li      a0, 0x80000000
    csrw    mepc, a0
    csrr    a0, mhartid
    la      a1, _dtb
    li      a2, 0x80
    csrc    mstatus, a2

# Print "BOOTEND" to the console
    li      t5, 0x60020000
    li      t0, 'B'
    sb      t0, 0(t5)
    li      t0, 'O'
    sb      t0, 0(t5)
    sb      t0, 0(t5)
    li      t0, 'T'
    sb      t0, 0(t5)
    li      t0, 'E'
    sb      t0, 0(t5)
    li      t0, 'N'
    sb      t0, 0(t5)
    li      t0, 'D'
    sb      t0, 0(t5)
    li      t0, '\n'
    sb      t0, 0(t5)

    mret
