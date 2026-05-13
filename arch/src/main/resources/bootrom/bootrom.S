# Buckyball Bootrom
# RISC-V 64-bit boot code for multi-core initialization

.section .text
.globl _start

# Reset vector entry point
_start:
    # Setup M-mode trap vector to 0x40
    auipc   a0, 0x0
    addi    a0, a0, 64
    csrw    mtvec, a0

    # Check if supervisor mode is supported (test 'S' bit in misa)
    csrr    t0, misa
    srai    t0, t0, 0x12
    andi    t0, t0, 1
    beqz    t0, 1f

    # Configure supervisor mode interrupts
    csrw    mideleg, zero
    li      a0, 546
    csrc    mip, a0

1:
    # Enable machine software interrupt only
    # MSIP (bit 3 = 0x8)
    li      a0, 0x8
    csrw    mie, a0
    csrs    mstatus, a0

2:
    # Wait for interrupt loop
    wfi
    j       2b

    .align  4

# Trap entry / Multi-core boot synchronization
trap_entry:
    # Load sync memory base (0x2000000)
    lui     a1, 0x2000

    # Get hart ID
    csrr    a0, mhartid
    beqz    a0, 1f          # Hart 0 goes to boot sequence
    j       2f              # Other harts wait

1:
    # Hart 0: Initialize sync array
    addi    a2, a1, 4
    li      a3, 1
3:
    sw      a3, 0(a2)
    addi    a2, a2, 4
    lw      a3, -4(a2)
    bnez    a3, 3b
    j       4f

    # Padding: 20 bytes of zeros (0x6c - 0x7f)
    .fill   20, 1, 0

    # Fallback: jump back to reset
    j       _start

2:
    # Secondary harts: wait for hart 0
    lw      t0, 0(a1)
    bnez    t0, 2b

    # Calculate this hart's sync slot
    slli    a0, a0, 0x2
    add     a1, a0, a1

4:
    # Common boot path: clear sync flag
    sw      zero, 0(a1)

    # Setup kernel entry point (0x80000000)
    addiw   a0, zero, 1
    slli    a0, a0, 0x1f
    csrw    mepc, a0

    # Prepare kernel arguments (hartid, dtb_addr)
    csrr    a0, mhartid
    auipc   a1, 0x0
    addi    a1, a1, 24

    # Disable interrupts before jumping to kernel
    li      a2, 128
    csrc    mstatus, a2

    # Jump to kernel
    mret
    nop
