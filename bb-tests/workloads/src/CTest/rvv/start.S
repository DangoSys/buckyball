.section .text.init
.global _start

_start:
    # Get hart ID
    csrr t0, mhartid

    # Set up stack pointer for each hart
    # Each hart gets 4KB stack space (0x1000 bytes)
    li sp, 0x80020000
    li t1, 0x1000
    mul t1, t0, t1
    sub sp, sp, t1
    # Jump to main for all harts (let MLIR code handle hart filtering)
    call main



.align 12
stack_space:
    # 32KB stack space
    .space 0x8000
