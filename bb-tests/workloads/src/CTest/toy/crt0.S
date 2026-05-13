/* crt0.S — minimal startup for BBSimHarness baremetal
 * Bootrom jumps to 0x80000000 (this code).
 * Sets up stack, clears BSS, then calls main. */
.section .text.init,"ax",@progbits
.global _start
_start:
    /* Enable FPU: set mstatus.FS = 01 (Initial) */
    li   t0, 0x2000          # mstatus.FS = 01 (bit 13)
    csrs mstatus, t0

    /* Set up trap vector (mtvec) to trap_entry */
    la   t0, trap_entry
    csrw mtvec, t0

    /* stack pointer: use a fixed address well above code */
    li   sp, 0x80400000

    /* clear BSS */
    la   a0, __bss_start
    la   a1, __bss_end
bss_loop:
    bgeu a0, a1, bss_done
    sd   zero, 0(a0)
    addi a0, a0, 8
    j    bss_loop
bss_done:

    call main

    /* if main returns, write 0 to sim_exit */
    li   a0, 0x60000000
    sw   zero, 0(a0)
1:  j    1b

/* Trap handler: print mcause and mepc, then exit */
.align 4
trap_entry:
    /* Save context */
    addi sp, sp, -16
    sd   ra, 8(sp)
    sd   a0, 0(sp)

    /* Read mcause and mepc for debugging */
    csrr a0, mcause
    csrr a1, mepc

    /* Exit simulation with error code */
    li   a0, 0x60000000
    li   t0, 1
    sw   t0, 0(a0)

    /* Infinite loop */
1:  j    1b
