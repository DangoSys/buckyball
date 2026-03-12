/* crt0.S — minimal startup for BBSimHarness baremetal
 * Bootrom jumps to 0x80000000 (this code).
 * Sets up stack, clears BSS, then calls main. */
.section .text.init,"ax",@progbits
.global _start
_start:
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
