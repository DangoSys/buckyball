/* crt0.S — minimal startup for BBSimHarness baremetal
 * Bootrom jumps to 0x80000000 (this code).
 * Sets up stack and TLS, clears BSS, then calls main. */
.section .text.init,"ax",@progbits
.global _start
_start:
    /* Establish interrupt state before any read-modify-write CSR operation. */
    csrci mstatus, 8
    /* Smrnmi resets mnstatus.NMIE to 0, which disables every interrupt until
     * software sets it. Set it where mnstatus (0x744) exists: on a core
     * without it the write traps to mtvec, the next instruction. */
    la t0, 1f
    csrw mtvec, t0
    li t0, 8
    csrs 0x744, t0
    .balign 4
1:
    /* Smdbltrp resets mstatus.MDT to 1 (and a trap into M-mode sets it), and
     * a write of mstatus with MDT set clears MIE, so M-mode could never enable
     * interrupts. Clear MDT (bit 42); without Smdbltrp it is read-only zero. */
    li t0, 1
    slli t0, t0, 42
    csrc mstatus, t0
    csrw mie, zero
    csrw mip, zero
    csrw mideleg, zero
    csrw medeleg, zero

    /* The generated target profile selects its test hart before boot. */
    csrr t0, mhartid
    li t1, HART_HART_ID
    bne t0, t1, park_hart

    /* Enable FPU: set mstatus.FS and XS = 01 (Initial) */
    li   t0, 0xa000          # mstatus.FS = 01 (bit 13)
    csrs mstatus, t0

    /* Set up trap vector (mtvec) to trap_entry */
    la   t0, trap_entry
    csrw mtvec, t0

    /* stack pointer: top of the 256 MiB BBSim DRAM window */
    li   sp, 0x90000000

    .option push
    .option norelax
    la   tp, __tls_start
    .option pop

    /* clear BSS */
    la   a0, __bss_start
    la   a1, __bss_end
bss_loop:
    bgeu a0, a1, bss_done
    sd   zero, 0(a0)
    addi a0, a0, 8
    j    bss_loop
bss_done:

    li a0, 0
    li a1, 0
    call main

    /* exit flushes stdio, then _exit writes the code to this hart's sim_exit */
    call exit

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
    csrr t1, mhartid
    slli t1, t1, 18
    add  a0, a0, t1
    li   t0, 1
    sw   t0, 0(a0)

    /* Infinite loop */
1:  j    1b

park_hart:
    wfi
    j park_hart
