.section .text.init
.global _start

#include <topology.h>

_start:
    # Get hart ID
    csrr t0, mhartid

    # Set up stack pointer for each hart
    # Each hart gets 4KB stack space (0x1000 bytes)
    li sp, 0x90000000
    li t1, 0x1000
    mul t1, t0, t1
    sub sp, sp, t1
    # Jump to main for all harts; the workload dispatches by core id.
    call main

    # The boot hart reports the workload result. Other harts remain parked.
    csrr t0, mhartid
    bnez t0, 1f
    li t1, 0x60000000
    sw a0, 0(t1)
1:
    wfi
    j 1b
