.section .text.init
.global _start

#include <topology.h>

_start:
    # Get hart ID
    csrr t0, mhartid

    # Enable the FPU on every hart: mstatus.FS = Initial
    li t1, 0x2000
    csrs mstatus, t1

    # Set up stack pointer for each hart
    # Each hart gets 4KB stack space (0x1000 bytes)
    li sp, 0x90000000
    li t1, 0x1000
    mul t1, t0, t1
    sub sp, sp, t1
    # Each hart owns a TLS copy at the top of its 4KB stack slot.
    la t2, __tls_start
    la t3, __tdata_end
    la t4, __tls_end
    sub t5, t4, t2
    addi t5, t5, 15
    andi t5, t5, -16
    sub sp, sp, t5
    mv tp, sp
    mv t6, tp
.Lcopy_tls:
    bgeu t2, t3, .Lzero_tls
    lbu a0, 0(t2)
    sb a0, 0(t6)
    addi t2, t2, 1
    addi t6, t6, 1
    j .Lcopy_tls
.Lzero_tls:
    bgeu t2, t4, .Lrun_main
    sb zero, 0(t6)
    addi t2, t2, 1
    addi t6, t6, 1
    j .Lzero_tls
.Lrun_main:
    # Loaders write only the file image, so .bss can hold a previous run's values on P2E DDR.
    # The reporting hart clears it, then releases the others through the last flag write.
    csrr t0, mhartid
    la t1, .Lbss_ready
    li t2, BB_REPORT_HART
    bne t0, t2, .Lwait_bss
    la t2, __bss_start
    la t3, __bss_end
    addi t4, t3, -63
.Lclear_bss_blocks:
    bgeu t2, t4, .Lclear_bss
    sd zero, 0(t2)
    sd zero, 8(t2)
    sd zero, 16(t2)
    sd zero, 24(t2)
    sd zero, 32(t2)
    sd zero, 40(t2)
    sd zero, 48(t2)
    sd zero, 56(t2)
    addi t2, t2, 64
    j .Lclear_bss_blocks
.Lclear_bss:
    bgeu t2, t3, .Lbss_cleared
    sd zero, 0(t2)
    addi t2, t2, 8
    j .Lclear_bss
.Lbss_cleared:
    li t2, 1
    fence rw, w
    sw t2, 0(t1)
    j .Lcall_main
.Lwait_bss:
    lw t2, 0(t1)
    beqz t2, .Lwait_bss
    fence r, rw
.Lcall_main:
    # Jump to main for all harts; the workload dispatches by core id.
    li a0, 0
    li a1, 0
    call main

    # Core 0 of the work tile reports the workload result. Other harts remain parked.
    csrr t0, mhartid
    li t1, BB_REPORT_HART
    bne t0, t1, 1f
    li t1, 0x60000000
    sw a0, 0(t1)
1:
    wfi
    j 1b

# Outside .bss so the clear cannot reset it; a stale 1 from a previous run is
# overwritten by the image load because .data is part of the file image.
.section .data
.balign 4
.Lbss_ready:
    .word 0
