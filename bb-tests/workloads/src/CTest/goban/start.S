# Multi-core SPMD startup for Goban (nCores=4 by default).
# All harts jump to main(); divergence is done via mhartid inside main().
# Stack layout: base = 0x80020000, each hart gets 4 KB downward.

.section .text.init
.global _start

_start:
    csrr  t0, mhartid           # t0 = hart id

    # Stack per hart: base - hartid * 0x1000
    li    sp, 0x80020000
    li    t1, 0x1000
    mul   t1, t0, t1
    sub   sp, sp, t1

    call  main

.align 12
stack_space:
    .space 0x8000               # 32 KB total stack space (8 harts × 4 KB)
