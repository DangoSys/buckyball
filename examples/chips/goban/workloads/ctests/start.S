# Multi-core SPMD startup for Goban.
# All harts jump to main(); divergence is done via mhartid inside main().
# Stack layout: top - hartid * GOBAN_STACK_SIZE.
# On return from main(), hart 0 writes exit code to BBSimHarness MMIO exit
# register (0x60000000); other harts spin forever.

#ifndef GOBAN_STACK_TOP
#define GOBAN_STACK_TOP 0x80200000
#endif

#ifndef GOBAN_STACK_SIZE
#define GOBAN_STACK_SIZE 0x1000
#endif

.section .text.init
.global _start

_start:
    csrr  t0, mhartid           # t0 = hart id

    # Stack per hart: top - hartid * stack_size
    li    sp, GOBAN_STACK_TOP
    li    t1, GOBAN_STACK_SIZE
    mul   t1, t0, t1
    sub   sp, sp, t1

    call  main

    # Only hart 0 writes the MMIO exit register
    csrr  t0, mhartid
    bnez  t0, 1f
    li    t1, 0x60000000
    sw    a0, 0(t1)             # write exit code; C++ mmio_tick() calls sim_exit()
1:  j     1b                   # spin forever (all harts)
