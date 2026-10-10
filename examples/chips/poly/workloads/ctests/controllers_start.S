#include <topology.h>
/* Only Linux control cores participate; data-plane workers remain parked. */
.section .text.init,"ax",@progbits
.globl _start
_start:
  csrr t0, mhartid
  li t1, BB_MAIN_CORES + BB_COMPUTE_TILES
  bltu t0, t1, controllers_start
1:
  wfi
  j 1b
#undef BB_REPORT_HART
#define BB_REPORT_HART 0
#define _start controllers_start
#include "start.S"
