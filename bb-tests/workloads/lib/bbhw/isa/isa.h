#ifndef BUCKYBALL_ISA_H
#define BUCKYBALL_ISA_H

#include <bb_isa.h>
#include <stddef.h>
#include <stdint.h>

// Data type for matrix elements
typedef int8_t elem_t;
typedef int32_t result_t;

// Base (mem/frontend) instructions, callable from C. Encodings are in
// bb_isa.h next to this header; Ball-specific ISA macros live under
// <ball>/workloads/isa/ of each Ball component.
#include "01_barrier.c"
#include "12_kernel.c"
#include "13_mvover.c"
#include "16_mvout.c"
#include "32_mset.c"
#include "33_mvin.c"
#include "34_mvin_2d.c"
#include "35_mvin_mmio.c"

#endif // BUCKYBALL_ISA_H
