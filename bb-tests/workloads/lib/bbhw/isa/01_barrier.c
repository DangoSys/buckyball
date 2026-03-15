#ifndef _BB_BARRIER_H_
#define _BB_BARRIER_H_

#include "isa.h"

#define BB_BARRIER_FUNC7 1

/**
 * bb_barrier() - hardware multi-core barrier synchronization.
 *
 * Semantics:
 *   1. Waits for this core's own instruction ROB to drain (implicit fence).
 *   2. Signals arrive to the tile-level BarrierUnit.
 *   3. Stalls until all nCores cores have arrived (hardware all-reduce).
 *
 * All cores in the same BBTile must call bb_barrier() at the same point.
 * Mixing bb_barrier() with bb_fence() within the same barrier epoch is
 * undefined behaviour.
 */
#define bb_barrier() BUCKYBALL_INSTRUCTION_R_R(0, 0, BB_BARRIER_FUNC7)

#endif // _BB_BARRIER_H_
