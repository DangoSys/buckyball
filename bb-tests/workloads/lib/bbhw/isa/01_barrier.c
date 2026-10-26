#ifndef _BB_BARRIER_H_
#define _BB_BARRIER_H_

#include "isa.h"

#define BB_BARRIER_FUNC7 1

/**
 * bb_barrier() - hardware multi-thread barrier synchronization.
 *
 * Semantics:
 *   1. Waits for this thread's own instruction ROB to drain.
 *   2. Signals arrive to the core-level BarrierUnit.
 *   3. Stalls until all nThreads threads have arrived (hardware all-reduce).
 *
 * All threads in the same BBCore must call bb_barrier() at the same point.
 */
#define bb_barrier() BUCKYBALL_INSTRUCTION_R_R(0, 0, BB_BARRIER_FUNC7)

#endif // _BB_BARRIER_H_
