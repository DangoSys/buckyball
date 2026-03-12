#include "pegasus.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// Assert DUT reset (high), hold for 10ms, then deassert (low)
void scu_reset(PegasusXDMA *xdma) {
    mmio_write32(xdma, SCU_RESET_OFFSET, 1);   // assert reset
    usleep(10000);                              // hold 10ms
    mmio_write32(xdma, SCU_RESET_OFFSET, 0);   // deassert reset
    usleep(1000);                               // settle 1ms
}

// Start free-running DUT clock: CTRL[0]=1, CTRL[1]=0
void scu_run(PegasusXDMA *xdma) {
    mmio_write32(xdma, SCU_CTRL_OFFSET, 0x1);
}

// Halt DUT clock: CTRL[0]=0
void scu_halt(PegasusXDMA *xdma) {
    mmio_write32(xdma, SCU_CTRL_OFFSET, 0x0);
}

// Single-step: write STEP_N (enters step mode automatically), wait for step_done
void scu_step(PegasusXDMA *xdma, uint32_t n_cycles) {
    if (n_cycles == 0) return;
    // Write STEP_N: SCU enters step mode, starts counting
    mmio_write32(xdma, SCU_STEP_N_OFFSET, n_cycles);
    // Poll STATUS[1] (step_done) until set
    while (true) {
        uint32_t status = mmio_read32(xdma, SCU_STATUS_OFFSET);
        if (status & 0x2) break;  // step_done set
        usleep(100);
    }
}

// Read current 64-bit DUT cycle counter
uint64_t scu_read_cycles(PegasusXDMA *xdma) {
    // Read low first, then high — if high changes between reads,
    // re-read until consistent (handles 32→64 bit rollover)
    uint32_t lo, hi, hi2;
    do {
        hi  = mmio_read32(xdma, SCU_CYCLE_HI_OFFSET);
        lo  = mmio_read32(xdma, SCU_CYCLE_LO_OFFSET);
        hi2 = mmio_read32(xdma, SCU_CYCLE_HI_OFFSET);
    } while (hi != hi2);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
