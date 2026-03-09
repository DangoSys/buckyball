#ifndef MONITOR_TRACE_H_
#define MONITOR_TRACE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // DPI-C function for instruction trace (itrace)
    // Called from GlobalROB when instructions are issued or completed
    void dpi_itrace(unsigned char is_issue, // 1 = issue, 0 = complete
                    unsigned int rob_id, unsigned int domain_id, unsigned int funct,
                    unsigned long long rs1, unsigned long long rs2);

    // DPI-C function for memory trace (mtrace)
    // Called from MemBackend when read/write requests are made
    void dpi_mtrace(unsigned char is_write,  // 1 = write, 0 = read
                    unsigned char is_shared, // 1 = shared path, 0 = private path
                    unsigned int channel, unsigned long long hart_id,
                    unsigned int vbank_id,
                    unsigned int group_id, unsigned int addr,
                    unsigned long long data_lo, unsigned long long data_hi);

    // DPI-C function for Ball PMC trace (pmctrace)
    // Called from BallCyclePMC when a Ball completes a task
    void dpi_pmctrace(unsigned int ball_id, unsigned int rob_id,
                      unsigned long long elapsed);

    // DPI-C function for memory PMC trace (pmctrace)
    // Called from MemCyclePMC when a load/store completes
    void dpi_mem_pmctrace(unsigned char is_store, // 1 = store, 0 = load
                          unsigned int rob_id, unsigned long long elapsed);

#ifdef __cplusplus
}
#endif

#endif // MONITOR_TRACE_H_
