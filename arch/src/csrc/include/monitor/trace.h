#ifndef MONITOR_TRACE_H_
#define MONITOR_TRACE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Harness clock cycle index (BBSimHarness posedge); must run before other trace
// DPI in same eval.
void dpi_bdb_set_clk(unsigned long long c);

// DPI-C function for instruction trace (itrace)
// Called from GlobalROB when instructions are allocated/issued/completed
void dpi_itrace(unsigned int is_issue, // 2 = alloc, 1 = issue, 0 = complete
                unsigned int hart_id_lo, unsigned int hart_id_hi,
                unsigned int rob_id, unsigned int inst_id_lo,
                unsigned int inst_id_hi, unsigned int domain_id,
                unsigned int funct, unsigned int pc_lo, unsigned int pc_hi,
                unsigned int rs1_lo, unsigned int rs1_hi, unsigned int rs2_lo,
                unsigned int rs2_hi);

// DPI-C function for memory trace (mtrace)
// Called from MemBackend when read/write requests are made
void dpi_mtrace(unsigned char is_write, // 1 = write, 0 = read
                unsigned char is_shared, unsigned int channel,
                unsigned long long hart_id, unsigned int vbank_id,
                unsigned int pbank_id, unsigned int group_id, unsigned int addr,
                unsigned long long data_lo, unsigned long long data_hi);

// DPI-C function for Ball PMC trace (pmctrace)
// Called from BallCyclePMC when a Ball completes a task
void dpi_pmctrace(unsigned int ball_id, unsigned int rob_id,
                  unsigned long long elapsed);

// DPI-C function for memory PMC trace (pmctrace)
// Called from MemCyclePMC when a load/store completes
void dpi_mem_pmctrace(unsigned char is_store, // 1 = store, 0 = load
                      unsigned int rob_id, unsigned long long elapsed);

// DPI-C function for cycle counter trace (ctrace)
// Called from TraceBall when counter commands are executed
void dpi_ctrace(unsigned char subcmd, // 0=START, 1=STOP, 2=READ
                unsigned int ctr_id, unsigned long long tag,
                unsigned long long elapsed, unsigned long long cycle);

#ifdef __cplusplus
}
#endif

#endif // MONITOR_TRACE_H_
