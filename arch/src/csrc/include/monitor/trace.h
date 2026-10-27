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
                unsigned int thread_id_lo, unsigned int thread_id_hi,
                unsigned int rob_id, unsigned int inst_id_lo,
                unsigned int inst_id_hi, unsigned int domain_id,
                unsigned int funct, unsigned int pc_lo, unsigned int pc_hi,
                unsigned int rs1_lo, unsigned int rs1_hi, unsigned int rs2_lo,
                unsigned int rs2_hi);

// Passive Ant retirement observation from the thread trace collector.
void dpi_ant_trace(unsigned int thread_lo, unsigned int thread_hi,
                   unsigned int pc_lo, unsigned int pc_hi,
                   unsigned int instruction, unsigned int rd,
                   unsigned int data_lo, unsigned int data_hi);

// DPI-C function for memory trace (mtrace)
// Called from MemBackend when read/write requests are made
void dpi_mtrace(unsigned int is_write, unsigned int is_shared,
                unsigned int channel, unsigned int thread_id_lo,
                unsigned int thread_id_hi, unsigned int rob_id,
                unsigned int inst_id_lo, unsigned int inst_id_hi,
                unsigned int vbank_id, unsigned int pbank_id,
                unsigned int group_id, unsigned int addr,
                unsigned int write_mask, unsigned int data_lo_lo,
                unsigned int data_lo_hi, unsigned int data_hi_lo,
                unsigned int data_hi_hi);

// DPI-C function for Ball PMC trace (pmctrace)
// Called from BallCyclePMC when a Ball completes a task
void dpi_pmctrace(unsigned int ball_id, unsigned int rob_id,
                  unsigned int elapsed_lo, unsigned int elapsed_hi);

// DPI-C function for memory PMC trace (pmctrace)
// Called from MemCyclePMC when a load/store completes
void dpi_mem_pmctrace(unsigned char is_store, // 1 = store, 0 = load
                      unsigned int rob_id, unsigned long long elapsed);

// DPI-C function for cycle counter trace (ctrace)
// Called from TraceBall when counter commands are executed
void dpi_ctrace(unsigned int subcmd, // 0=START, 1=STOP, 2=READ
                unsigned int ctr_id, unsigned int tag_lo, unsigned int tag_hi,
                unsigned int elapsed_lo, unsigned int elapsed_hi,
                unsigned int cycle_lo, unsigned int cycle_hi);

// DPI-C function for bank hash trace (banktrace)
// Called when a Ball instruction writes a bank
void dpi_btrace(unsigned int inst_id_lo, unsigned int inst_id_hi,
                unsigned int thread_id_lo, unsigned int thread_id_hi,
                unsigned int owner_thread_id_lo,
                unsigned int owner_thread_id_hi, unsigned int w0_vbank,
                unsigned int w0_hash);

#ifdef __cplusplus
}
#endif

#endif // MONITOR_TRACE_H_
