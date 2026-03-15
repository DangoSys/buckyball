#ifndef MONITOR_TRACE_H_
#define MONITOR_TRACE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// DPI-C function for instruction trace (itrace)
// Called from GlobalROB when instructions are issued or completed
void dpi_itrace(unsigned char is_issue, // 1 = issue, 0 = complete
                unsigned int rob_id, unsigned int domain_id, unsigned int funct,
                unsigned long long rs1, unsigned long long rs2,
                unsigned char bank_enable);

// DPI-C function for memory trace (mtrace)
// Called from MemBackend when read/write requests are made
void dpi_mtrace(unsigned char is_write, // 1 = write, 0 = read
                unsigned char is_shared, unsigned int channel,
                unsigned long long hart_id, unsigned int vbank_id,
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

// DPI-C function for cycle counter trace (ctrace)
// Called from TraceBall when counter commands are executed
void dpi_ctrace(unsigned char subcmd, // 0=START, 1=STOP, 2=READ
                unsigned int ctr_id, unsigned long long tag,
                unsigned long long elapsed, unsigned long long cycle);

// DPI-C functions for bank backdoor (TraceBall)
// RTL calls these to get parameters from C++ testbench
unsigned long long dpi_backdoor_get_read_addr(void);
unsigned long long dpi_backdoor_get_write_addr(void);
void dpi_backdoor_get_write_data(unsigned long long *data_lo,
                                 unsigned long long *data_hi);
void dpi_backdoor_put_read_data(unsigned int bank_id, unsigned int row,
                                unsigned long long data_lo,
                                unsigned long long data_hi);
void dpi_backdoor_put_write_done(unsigned int bank_id, unsigned int row,
                                 unsigned long long data_lo,
                                 unsigned long long data_hi);

#ifdef __cplusplus
}
#endif

#endif // MONITOR_TRACE_H_
