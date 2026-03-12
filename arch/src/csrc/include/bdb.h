#ifndef _BDB_H_
#define _BDB_H_

// DPI-C
#include "verilated_dpi.h"
#include "svdpi.h"
// verilator
#include "verilated.h"

#include "VBBSimHarness.h"
#include "verilated_fst_c.h"
#if VM_COVERAGE
#include "verilated_cov.h"
#endif

extern VBBSimHarness *top;

// ================ BDB Config ===================
// Log file path
extern const char *log_path;
// FST file path
extern const char *fst_path;
// UART stdout file path
extern const char *stdout_path;
// Raw stdout fd saved before dup2 — UART writes here for real-time display
extern int raw_stdout_fd;

void init_monitor(int argc, char *argv[]);
void bdb_mainloop();
void ball_exec_once();
void bdb_set_batch_mode();
void sim_exit();

#endif // _BDB_H_
